import io
import re
import json
import base64
from typing import List, Tuple

import streamlit as st
from PyPDF2 import PdfReader
import fitz  # PyMuPDF
from google.cloud import storage
from google.oauth2 import service_account
import google.generativeai as genai
from tenacity import retry, stop_after_attempt, wait_exponential

# =========================
# 0. 기본 설정
# =========================
st.set_page_config(page_title="PDF 페이지별 3줄 요약 (Gemini 2.0 Flash)", layout="wide")
st.title("📄 PDF 업로드 → 페이지별 3줄 요약 (gemini-2.0-flash@001)")

# =========================
# 1. 시크릿 체크
# =========================
missing = []
gemini_key   = st.secrets.get("GEMINI_API_KEY")   or missing.append("GEMINI_API_KEY")
gcs_b64      = st.secrets.get("GCS_SA_KEY_B64")   or missing.append("GCS_SA_KEY_B64")
bucket_name  = st.secrets.get("GCS_BUCKET_NAME")  or missing.append("GCS_BUCKET_NAME")

if missing:
    st.error(f"Secrets에 {', '.join(missing)} 가 없습니다. Manage app → Settings → Secrets 에 등록하세요.")
    st.stop()

# =========================
# 2. GCS 클라이언트
# =========================
gcs_info   = json.loads(base64.b64decode(gcs_b64))
gcs_creds  = service_account.Credentials.from_service_account_info(gcs_info)
gcs_client = storage.Client(credentials=gcs_creds, project=gcs_info["project_id"])
bucket     = gcs_client.bucket(bucket_name)

def list_pdfs() -> List[str]:
    return [
        b.name.split("/", 1)[1]
        for b in gcs_client.list_blobs(bucket_name, prefix="pdfs/")
        if b.name.endswith(".pdf")
    ]

def upload_pdf(name: str, data: bytes):
    bucket.blob(f"pdfs/{name}").upload_from_file(io.BytesIO(data), content_type="application/pdf")

def download_pdf(name: str) -> bytes:
    return bucket.blob(f"pdfs/{name}").download_as_bytes()

# =========================
# 3. Gemini 설정
# =========================
genai.configure(api_key=gemini_key)
MODEL_ID = "gemini-2.0-flash@001"
model    = genai.GenerativeModel(MODEL_ID)

@retry(reraise=True, stop=stop_after_attempt(4), wait=wait_exponential(min=1, max=6))
def gemini_text(prompt: str) -> str:
    resp = model.generate_content(
        prompt,
        generation_config={"temperature": 0.3, "max_output_tokens": 256},
    )
    return (resp.text or "").strip()

@retry(reraise=True, stop=stop_after_attempt(4), wait=wait_exponential(min=1, max=6))
def gemini_image(prompt: str, image_bytes: bytes) -> str:
    resp = model.generate_content(
        [prompt, {"mime_type": "image/png", "data": image_bytes}],
        generation_config={"temperature": 0.3, "max_output_tokens": 512},
    )
    return (resp.text or "").strip()

# =========================
# 4. PDF 처리 함수들
# =========================
def extract_pages_pypdf2(pdf_bytes: bytes) -> List[str]:
    pages = []
    reader = PdfReader(io.BytesIO(pdf_bytes))
    for p in reader.pages:
        t = p.extract_text() or ""
        pages.append(t.strip())
    return pages

def extract_pages_pymupdf(pdf_bytes: bytes) -> List[str]:
    pages = []
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        for page in doc:
            text = page.get_text("text")
            pages.append(text.strip())
    return pages

def render_page_png(pdf_bytes: bytes, page_index: int, dpi: int = 150) -> bytes:
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        page = doc[page_index]
        pix  = page.get_pixmap(dpi=dpi)
        return pix.tobytes("png")

def summarize_pages(pdf_bytes: bytes) -> List[Tuple[int, str, str]]:
    """
    return: [(page_no, preview_text_or_note, summary)]
    """
    # 1차: PyPDF2
    pages = extract_pages_pypdf2(pdf_bytes)
    # 2차: 비었으면 PyMuPDF
    if all(len(p) < 50 for p in pages):
        pages = extract_pages_pymupdf(pdf_bytes)

    results = []
    for idx, page_text in enumerate(pages, 1):
        if page_text and len(page_text) > 50:
            clipped = page_text[:2000]
            prompt = (
                f"다음은 PDF {idx}페이지 기사(들)입니다.\n"
                f"각 기사(문단)별로 핵심만 뽑아 **3줄**로 요약해 주세요.\n"
                f"- 각 줄은 '-'로 시작하는 bullet 형식\n"
                f"- 수치, 기관/회사명, 정책명 등은 그대로 남기기\n\n"
                f"{clipped}"
            )
            try:
                summary = gemini_text(prompt)
            except Exception as e:
                summary = f"요약 실패(텍스트): {e}"
            preview = page_text[:400].replace("\n", " ")
            results.append((idx, preview, summary))
        else:
            # 텍스트 추출 실패 → 이미지로 멀티모달 요약
            try:
                img_bytes = render_page_png(pdf_bytes, idx - 1)
                prompt = (
                    f"아래는 PDF {idx}페이지 이미지입니다.\n"
                    f"이미지 안의 기사(문단)별 핵심을 3줄씩 '-' bullet로 요약해 주세요.\n"
                    f"- 수치/기관명 유지\n"
                )
                summary = gemini_image(prompt, img_bytes)
                preview = "(텍스트 추출 실패 → 이미지로 요약)"
            except Exception as e:
                summary = f"텍스트/이미지 추출 모두 실패: {e}"
                preview = "(추출 실패)"
            results.append((idx, preview, summary))
    return results

# =========================
# 5. UI: 업로드
# =========================
st.sidebar.header("📤 PDF 업로드")
uploaded = st.sidebar.file_uploader("PDF 파일 선택", type="pdf")
if uploaded:
    upload_pdf(uploaded.name, uploaded.read())
    st.sidebar.success(f"✅ GCS에 저장됨: {uploaded.name}")

# =========================
# 6. UI: 목록 & 페이지별 요약
# =========================
st.header("📑 저장된 PDF → 페이지별 3줄 요약")
pdfs = list_pdfs()

if not pdfs:
    st.info("GCS 버킷의 pdfs/ 폴더에 PDF를 업로드해 보세요.")
else:
    for name in sorted(pdfs):
        st.subheader(name)
        if st.button(f"요약: {name}", key=name):
            data = download_pdf(name)
            with st.spinner("페이지별 요약 생성 중…"):
                items = summarize_pages(data)

            if not items:
                st.warning("추출된 내용이 없습니다.")
            else:
                for page_no, preview, summ in items:
                    with st.expander(f"📄 {page_no} 페이지 (미리보기)"):
                        st.text(preview)
                    st.markdown("**📝 3줄 요약**")
                    st.text_area(f"p{page_no} 요약", summ, height=180)
                    st.markdown("---")
