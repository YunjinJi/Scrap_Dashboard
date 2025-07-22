import io
import json
import base64
from typing import List

import streamlit as st
from PyPDF2 import PdfReader
from google.cloud import storage
from google.oauth2 import service_account
import google.generativeai as genai
from tenacity import retry, stop_after_attempt, wait_exponential

# ─────────────────────────────────────────
# 0. 기본 설정
# ─────────────────────────────────────────
st.set_page_config(page_title="PDF 페이지별 3줄 요약 (Gemini 2.0 Flash)", layout="wide")
st.title("📄 PDF 업로드 → 페이지별 3줄 요약 (gemini-2.0-flash@001)")

# ─────────────────────────────────────────
# 1. 시크릿 체크
# ─────────────────────────────────────────
missing = []
gemini_key   = st.secrets.get("GEMINI_API_KEY")   or missing.append("GEMINI_API_KEY")
gcs_b64      = st.secrets.get("GCS_SA_KEY_B64")   or missing.append("GCS_SA_KEY_B64")
bucket_name  = st.secrets.get("GCS_BUCKET_NAME")  or missing.append("GCS_BUCKET_NAME")

if missing:
    st.error(f"Secrets에 {', '.join(missing)} 가 없습니다. Manage app → Settings → Secrets 에 등록하세요.")
    st.stop()

# ─────────────────────────────────────────
# 2. GCS 클라이언트
# ─────────────────────────────────────────
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

# ─────────────────────────────────────────
# 3. Gemini 설정
# ─────────────────────────────────────────
genai.configure(api_key=gemini_key)
MODEL_ID = "gemini-2.0-flash@001"
model    = genai.GenerativeModel(MODEL_ID)

@retry(reraise=True, stop=stop_after_attempt(4), wait=wait_exponential(min=1, max=6))
def summarize_with_gemini(prompt: str) -> str:
    """Gemini 호출(재시도 포함)"""
    resp = model.generate_content(
        prompt,
        generation_config={"temperature": 0.3, "max_output_tokens": 256},
    )
    return (resp.text or "").strip()

# ─────────────────────────────────────────
# 4. PDF → 페이지별 텍스트 추출 & 요약
# ─────────────────────────────────────────
def extract_pages(pdf_bytes: bytes) -> List[str]:
    reader = PdfReader(io.BytesIO(pdf_bytes))
    pages = []
    for p in reader.pages:
        t = p.extract_text() or ""
        pages.append(t.strip())
    return pages

def summarize_pages(pages: List[str]) -> List[str]:
    results = []
    for i, page_text in enumerate(pages, 1):
        if not page_text or len(page_text) < 50:
            results.append("해당 페이지에서 텍스트를 거의 찾을 수 없습니다.")
            continue

        clipped = page_text[:2000]  # 토큰/요금 절약
        prompt = (
            f"다음은 PDF {i}페이지 기사(들)입니다.\n"
            f"각 기사(문단)별로 핵심만 뽑아 **3줄**로 요약해 주세요.\n"
            f"- 각 줄은 '-'로 시작하는 bullet 형식\n"
            f"- 수치, 기관/회사명, 정책명 등은 그대로 남기기\n\n"
            f"{clipped}"
        )
        try:
            summary = summarize_with_gemini(prompt)
        except Exception as e:
            summary = f"요약 실패: {type(e).__name__}: {e}"
        results.append(summary)
    return results

# ─────────────────────────────────────────
# 5. 사이드바: 업로드
# ─────────────────────────────────────────
st.sidebar.header("📤 PDF 업로드")
uploaded = st.sidebar.file_uploader("PDF 파일 선택", type="pdf")
if uploaded:
    upload_pdf(uploaded.name, uploaded.read())
    st.sidebar.success(f"✅ GCS에 저장됨: {uploaded.name}")

# ─────────────────────────────────────────
# 6. 메인: 목록 & 페이지별 요약
# ─────────────────────────────────────────
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
                pages = extract_pages(data)
                summaries = summarize_pages(pages)

            if not summaries:
                st.warning("텍스트 추출 실패 또는 비어있는 PDF입니다.")
            else:
                for idx, (src, summ) in enumerate(zip(pages, summaries), 1):
                    with st.expander(f"📄 {idx} 페이지 미리보기"):
                        st.text(src[:400] + ("..." if len(src) > 400 else ""))
                    st.markdown("**📝 3줄 요약**")
                    st.text_area(f"p{idx} 요약", summ, height=160)
                    st.markdown("---")
