import io
import re
import json
import base64
from typing import List, Tuple

import streamlit as st
from PyPDF2 import PdfReader
from google.cloud import storage
from google.oauth2 import service_account
import google.generativeai as genai
from tenacity import retry, stop_after_attempt, wait_exponential

# ─────────────────────────────────────────────────────────────
# 기본 설정
# ─────────────────────────────────────────────────────────────
st.set_page_config(page_title="PDF → 기사별 3줄 요약 (Gemini 2.0 Flash)", layout="wide")
st.title("📄 PDF 업로드 → 기사별 3줄 요약 (gemini-2.0-flash@001)")

# ─────────────────────────────────────────────────────────────
# 시크릿 로드 & 안전 체크
# ─────────────────────────────────────────────────────────────
missing = []
gemini_key   = st.secrets.get("GEMINI_API_KEY") or missing.append("GEMINI_API_KEY")
gcs_b64      = st.secrets.get("GCS_SA_KEY_B64") or missing.append("GCS_SA_KEY_B64")
bucket_name  = st.secrets.get("GCS_BUCKET_NAME") or missing.append("GCS_BUCKET_NAME")

if missing:
    st.error(f"Secrets에 {', '.join(missing)} 가 없습니다. Manage app → Settings → Secrets 에 등록하세요.")
    st.stop()

# ─────────────────────────────────────────────────────────────
# GCS 클라이언트
# ─────────────────────────────────────────────────────────────
gcs_info   = json.loads(base64.b64decode(gcs_b64))
gcs_creds  = service_account.Credentials.from_service_account_info(gcs_info)
gcs_client = storage.Client(credentials=gcs_creds, project=gcs_info["project_id"])
bucket     = gcs_client.bucket(bucket_name)

# ─────────────────────────────────────────────────────────────
# Gemini 설정
# ─────────────────────────────────────────────────────────────
genai.configure(api_key=gemini_key)
MODEL_ID = "gemini-2.0-flash-001"
model    = genai.GenerativeModel(MODEL_ID)

# ─────────────────────────────────────────────────────────────
# 유틸 함수: GCS
# ─────────────────────────────────────────────────────────────
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

# ─────────────────────────────────────────────────────────────
# 텍스트 분리 로직: 기사 단위로 나누기
# ─────────────────────────────────────────────────────────────
def split_articles(raw_text: str) -> List[str]:
    """
    매우 단순한 휴리스틱:
      1) 특수 기호(■▲▶◆□) 앞에 빈 줄 삽입
      2) 빈 줄 2개 이상 기준으로 분리
      3) 너무 짧은 조각 제외
    프로젝트 데이터 특성에 맞춰 정규식은 조정하세요.
    """
    # 기호를 기사 구분자로 사용 (필요시 추가)
    text = re.sub(r"[■▲▶◆□]\s*", "\n\n", raw_text)
    # 빈 줄 2개 이상을 기준으로 split
    chunks = re.split(r"\n\s*\n\s*\n+", text)
    # 길이가 너무 짧은 조각 제거 (120자 기준 임의 설정)
    articles = [c.strip() for c in chunks if len(c.strip()) > 120]
    return articles

# ─────────────────────────────────────────────────────────────
# Gemini 요약 호출 (재시도 포함)
# ─────────────────────────────────────────────────────────────
@retry(reraise=True, stop=stop_after_attempt(4), wait=wait_exponential(min=1, max=6))
def summarize_with_gemini(prompt: str) -> str:
    resp = model.generate_content(
        prompt,
        generation_config={"temperature": 0.3, "max_output_tokens": 256},
    )
    return (resp.text or "").strip()

# ─────────────────────────────────────────────────────────────
# PDF → 기사별 텍스트 추출 & 요약
# ─────────────────────────────────────────────────────────────
def summarize_pdf_bytes(pdf_bytes: bytes) -> List[Tuple[int, str, str]]:
    """
    return: [(기사번호, 미리보기텍스트, 요약문)]
    """
    reader = PdfReader(io.BytesIO(pdf_bytes))
    raw = "".join((p.extract_text() or "") for p in reader.pages)
    if not raw.strip():
        return []

    articles = split_articles(raw)
    results  = []

    for idx, art in enumerate(articles, 1):
        # 토큰 비용 절약: 기사 본문도 너무 길면 앞부분만
        clipped = art[:2000]
        prompt  = (
            "다음 기사 내용을 3줄 이내로 요약해 주세요.\n"
            "조건:\n"
            "1. 핵심 사실/수치/주체만 남기고 군더더기 제거\n"
            "2. 줄바꿈으로 3줄을 명확히 구분\n\n"
            f"{clipped}"
        )
        try:
            summary = summarize_with_gemini(prompt)
        except Exception as e:
            summary = f"요약 실패: {type(e).__name__}: {e}"
        preview = art[:150].replace("\n", " ")
        results.append((idx, preview, summary))

    return results

# ─────────────────────────────────────────────────────────────
# UI: PDF 업로드
# ─────────────────────────────────────────────────────────────
st.sidebar.header("📤 PDF 업로드")
uploaded = st.sidebar.file_uploader("PDF 파일 선택", type="pdf")
if uploaded:
    upload_pdf(uploaded.name, uploaded.read())
    st.sidebar.success(f"✅ GCS에 저장됨: {uploaded.name}")

# ─────────────────────────────────────────────────────────────
# UI: 목록 & 기사별 3줄 요약
# ─────────────────────────────────────────────────────────────
st.header("📑 저장된 PDF → 기사별 3줄 요약")
pdfs = list_pdfs()

if not pdfs:
    st.info("GCS 버킷의 pdfs/ 폴더에 PDF를 업로드해 보세요.")
else:
    for name in sorted(pdfs):
        st.subheader(name)
        if st.button(f"요약: {name}", key=name):
            data = download_pdf(name)
            with st.spinner("요약 생성 중…"):
                items = summarize_pdf_bytes(data)

            if not items:
                st.warning("텍스트 추출에 실패했거나 기사를 찾지 못했습니다.")
            else:
                for idx, preview, summary in items:
                    with st.expander(f"기사 #{idx}  |  미리보기: {preview} ..."):
                        st.text_area("요약 (3줄)", summary, height=150)
        st.markdown("---")
