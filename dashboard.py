import io
import json
import base64

import streamlit as st
from PyPDF2 import PdfReader
from google.cloud import storage
from google.oauth2 import service_account
import google.generativeai as genai
from tenacity import retry, stop_after_attempt, wait_exponential

# ------------------ Streamlit 기본 설정 ------------------
st.set_page_config(page_title="PDF → Gemini 2.0 Flash 요약", layout="wide")
st.title("📄 PDF 업로드 & Gemini‑2.0‑Flash@001 3줄 요약")

# ------------------ GCS 인증 ------------------
gcs_b64     = st.secrets["GCS_SA_KEY_B64"]
gcs_info    = json.loads(base64.b64decode(gcs_b64))
bucket_name = st.secrets["GCS_BUCKET_NAME"]

gcs_creds   = service_account.Credentials.from_service_account_info(gcs_info)
gcs_client  = storage.Client(credentials=gcs_creds, project=gcs_info["project_id"])
bucket      = gcs_client.bucket(bucket_name)

# ------------------ Gemini 인증 ------------------
# API 키 방식 (가장 간단). makersuite/ai.google.dev 에서 키 발급 후 secrets.toml 에 저장
gemini_key = st.secrets["GEMINI_API_KEY"]
genai.configure(api_key=gemini_key)

MODEL_ID = "gemini-2.0-flash-001"
model    = genai.GenerativeModel(MODEL_ID)

# ------------------ 요약 함수 ------------------
@retry(reraise=True, stop=stop_after_attempt(4), wait=wait_exponential(min=1, max=6))
def summarize_with_gemini(prompt: str) -> str:
    resp = model.generate_content(
        prompt,
        generation_config={"temperature":0.3, "max_output_tokens":256},
    )
    return resp.text.strip()

def summarize_pdf_bytes(pdf_bytes: bytes) -> str:
    reader = PdfReader(io.BytesIO(pdf_bytes))
    text = ""
    for p in reader.pages:
        t = p.extract_text() or ""
        text += t
        if len(text) > 1000:  # 토큰/요금 절약용
            text = text[:1000]
            break

    if not text.strip():
        return "PDF에서 텍스트를 추출할 수 없습니다."

    prompt = "다음 내용을 3줄 이내로 요약해 주세요:\n\n" + text
    try:
        return summarize_with_gemini(prompt)
    except Exception as e:
        st.error(f"요약 실패: {type(e).__name__}: {e}")
        return "요약 생성 실패"

# ------------------ GCS 유틸 ------------------
def list_pdfs():
    return [
        b.name.split("/",1)[1]
        for b in gcs_client.list_blobs(bucket_name, prefix="pdfs/")
        if b.name.endswith(".pdf")
    ]

def upload_pdf(name: str, data: bytes):
    bucket.blob(f"pdfs/{name}").upload_from_file(
        io.BytesIO(data), content_type="application/pdf"
    )

def download_pdf(name: str) -> bytes:
    return bucket.blob(f"pdfs/{name}").download_as_bytes()

# ------------------ UI: 업로드 ------------------
st.sidebar.header("📤 PDF 업로드")
up = st.sidebar.file_uploader("PDF 파일 선택", type="pdf")
if up:
    upload_pdf(up.name, up.read())
    st.sidebar.success(f"✅ GCS에 저장됨: {up.name}")

# ------------------ UI: 목록 & 요약 ------------------
st.header("📑 저장된 PDF 및 3줄 요약")
pdfs = list_pdfs()
if not pdfs:
    st.info("GCS 버킷의 pdfs/ 폴더에 PDF를 업로드해 보세요.")
else:
    for name in sorted(pdfs):
        st.subheader(name)
        if st.button(f"요약: {name}", key=name):
            data = download_pdf(name)
            with st.spinner("요약 생성 중…"):
                summary = summarize_pdf_bytes(data)
            st.text_area("📝 요약 결과", summary, height=200)
        st.markdown("---")
