# app.py
import io
import json
import base64

import streamlit as st
from PyPDF2 import PdfReader
from google.cloud import storage
from google.oauth2 import service_account
import google.generativeai as genai
from tenacity import retry, stop_after_attempt, wait_exponential

# ─── 페이지 설정 ────────────────────────────────────────────
st.set_page_config(page_title="PDF → text‑bison@001 요약", layout="wide")
st.title("📄 PDF 업로드 & text‑bison@001 요약")

# ─── 시크릿 로드 & GCS 인증 ─────────────────────────────────
# 1) GCS 서비스 계정 JSON(Base64)과 버킷명
gcs_b64     = st.secrets["GCS_SA_KEY_B64"]
gcs_info    = json.loads(base64.b64decode(gcs_b64))
bucket_name = st.secrets["GCS_BUCKET_NAME"]

gcs_creds   = service_account.Credentials.from_service_account_info(gcs_info)
gcs_client  = storage.Client(credentials=gcs_creds, project=gcs_info["project_id"])
bucket      = gcs_client.bucket(bucket_name)

# 2) Generative AI 서비스 계정 JSON(Base64)
genai_b64   = st.secrets["GENAI_SA_KEY_B64"]
genai_info  = json.loads(base64.b64decode(genai_b64))
genai_creds = service_account.Credentials.from_service_account_info(genai_info)

# 3) SDK 구성
genai.configure(
    api_key=None,
    api_key_from=genai_creds,
)

MODEL_ID = "text-bison@001"

# ─── GCS PDF 관리 함수 ────────────────────────────────────────
def list_pdfs() -> list[str]:
    """pdfs/ 폴더에 있는 PDF 파일명 리스트"""
    blobs = gcs_client.list_blobs(bucket_name, prefix="pdfs/")
    return [b.name.split("/",1)[1] for b in blobs if b.name.endswith(".pdf")]

def upload_pdf(name: str, data: bytes):
    """pdfs/{name} 으로 GCS에 업로드"""
    blob = bucket.blob(f"pdfs/{name}")
    blob.upload_from_file(io.BytesIO(data), content_type="application/pdf")

def download_pdf(name: str) -> bytes:
    """pdfs/{name} 에서 바이트로 다운로드"""
    return bucket.blob(f"pdfs/{name}").download_as_bytes()

# ─── 요약 호출 (재시도 포함) ─────────────────────────────────
@retry(reraise=True, stop=stop_after_attempt(5), wait=wait_exponential(min=1, max=8))
def summarize_with_bison(prompt: str) -> str:
    resp = genai.generate_text(
        model=MODEL_ID,
        prompt=prompt,
        temperature=0.3,
        max_output_tokens=256
    )
    return resp.text.strip()

# ─── PDF → 텍스트 추출 & 요약 ─────────────────────────────────
def summarize_pdf_bytes(pdf_bytes: bytes) -> str:
    reader = PdfReader(io.BytesIO(pdf_bytes))
    text = ""
    for page in reader.pages:
        t = page.extract_text() or ""
        text += t
        if len(text) > 1000:
            text = text[:1000]
            break

    if not text.strip():
        return "PDF에서 텍스트를 추출할 수 없습니다."

    prompt = "다음 PDF 내용을 5문장 이내로 요약해 주세요:\n\n" + text
    try:
        return summarize_with_bison(prompt)
    except Exception as e:
        st.error(f"요약 실패: {type(e).__name__}: {e}")
        return "요약 생성 실패"

# ─── UI: 사이드바 – PDF 업로드 ───────────────────────────────
st.sidebar.header("📤 PDF 업로드")
uploaded = st.sidebar.file_uploader("PDF 파일 선택", type="pdf")
if uploaded:
    upload_pdf(uploaded.name, uploaded.read())
    st.sidebar.success(f"✅ GCS에 저장됨: {uploaded.name}")

# ─── UI: 메인 – GCS 목록 & 요약 ───────────────────────────────
st.header("📑 GCS에 저장된 PDF 및 요약")
pdfs = list_pdfs()
if not pdfs:
    st.info("GCS 버킷의 pdfs/ 폴더에 PDF를 업로드해 보세요.")
else:
    for name in sorted(pdfs):
        st.subheader(name)
        if st.button(f"요약: {name}", key=name):
            data    = download_pdf(name)
            with st.spinner("요약 생성 중…"):
                summary = summarize_pdf_bytes(data)
            st.text_area("📝 요약 결과", summary, height=200)
        st.markdown("---")
