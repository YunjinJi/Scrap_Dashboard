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
st.title("📄 PDF 업로드 & text‑bison@001 요약 (3줄)")

# ─── 시크릿 로드 & GCS 인증 ─────────────────────────────────
gcs_b64     = st.secrets["GCS_SA_KEY_B64"]
gcs_info    = json.loads(base64.b64decode(gcs_b64))
bucket_name = st.secrets["GCS_BUCKET_NAME"]

gcs_creds   = service_account.Credentials.from_service_account_info(gcs_info)
gcs_client  = storage.Client(credentials=gcs_creds, project=gcs_info["project_id"])
bucket      = gcs_client.bucket(bucket_name)

# ─── Generative AI 인증 ─────────────────────────────────────
genai_b64   = st.secrets["GENAI_SA_KEY_B64"]
genai_info  = json.loads(base64.b64decode(genai_b64))
genai_creds = service_account.Credentials.from_service_account_info(genai_info)
genai.configure(api_key=None, credentials=genai_creds)

MODEL_ID = "text-bison@001"

# ─── GCS 유틸 함수 ───────────────────────────────────────────
def list_pdfs() -> list[str]:
    blobs = gcs_client.list_blobs(bucket_name, prefix="pdfs/")
    return [b.name.split("/",1)[1] for b in blobs if b.name.endswith(".pdf")]

def upload_pdf(name: str, data: bytes):
    blob = bucket.blob(f"pdfs/{name}")
    blob.upload_from_file(io.BytesIO(data), content_type="application/pdf")

def download_pdf(name: str) -> bytes:
    return bucket.blob(f"pdfs/{name}").download_as_bytes()

# ─── 요약 호출 (text completions) ────────────────────────────
@retry(reraise=True, stop=stop_after_attempt(5), wait=wait_exponential(min=1, max=8))
def summarize_with_bison(prompt: str) -> str:
    resp = genai.text.completions.create(
        model=MODEL_ID,
        prompt=prompt,
        temperature=0.3,
        max_output_tokens=256,
    )
    return resp.choices[0].text.strip()

# ─── PDF → 텍스트 추출 & 3줄 요약 ─────────────────────────────
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

    prompt = "다음 PDF 내용을 **3줄 이내**로 요약해 주세요:\n\n" + text
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

# ─── UI: 메인 – GCS 목록 & 3줄 요약 ───────────────────────────
st.header("📑 GCS에 저장된 PDF 및 요약 (3줄)")
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
