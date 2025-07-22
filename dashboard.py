import io
import json
import base64

import streamlit as st
from PyPDF2 import PdfReader
from google.cloud import storage
from google.oauth2 import service_account
import vertexai
from vertexai.preview.language_models import ChatModel
from tenacity import retry, stop_after_attempt, wait_exponential

# ─── 페이지 설정 ────────────────────────────────────────────
st.set_page_config(page_title="PDF → Gemini‑2.0‑Flash 요약", layout="wide")
st.title("📄 PDF 업로드 & Gemini‑2.0‑Flash 요약 (3줄)")

# ─── 시크릿 & GCS 인증 ─────────────────────────────────────
gcs_b64     = st.secrets["GCS_SA_KEY_B64"]
gcs_info    = json.loads(base64.b64decode(gcs_b64))
bucket_name = st.secrets["GCS_BUCKET_NAME"]

gcs_creds   = service_account.Credentials.from_service_account_info(gcs_info)
gcs_client  = storage.Client(credentials=gcs_creds, project=gcs_info["project_id"])
bucket      = gcs_client.bucket(bucket_name)

# ─── Vertex AI 초기화 ────────────────────────────────────────
vert_b64   = st.secrets["VERTEX_SA_KEY_B64"]
vert_info  = json.loads(base64.b64decode(vert_b64))
vert_creds = service_account.Credentials.from_service_account_info(vert_info)

vertexai.init(
    project=vert_info["project_id"],
    location="us-central1",
    credentials=vert_creds,
)

# ─── Gemini‑2.0‑Flash 채팅 모델 로드 ─────────────────────────
MODEL_NAME = "gemini-2.0-flash@001"
chat_model = ChatModel.from_pretrained(MODEL_NAME)

@retry(reraise=True, stop=stop_after_attempt(4), wait=wait_exponential(min=1, max=5))
def summarize_with_gemini(prompt: str) -> str:
    session = chat_model.start_chat()
    resp = session.send_message(
        prompt,
        temperature=0.3,
        max_output_tokens=256,
    )
    return resp.text.strip()

# ─── GCS 유틸 함수 ────────────────────────────────────────────
def list_pdfs() -> list[str]:
    return [
        blob.name.split("/", 1)[1]
        for blob in gcs_client.list_blobs(bucket_name, prefix="pdfs/")
        if blob.name.endswith(".pdf")
    ]

def upload_pdf(name: str, data: bytes):
    blob = bucket.blob(f"pdfs/{name}")
    blob.upload_from_file(io.BytesIO(data), content_type="application/pdf")

def download_pdf(name: str) -> bytes:
    return bucket.blob(f"pdfs/{name}").download_as_bytes()

# ─── PDF → 텍스트 추출 & 3줄 요약 ─────────────────────────────
def summarize_pdf_bytes(pdf_bytes: bytes) -> str:
    reader = PdfReader(io.BytesIO(pdf_bytes))
    text = ""
    for page in reader.pages:
        txt = page.extract_text() or ""
        text += txt
        if len(text) > 1000:
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

# ─── UI: 사이드바 – PDF 업로드 ───────────────────────────────
st.sidebar.header("📤 PDF 업로드")
uploaded = st.sidebar.file_uploader("PDF 파일 선택", type="pdf")
if uploaded:
    upload_pdf(uploaded.name, uploaded.read())
    st.sidebar.success(f"✅ GCS에 저장됨: {uploaded.name}")

# ─── UI: 메인 – 저장된 PDF 및 요약 ───────────────────────────
st.header("📑 저장된 PDF 및 요약 (3줄)")
pdfs = list_pdfs()
if not pdfs:
    st.info("GCS 버킷의 pdfs/ 폴더에 PDF를 업로드해 보세요.")
else:
    for name in sorted(pdfs):
        st.subheader(name)
        if st.button(f"요약: {name}", key=name):
            pdf_bytes = download_pdf(name)
            with st.spinner("요약 생성 중…"):
                summary = summarize_pdf_bytes(pdf_bytes)
            st.text_area("📝 요약 결과", summary, height=200)
        st.markdown("---")
