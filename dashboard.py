import io
import json
import base64

import streamlit as st
from PyPDF2 import PdfReader
from google.cloud import storage, aiplatform_v1
from google.oauth2 import service_account
from tenacity import retry, stop_after_attempt, wait_exponential

# ─── 페이지 설정 ────────────────────────────────────────────
st.set_page_config(page_title="PDF → Gemini‑2.0‑Flash 요약", layout="wide")
st.title("📄 PDF 업로드 & Gemini‑2.0‑Flash@001 3줄 요약")

# ─── 1) GCS 인증 ────────────────────────────────────────────
gcs_b64     = st.secrets["GCS_SA_KEY_B64"]
gcs_info    = json.loads(base64.b64decode(gcs_b64))
bucket_name = st.secrets["GCS_BUCKET_NAME"]
gcs_creds   = service_account.Credentials.from_service_account_info(gcs_info)
gcs_client  = storage.Client(credentials=gcs_creds, project=gcs_info["project_id"])
bucket      = gcs_client.bucket(bucket_name)

# ─── 2) Vertex AI Prediction 클라이언트 초기화 ─────────────
vert_b64   = st.secrets["VERTEX_SA_KEY_B64"]
vert_info  = json.loads(base64.b64decode(vert_b64))
vert_creds = service_account.Credentials.from_service_account_info(vert_info)

prediction_client = aiplatform_v1.PredictionServiceClient(credentials=vert_creds)
project_id = vert_info["project_id"]
location   = "us-central1"
MODEL_ID   = "gemini-2.0-flash-001"
endpoint   = f"projects/{project_id}/locations/{location}/publishers/google/models/{MODEL_ID}"

# ─── 3) PDF 목록 함수 & 업로드 UI ─────────────────────────────
def list_pdfs():
    return [
        blob.name.split("/", 1)[1]
        for blob in gcs_client.list_blobs(bucket_name, prefix="pdfs/")
        if blob.name.endswith(".pdf")
    ]

st.sidebar.header("📤 PDF 업로드")
uploaded = st.sidebar.file_uploader("PDF 선택", type="pdf")
if uploaded:
    blob = bucket.blob(f"pdfs/{uploaded.name}")
    blob.upload_from_file(io.BytesIO(uploaded.read()), content_type="application/pdf")
    st.sidebar.success(f"✅ GCS에 저장됨: {uploaded.name}")

# ─── 4) 요약 함수 ────────────────────────────────────────────
@retry(reraise=True, stop=stop_after_attempt(4), wait=wait_exponential(min=1, max=5))
def summarize_with_gemini(prompt: str) -> str:
    response = prediction_client.predict(
        endpoint=endpoint,
        instances=[{"content": prompt}],
        parameters={"temperature": 0.3, "maxOutputTokens": 256},
    )
    # 반환된 predictions 리스트의 첫 번째 요소에서 content 키 추출
    return response.predictions[0]["content"].strip()

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

    prompt = "다음 내용을 3줄 이내로 요약해 주세요:\n\n" + text
    try:
        return summarize_with_gemini(prompt)
    except Exception as e:
        st.error(f"요약 실패: {e}")
        return "요약 생성 실패"

# ─── 5) 메인 UI: PDF 선택 & 요약 실행 ─────────────────────────
st.header("📑 저장된 PDF 및 3줄 요약")
pdfs = list_pdfs()
if not pdfs:
    st.info("GCS 버킷의 pdfs/ 폴더에 PDF를 업로드해 보세요.")
else:
    for name in sorted(pdfs):
        st.subheader(name)
        if st.button(f"요약: {name}", key=name):
            data = bucket.blob(f"pdfs/{name}").download_as_bytes()
            with st.spinner("요약 생성 중…"):
                summary = summarize_pdf_bytes(data)
            st.text_area("📝 요약 결과", summary, height=200)
        st.markdown("---")
