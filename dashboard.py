import io
import json
import base64

import streamlit as st
from PyPDF2 import PdfReader
from google.cloud import storage, aiplatform_v1
from google.oauth2 import service_account
from tenacity import retry, stop_after_attempt, wait_exponential

# ----------------- 페이지 설정 -----------------
st.set_page_config(page_title="PDF 요약 (Gemini/Text-Bison)", layout="wide")
st.title("📂 GCS PDF 업로드 → Gemini(Text-Bison) 요약")

# ----------------- 시크릿 로드 -----------------
# GCS
gcs_b64      = st.secrets["GCS_SA_KEY_B64"]
gcs_info     = json.loads(base64.b64decode(gcs_b64))
bucket_name  = st.secrets["GCS_BUCKET_NAME"]

# Vertex AI
vertex_b64   = st.secrets["VERTEX_SA_KEY_B64"]
vertex_info  = json.loads(base64.b64decode(vertex_b64))

# ----------------- 인증 클라이언트 생성 -----------------
gcs_credentials    = service_account.Credentials.from_service_account_info(gcs_info)
gcs_client         = storage.Client(credentials=gcs_credentials, project=gcs_info["project_id"])
bucket             = gcs_client.bucket(bucket_name)

vertex_credentials = service_account.Credentials.from_service_account_info(vertex_info)
prediction_client  = aiplatform_v1.PredictionServiceClient(credentials=vertex_credentials)
project_id         = vertex_info["project_id"]
location           = "us-central1"  # 모델 리전
model_name         = "gemini-1.5-flash"
endpoint           = f"projects/{project_id}/locations/{location}/publishers/google/models/{model_name}"

# ----------------- GCS 유틸 함수 -----------------
def list_pdfs():
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

# ----------------- Gemini 요약 함수 -----------------
@retry(reraise=True, stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=1, max=15))
def summarize_with_bison(text: str) -> str:
    # Vertex AI Prediction 호출
    response = prediction_client.predict(
        endpoint=endpoint,
        instances=[{"content": text}],
        parameters={"temperature": 0.3, "maxOutputTokens": 256},
    )
    # response.predictions[0] 구조: {"content": "..."}
    return response.predictions[0].get("content", "").strip()

# ----------------- PDF 텍스트 → 요약 -----------------
def summarize_pdf(pdf_bytes: bytes) -> str:
    reader = PdfReader(io.BytesIO(pdf_bytes))
    combined = ""
    for page in reader.pages:
        combined += page.extract_text() or ""
        if len(combined) > 1000:
            combined = combined[:1000]
            break
    if not combined.strip():
        return "PDF에서 추출된 텍스트가 없습니다."
    prompt_text = (
        "다음 한국어 PDF 내용을 5문장 이내로 핵심만 요약해 주세요:\n\n"
        f"{combined}"
    )
    try:
        return summarize_with_bison(prompt_text)
    except Exception as e:
        st.error(f"요약 호출 실패: {type(e).__name__}: {e}")
        return "요약 생성 실패"

# ----------------- 사이드바 업로드 -----------------
st.sidebar.header("📤 PDF 업로드")
uploaded = st.sidebar.file_uploader("PDF 선택", type="pdf")
if uploaded:
    upload_pdf(uploaded.name, uploaded.read())
    st.sidebar.success("업로드 완료! 페이지를 새로고침해 보세요.")

# ----------------- 메인: 목록/요약 -----------------
st.header("📑 저장된 PDF 목록 및 요약")
pdf_list = list_pdfs()

if not pdf_list:
    st.info("pdfs/ 경로에 PDF가 없습니다. 사이드바에서 업로드하세요.")
else:
    # 세션 상태 초기화
    if "summaries" not in st.session_state:
        st.session_state["summaries"] = {}

    for pdf_name in sorted(pdf_list):
        st.subheader(pdf_name)

        # 버튼 클릭 시 요약 생성하여 세션에 저장
        if st.button(f"요약 보기: {pdf_name}", key=f"btn_{pdf_name}"):
            data = download_pdf(pdf_name)
            with st.spinner("요약 생성 중..."):
                summary = summarize_pdf(data)
            st.session_state["summaries"][pdf_name] = summary

        # 세션에 저장된 요약이 있으면 항상 표시
        if pdf_name in st.session_state["summaries"]:
            st.write("**요약:**")
            st.write(st.session_state["summaries"][pdf_name])

        st.markdown("---")
