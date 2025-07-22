import io, json, base64, streamlit as st
from PyPDF2 import PdfReader
from google.cloud import storage, aiplatform

# 페이지 설정
st.set_page_config(page_title="PDF 요약 (Gemini 무료)", layout="wide")
st.title("📂 PDF 업로드 → Gemini(Text Bison) 요약")

# GCS 시크릿
b64_gcs      = st.secrets["GCS_SA_KEY_B64"]
gcs_info     = json.loads(base64.b64decode(b64_gcs))
gcs_bucket   = st.secrets["GCS_BUCKET_NAME"]

# Vertex AI 시크릿
b64_vertex   = st.secrets["VERTEX_SA_KEY_B64"]
vertex_info  = json.loads(base64.b64decode(b64_vertex))

# GCS 클라이언트
gcs_client   = storage.Client.from_service_account_info(gcs_info)
bucket       = gcs_client.bucket(gcs_bucket)

# Vertex AI 초기화
aiplatform.init(
    credentials=vertex_info,
    project=vertex_info["project_id"],
    location="us-central1"  # Text Bison 지원 리전
)

def list_pdfs():
    return [blob.name.split("/",1)[1] for blob in gcs_client.list_blobs(bucket, prefix="pdfs/") if blob.name.endswith(".pdf")]

def upload_pdf(name, data):
    blob = bucket.blob(f"pdfs/{name}")
    blob.upload_from_file(io.BytesIO(data), content_type="application/pdf")

def download_pdf(name):
    return bucket.blob(f"pdfs/{name}").download_as_bytes()

def summarize_with_bison(text: str) -> str:
    endpoint = f"projects/{vertex_info['project_id']}/locations/us-central1/publishers/google/models/text-bison@001"
    response = aiplatform.gapic.PredictionServiceClient(credentials=vertex_info) \
      .predict(
         endpoint=endpoint,
         instances=[{"content": text}],
         parameters={"temperature":0.3, "maxOutputTokens":256}
      )
    return response.predictions[0]["content"]

# 사이드바: PDF 업로드
uploaded = st.sidebar.file_uploader("PDF 업로드", type="pdf")
if uploaded:
    data = uploaded.read()
    upload_pdf(uploaded.name, data)
    st.sidebar.success("✅ 업로드 완료")

# 메인: PDF 목록 & 요약
st.header("📑 PDF 및 Gemini 요약")
pdfs      = list_pdfs()
if not pdfs:
    st.info("pdfs/ 폴더에 PDF를 올려보세요.")
else:
    for name in sorted(pdfs):
        st.subheader(name)
        raw = download_pdf(name)
        text = ""
        for p in PdfReader(io.BytesIO(raw)).pages:
            text += (p.extract_text() or "")
            if len(text) > 1000: 
                text = text[:1000]; break
        summary = summarize_with_bison(text)
        st.markdown(summary)
        st.markdown("---")
