import io, json, base64, streamlit as st
from PyPDF2 import PdfReader
from google.cloud import storage
from google.oauth2 import service_account
import vertexai
from vertexai.preview.language_models import TextGenerationModel
from tenacity import retry, stop_after_attempt, wait_exponential

# ─── 페이지 설정 ─────────────────
st.set_page_config(page_title="PDF → Gemini 요약", layout="wide")
st.title("📄 PDF 업로드 & Gemini(Text Bison) 요약")

# ─── 시크릿 & 클라이언트 초기화 ────────────────
# GCS
gcs_b64     = st.secrets["GCS_SA_KEY_B64"]
gcs_info    = json.loads(base64.b64decode(gcs_b64))
bucket_name = st.secrets["GCS_BUCKET_NAME"]

gcs_creds   = service_account.Credentials.from_service_account_info(gcs_info)
gcs_client  = storage.Client(credentials=gcs_creds, project=gcs_info["project_id"])
bucket      = gcs_client.bucket(bucket_name)

# Vertex AI
vert_b64    = st.secrets["VERTEX_SA_KEY_B64"]
vert_info   = json.loads(base64.b64decode(vert_b64))
vert_creds  = service_account.Credentials.from_service_account_info(vert_info)

vertexai.init(
    project=vert_info["project_id"],
    location="us-central1",
    credentials=vert_creds,
)
# 무료 할당 모델
MODEL_NAME = "text-bison@001"
model = TextGenerationModel.from_pretrained(MODEL_NAME)

@retry(reraise=True, stop=stop_after_attempt(5), wait=wait_exponential(min=1, max=15))
def summarize_with_gemini(text: str) -> str:
    resp = model.predict(
        prompt=text,
        temperature=0.3,
        max_output_tokens=256,
    )
    return resp.text.strip()

# ─── GCS PDF 관리 함수 ─────────────────
def list_pdfs():
    return [
        blob.name.split("/",1)[1]
        for blob in gcs_client.list_blobs(bucket_name, prefix="pdfs/")
        if blob.name.endswith(".pdf")
    ]

def upload_pdf(name: str, data: bytes):
    b = bucket.blob(f"pdfs/{name}")
    b.upload_from_file(io.BytesIO(data), content_type="application/pdf")

def download_pdf(name: str) -> bytes:
    return bucket.blob(f"pdfs/{name}").download_as_bytes()

# ─── PDF → 요약 ─────────────────
def summarize_pdf_bytes(pdf_bytes: bytes) -> str:
    reader = PdfReader(io.BytesIO(pdf_bytes))
    txt = "".join(p.extract_text() or "" for p in reader.pages)[:1000]
    if not txt:
        return "PDF에서 텍스트를 추출할 수 없습니다."
    prompt = "다음 PDF를 5문장 이내로 요약해 주세요:\n\n" + txt
    try:
        return summarize_with_gemini(prompt)
    except Exception as e:
        st.error(f"요약 실패: {e}")
        return "요약 생성 실패"

# ─── UI: 업로드 & 요약 ─────────────────
st.sidebar.header("📤 PDF 업로드")
up = st.sidebar.file_uploader("PDF 선택", type="pdf")
if up:
    upload_pdf(up.name, up.read())
    st.sidebar.success("GCS에 저장됨")

st.header("📑 저장된 PDF 및 요약")
for name in sorted(list_pdfs()):
    st.subheader(name)
    if st.button(f"요약 보기: {name}", key=name):
        data = download_pdf(name)
        with st.spinner("요약 중…"):
            summary = summarize_pdf_bytes(data)
        st.text_area("요약 결과", summary, height=200)
