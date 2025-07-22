import io, json, base64, streamlit as st
from PyPDF2 import PdfReader
from google.cloud import storage
from google.oauth2 import service_account
import vertexai
from vertexai.preview.language_models import TextGenerationModel
from tenacity import retry, stop_after_attempt, wait_exponential

# ─── 페이지 설정 ─────────────────────────────────────────
st.set_page_config(page_title="PDF 요약 (Vertex AI Generative)", layout="wide")
st.title("📄 GCS PDF → Vertex AI 요약")

# ─── 시크릿 & 인증 ─────────────────────────────────────────
# GCS 서비스 계정 JSON (Base64)
gcs_b64     = st.secrets["GCS_SA_KEY_B64"]
gcs_info    = json.loads(base64.b64decode(gcs_b64))
bucket_name = st.secrets["GCS_BUCKET_NAME"]

# Vertex AI 서비스 계정 JSON (Base64)
vert_b64    = st.secrets["VERTEX_SA_KEY_B64"]
vert_info   = json.loads(base64.b64decode(vert_b64))

# GCS 클라이언트
gcs_creds   = service_account.Credentials.from_service_account_info(gcs_info)
gcs_client  = storage.Client(credentials=gcs_creds, project=gcs_info["project_id"])
bucket      = gcs_client.bucket(bucket_name)

# Vertex AI 초기화
vertexai.init(
    project=vert_info["project_id"],
    location="us-central1",
    credentials=service_account.Credentials.from_service_account_info(vert_info),
)

# 사전 로드할 모델 이름: 무료 할당(text-bison@001) 또는 Gemini 모델
MODEL_NAME = "text-bison@001"
model = TextGenerationModel.from_pretrained(MODEL_NAME)

# ─── GCS 유틸 함수 ─────────────────────────────────────────
def list_pdfs() -> list[str]:
    return [
        blob.name.split("/",1)[1]
        for blob in gcs_client.list_blobs(bucket_name, prefix="pdfs/")
        if blob.name.endswith(".pdf")
    ]

def upload_pdf(name: str, data: bytes):
    blob = bucket.blob(f"pdfs/{name}")
    blob.upload_from_file(io.BytesIO(data), content_type="application/pdf")

def download_pdf(name: str) -> bytes:
    return bucket.blob(f"pdfs/{name}").download_as_bytes()

# ─── 요약 함수 (백오프 포함) ─────────────────────────────────
@retry(reraise=True, stop=stop_after_attempt(5), wait=wait_exponential(min=1, max=15))
def summarize_text(prompt: str) -> str:
    response = model.predict(
        prompt=prompt,
        temperature=0.3,
        max_output_tokens=256,
    )
    return response.text.strip()

def summarize_pdf(pdf_bytes: bytes) -> str:
    reader = PdfReader(io.BytesIO(pdf_bytes))
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
        if len(text) > 1000:
            text = text[:1000]
            break
    if not text.strip():
        return "PDF에서 텍스트를 추출할 수 없습니다."
    prompt = "다음 한국어 PDF 내용을 5문장 이내로 요약해 주세요:\n\n" + text
    try:
        return summarize_text(prompt)
    except Exception as e:
        st.error(f"❌ 요약 실패: {type(e).__name__}: {e}")
        return "요약 생성 실패"

# ─── UI: 사이드바 업로드 ────────────────────────────────────
st.sidebar.header("📤 PDF 업로드")
uploaded = st.sidebar.file_uploader("파일 선택", type="pdf")
if uploaded:
    upload_pdf(uploaded.name, uploaded.read())
    st.sidebar.success("✅ 업로드 완료! 목록에서 선택하세요.")

# ─── UI: 메인 – 목록 & 요약 ────────────────────────────────
st.header("📑 저장된 PDF 및 요약")
pdfs      = list_pdfs()
if not pdfs:
    st.info("pdfs/ 폴더에 PDF가 없습니다. 업로드해 보세요.")
else:
    for name in sorted(pdfs):
        st.subheader(name)
        if st.button(f"요약 보기: {name}", key=name):
            pdf_data = download_pdf(name)
            with st.spinner("요약 생성 중…"):
                summary = summarize_pdf(pdf_data)
            st.write("**요약:**")
            st.write(summary)
        st.markdown("---")
