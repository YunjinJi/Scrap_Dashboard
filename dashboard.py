import io, json, base64, streamlit as st
from PyPDF2 import PdfReader
from google.cloud import storage
from google.oauth2 import service_account
from google.ai.generative.v1beta import TextGenerationClient
from google.ai.generative.v1beta.types import GenerateTextRequest
from tenacity import retry, stop_after_attempt, wait_exponential

# ─── 페이지 설정 ────────────────────────────────────────────
st.set_page_config(page_title="PDF → Gemini 2.0 Flash 요약", layout="wide")
st.title("📄 PDF 업로드 & Gemini‑2.0‑Flash@001 3줄 요약")

# ─── 1) GCS 인증 ─────────────────────────────────────────────
gcs_b64     = st.secrets["GCS_SA_KEY_B64"]
gcs_info    = json.loads(base64.b64decode(gcs_b64))
bucket_name = st.secrets["GCS_BUCKET_NAME"]

gcs_creds   = service_account.Credentials.from_service_account_info(gcs_info)
gcs_client  = storage.Client(credentials=gcs_creds, project=gcs_info["project_id"])
bucket      = gcs_client.bucket(bucket_name)

# ─── 2) Generative AI 클라이언트 초기화 ──────────────────────
#   서비스 계정은 GCP metadata 또는 ADC를 쓰면 자동으로
#   올라가지만, 명시적으로 지정하려면 아래처럼 해주셔도 됩니다:
genai_b64   = st.secrets["GENAI_SA_KEY_B64"]
genai_info  = json.loads(base64.b64decode(genai_b64))
genai_creds = service_account.Credentials.from_service_account_info(genai_info)

text_client = TextGenerationClient(credentials=genai_creds)
# 모델 전체 리소스 경로
MODEL_NAME = (
    f"projects/{genai_info['project_id']}/"
    f"locations/us-central1/"
    f"publishers/google/models/gemini-2.0-flash-001"
)

# ─── 3) PDF → 텍스트 추출 & 요약 함수 ────────────────────────
@retry(reraise=True, stop=stop_after_attempt(4), wait=wait_exponential(min=1, max=5))
def summarize_with_gemini(prompt: str) -> str:
    req = GenerateTextRequest(
        model=MODEL_NAME,
        prompt=prompt,
        temperature=0.3,
        max_output_tokens=256,
    )
    res = text_client.generate_text(request=req)
    return res.text  

def summarize_pdf_bytes(pdf_bytes: bytes) -> str:
    # PDF에서 앞 1,000자만 뽑아서 요약하도록
    reader = PdfReader(io.BytesIO(pdf_bytes))
    full = "".join((p.extract_text() or "") for p in reader.pages)[:1000]
    if not full.strip():
        return "PDF에서 텍스트를 추출할 수 없습니다."
    prompt = "다음 내용을 3줄 이내로 요약해 주세요:\n\n" + full
    try:
        return summarize_with_gemini(prompt).strip()
    except Exception as e:
        st.error(f"요약 실패: {e}")
        return "요약 생성 실패"

# ─── 4) 사이드바: PDF 업로드 ────────────────────────────────
st.sidebar.header("📤 PDF 업로드")
uploaded = st.sidebar.file_uploader("PDF 파일 선택", type="pdf")
if uploaded:
    bucket.blob(f"pdfs/{uploaded.name}").upload_from_file(
        io.BytesIO(uploaded.read()),
        content_type="application/pdf"
    )
    st.sidebar.success(f"✅ GCS에 저장됨: {uploaded.name}")

# ─── 5) 메인: 저장된 PDF 목록 & 요약 ────────────────────────
st.header("📑 GCS에 저장된 PDF 및 3줄 요약")
pdfs = [
    blob.name.split("/",1)[1]
    for blob in gcs_client.list_blobs(bucket_name, prefix="pdfs/")
    if blob.name.endswith(".pdf")
]

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
