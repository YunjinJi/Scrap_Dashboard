import io, json, base64, re, streamlit as st
from PyPDF2 import PdfReader
from google.cloud import storage
from google.oauth2 import service_account

# ─── 페이지 설정 ─────────────────────────────────────────
st.set_page_config(page_title="PDF 요약 + GCS 저장 (무료)", layout="wide")
st.title("📄 PDF 업로드 → GCS 저장 → 간단 문장 분할 요약")

# ─── 시크릿 로드 & GCS 인증 ───────────────────────────────
gcs_b64     = st.secrets["GCS_SA_KEY_B64"]
gcs_info    = json.loads(base64.b64decode(gcs_b64))
bucket_name = st.secrets["GCS_BUCKET_NAME"]

gcs_creds  = service_account.Credentials.from_service_account_info(gcs_info)
gcs_client = storage.Client(credentials=gcs_creds, project=gcs_info["project_id"])
bucket     = gcs_client.bucket(bucket_name)

# ─── GCS 유틸 함수 ─────────────────────────────────────────
def list_pdfs() -> list[str]:
    blobs = gcs_client.list_blobs(bucket_name, prefix="pdfs/")
    return [b.name.split("/",1)[1] for b in blobs if b.name.endswith(".pdf")]

def upload_pdf_to_gcs(name: str, data: bytes):
    blob = bucket.blob(f"pdfs/{name}")
    blob.upload_from_file(io.BytesIO(data), content_type="application/pdf")

def download_pdf_from_gcs(name: str) -> bytes:
    return bucket.blob(f"pdfs/{name}").download_as_bytes()

# ─── 간단 문장 분리 요약 함수 ───────────────────────────────
def simple_summary(text: str, num_sentences: int) -> str:
    # 마침표, 물음표, 느낌표 기준으로 문장 분리
    parts = re.split(r'(?<=[\.\?\!])\s+', text.strip())
    # 빈 문자열 제거
    sents = [s.strip() for s in parts if s.strip()]
    # 앞 num_sentences개 문장만 반환
    return "\n".join(sents[:num_sentences])

def summarize_pdf_bytes(pdf_bytes: bytes, num_sentences: int) -> str:
    reader = PdfReader(io.BytesIO(pdf_bytes))
    full_text = []
    for p in reader.pages:
        t = p.extract_text()
        if t:
            full_text.append(t)
    doc = "\n".join(full_text)
    if not doc.strip():
        return "PDF에서 텍스트를 추출할 수 없습니다."
    # (원한다면 길이 제한: doc = doc[:2000])
    return simple_summary(doc, num_sentences)

# ─── UI: 사이드바 업로드 ────────────────────────────────────
st.sidebar.header("📤 PDF 업로드 & GCS 저장")
uploaded = st.sidebar.file_uploader("PDF 선택", type="pdf")
if uploaded:
    upload_pdf_to_gcs(uploaded.name, uploaded.read())
    st.sidebar.success(f"✅ GCS에 저장됨: {uploaded.name}")

# ─── UI: 메인 – GCS 목록 & 요약 ─────────────────────────────
st.header("📑 GCS에 저장된 PDF 및 요약")
pdf_list = list_pdfs()
num_sentences = st.sidebar.slider("요약할 문장 수", 1, 10, 5)

if not pdf_list:
    st.info("GCS의 pdfs/ 폴더에 PDF를 업로드해 보세요.")
else:
    for name in sorted(pdf_list):
        st.subheader(name)
        if st.button(f"요약 보기: {name}", key=name):
            with st.spinner("요약 생성 중…"):
                data    = download_pdf_from_gcs(name)
                summary = summarize_pdf_bytes(data, num_sentences)
            st.text_area("요약 결과", summary, height=200)
        st.markdown("---")
