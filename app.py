import io, json, base64, streamlit as st
from PyPDF2 import PdfReader
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer
from google.cloud import storage
from google.oauth2 import service_account

# ─── 페이지 설정 ─────────────────────────────────────────
st.set_page_config(page_title="PDF 요약 + GCS 저장 (무료)", layout="wide")
st.title("📄 PDF 업로드 → GCS 저장 → 추출적 요약")

# ─── 시크릿 로드 & GCS 인증 ───────────────────────────────
# 1) 서비스 계정 JSON을 Base64로 인코딩해 Secrets에 등록
#      GCS_SA_KEY_B64, GCS_BUCKET_NAME
gcs_b64     = st.secrets["GCS_SA_KEY_B64"]
gcs_info    = json.loads(base64.b64decode(gcs_b64))
bucket_name = st.secrets["GCS_BUCKET_NAME"]

# GCS 클라이언트 초기화
gcs_creds  = service_account.Credentials.from_service_account_info(gcs_info)
gcs_client = storage.Client(credentials=gcs_creds, project=gcs_info["project_id"])
bucket     = gcs_client.bucket(bucket_name)

# ─── 유틸: GCS PDF 목록/업로드/다운로드 ───────────────────────
def list_pdfs() -> list[str]:
    """pdfs/ 폴더에서 .pdf 파일명만 리스트로 반환"""
    blobs = gcs_client.list_blobs(bucket_name, prefix="pdfs/")
    return [b.name.split("/",1)[1] for b in blobs if b.name.endswith(".pdf")]

def upload_pdf_to_gcs(name: str, data: bytes):
    """pdfs/{name} 으로 업로드"""
    blob = bucket.blob(f"pdfs/{name}")
    blob.upload_from_file(io.BytesIO(data), content_type="application/pdf")

def download_pdf_from_gcs(name: str) -> bytes:
    """pdfs/{name} 에서 바이트로 다운로드"""
    return bucket.blob(f"pdfs/{name}").download_as_bytes()

# ─── 추출적 요약 함수 ───────────────────────────────────────
def extractive_summary(text: str, num_sentences: int) -> str:
    parser     = PlaintextParser.from_string(text, Tokenizer("korean"))
    summarizer = LexRankSummarizer()
    sents      = summarizer(parser.document, num_sentences)
    return "\n".join(str(s).strip() for s in sents)

def summarize_pdf_bytes(pdf_bytes: bytes, num_sent: int) -> str:
    reader = PdfReader(io.BytesIO(pdf_bytes))
    full    = []
    for p in reader.pages:
        t = p.extract_text()
        if t: full.append(t)
    doc = "\n".join(full)
    if not doc:
        return "PDF에서 텍스트를 추출할 수 없습니다."
    # (원한다면 길이 제한: doc = doc[:2000])
    return extractive_summary(doc, num_sent)

# ─── UI: 사이드바 업로드 ────────────────────────────────────
st.sidebar.header("📤 PDF 업로드 & GCS 저장")
uploaded = st.sidebar.file_uploader("PDF 선택", type="pdf")
if uploaded:
    upload_pdf_to_gcs(uploaded.name, uploaded.read())
    st.sidebar.success(f"✅ GCS에 저장됨: {uploaded.name}")

# ─── UI: 메인 – GCS 목록 & 요약 ─────────────────────────────
st.header("📑 GCS에 저장된 PDF 및 요약")
pdf_list     = list_pdfs()
num_sent     = st.sidebar.slider("요약할 문장 수", 1, 10, 5, key="sentences")

if not pdf_list:
    st.info("GCS의 pdfs/ 폴더에 PDF를 업로드해 보세요.")
else:
    for name in sorted(pdf_list):
        st.subheader(name)
        if st.button(f"요약 보기: {name}", key=name):
            with st.spinner("요약 생성 중…"):
                data    = download_pdf_from_gcs(name)
                summary = summarize_pdf_bytes(data, num_sent)
            st.text_area("요약 결과", summary, height=200)
        st.markdown("---")
