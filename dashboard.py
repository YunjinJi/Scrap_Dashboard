import io
import json
import base64
import streamlit as st
import openai
from PyPDF2 import PdfReader
from google.cloud import storage
from tenacity import retry, stop_after_attempt, wait_exponential
from openai.error import RateLimitError

# ─── 페이지 설정 ────────────────────────────────────────────────
st.set_page_config(page_title="GCS PDF 요약", layout="wide")
st.title("📂 GCS PDF 업로드 & 요약")

# ─── 시크릿 로드 ───────────────────────────────────────────────
openai.api_key    = st.secrets["OPENAI_API_KEY"]
b64               = st.secrets["GCS_SA_KEY_B64"]
bucket_name       = st.secrets["GCS_BUCKET_NAME"]

# ─── GCS 클라이언트 인증 ───────────────────────────────────────
sa_info = json.loads(base64.b64decode(b64))
client  = storage.Client.from_service_account_info(sa_info)
bucket  = client.bucket(bucket_name)

# ─── 유틸 함수 ───────────────────────────────────────────────────

def list_pdfs():
    """버킷의 pdfs/ 폴더 내 PDF 이름 목록 반환"""
    return [
        blob.name.split("/", 1)[1]
        for blob in client.list_blobs(bucket, prefix="pdfs/")
        if blob.name.endswith(".pdf")
    ]

def list_summaries():
    """버킷의 summaries/ 폴더 내 요약 텍스트 목록(name->Blob) 반환"""
    return {
        blob.name.split("/", 1)[1]: blob
        for blob in client.list_blobs(bucket, prefix="summaries/")
        if blob.name.endswith("_summary.txt")
    }

def upload_pdf(pdf_name: str, data_bytes: bytes):
    """pdfs/ 폴더에 PDF 업로드"""
    blob = bucket.blob(f"pdfs/{pdf_name}")
    blob.upload_from_file(io.BytesIO(data_bytes), content_type="application/pdf")

def download_pdf_bytes(path: str) -> bytes:
    """지정된 경로의 객체를 바이너리로 다운로드"""
    blob = bucket.blob(path)
    return blob.download_as_bytes()

def upload_summary(name: str, text: str):
    """summaries/ 폴더에 요약 텍스트 업로드"""
    blob = bucket.blob(f"summaries/{name}")
    blob.upload_from_string(text, content_type="text/plain")

# ─── OpenAI 호출 백오프 로직 ────────────────────────────────────

@retry(
    reraise=True,
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=1, max=10)
)
def summarize_with_retry(content: str) -> str:
    resp = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": content}],
        temperature=0.3,
    )
    return resp.choices[0].message.content.strip()

# ─── 요약 생성/조회 ─────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def get_or_create_summary(pdf_name: str, existing: dict) -> str:
    summary_name = pdf_name.replace(".pdf", "_summary.txt")
    # 이미 생성된 요약 있으면 다운로드
    if summary_name in existing:
        return existing[summary_name].download_as_text()
    # PDF 다운로드 및 텍스트 추출
    pdf_bytes = download_pdf_bytes(f"pdfs/{pdf_name}")
    reader    = PdfReader(io.BytesIO(pdf_bytes))
    text      = "\n".join(page.extract_text() or "" for page in reader.pages)
    prompt    = f"다음 PDF를 5문장 이내로 요약해 주세요:\n\n{text[:2000]}"
    try:
        summary = summarize_with_retry(prompt)
    except RateLimitError:
        summary = "⚠️ 요약 요청이 과부하 상태입니다. 잠시 후 다시 시도해주세요."
    # 요약 업로드
    upload_summary(summary_name, summary)
    return summary

# ─── 사이드바: PDF 업로드 ───────────────────────────────────────

st.sidebar.header("📤 PDF 업로드")
uploaded = st.sidebar.file_uploader("PDF 선택", type="pdf")
if uploaded:
    data = uploaded.read()
    upload_pdf(uploaded.name, data)
    st.sidebar.success("✅ 업로드 완료! 페이지를 새로고침하세요.")

# ─── 메인 화면: PDF 목록 및 요약 표시 ────────────────────────────

st.header("📑 저장된 PDF 및 요약")
pdfs       = list_pdfs()
summaries  = list_summaries()

if not pdfs:
    st.info("버킷에 pdfs/ 폴더를 만들고 PDF 파일을 업로드해 보세요.")
else:
    for pdf_name in sorted(pdfs):
        st.subheader(pdf_name)
        summary = get_or_create_summary(pdf_name, summaries)
        st.markdown(f"**요약:** {summary}")
        st.markdown("---")
