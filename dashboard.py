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
openai.api_key  = st.secrets["OPENAI_API_KEY"]
b64             = st.secrets["GCS_SA_KEY_B64"]
bucket_name     = st.secrets["GCS_BUCKET_NAME"]

# ─── GCS 클라이언트 인증 ───────────────────────────────────────
sa_info = json.loads(base64.b64decode(b64))
client  = storage.Client.from_service_account_info(sa_info)
bucket  = client.bucket(bucket_name)

# ─── 유틸: PDF 목록 ─────────────────────────────────────────────
def list_pdfs() -> list[str]:
    blobs = client.list_blobs(bucket, prefix="pdfs/")
    return [blob.name.split("/",1)[1] for blob in blobs if blob.name.endswith(".pdf")]

# ─── 유틸: 요약 목록 ────────────────────────────────────────────
def list_summaries() -> dict[str, storage.Blob]:
    blobs = client.list_blobs(bucket, prefix="summaries/")
    return {blob.name.split("/",1)[1]: blob for blob in blobs if blob.name.endswith("_summary.txt")}

# ─── 유틸: PDF 업로드 ────────────────────────────────────────────
def upload_pdf(pdf_name: str, data: bytes) -> None:
    blob = bucket.blob(f"pdfs/{pdf_name}")
    blob.upload_from_file(io.BytesIO(data), content_type="application/pdf")

# ─── 유틸: 다운로드 ──────────────────────────────────────────────
def download_bytes(path: str) -> bytes:
    return bucket.blob(path).download_as_bytes()

# ─── 유틸: 요약 업로드 ───────────────────────────────────────────
def upload_summary(name: str, text: str) -> None:
    blob = bucket.blob(f"summaries/{name}")
    blob.upload_from_string(text, content_type="text/plain")

# ─── OpenAI 재시도 로직 ─────────────────────────────────────────
@retry(reraise=True, stop=stop_after_attempt(5), wait=wait_exponential(min=1, max=10))
def summarize_with_retry(prompt: str) -> str:
    resp = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role":"user","content":prompt}],
        temperature=0.3,
    )
    return resp.choices[0].message.content.strip()

# ─── 요약 생성/조회 ─────────────────────────────────────────────
@st.cache_data(show_spinner=False, hash_funcs={dict: lambda _: None})
def get_or_create_summary(pdf_name: str, existing: dict[str, storage.Blob]) -> str:
    summary_file = pdf_name.replace(".pdf","_summary.txt")
    # 기존 요약이 “⚠️”로 시작하면 새로 생성
    if summary_file in existing:
        text = existing[summary_file].download_as_text()
        if not text.startswith("⚠️"):
            return text

    # PDF 텍스트 추출 (최대 1000자)
    data = download_bytes(f"pdfs/{pdf_name}")
    reader = PdfReader(io.BytesIO(data))
    content = ""
    for page in reader.pages:
        content += page.extract_text() or ""
        if len(content) > 1000:
            content = content[:1000]
            break

    prompt = f"다음 PDF를 5문장 이내로 요약해 주세요:\n\n{content}"
    try:
        summary = summarize_with_retry(prompt)
    except RateLimitError:
        st.error("⚠️ OpenAI 속도 제한에 걸렸습니다. 잠시 후 다시 시도해 주세요.")
        return "요약을 생성하지 못했습니다."
    except Exception as e:
        st.error(f"❌ 요약 중 예외 발생: {e}")
        return "요약을 생성하지 못했습니다."

    upload_summary(summary_file, summary)
    return summary

# ─── 사이드바: PDF 업로드 ───────────────────────────────────────
st.sidebar.header("📤 PDF 업로드")
uploaded = st.sidebar.file_uploader("새 PDF 업로드", type="pdf")
if uploaded:
    data = uploaded.read()
    upload_pdf(uploaded.name, data)
    st.sidebar.success("✅ 업로드 완료! 페이지를 새로고침하세요.")

# ─── 메인 화면 ──────────────────────────────────────────────────
st.header("📑 저장된 PDF 및 요약")
pdfs = list_pdfs()
summaries = list_summaries()

if not pdfs:
    st.info("버킷의 pdfs/ 폴더에 PDF를 업로드해 보세요.")
else:
    for name in sorted(pdfs):
        st.subheader(name)
        summary = get_or_create_summary(name, summaries)
        st.markdown(f"**요약:** {summary}")
        st.markdown("---")
