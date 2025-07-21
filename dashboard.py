import io
import json
import base64

import streamlit as st
import openai
from PyPDF2 import PdfReader
from google.cloud import storage
from tenacity import retry, stop_after_attempt, wait_exponential

# ─── 페이지 설정 ────────────────────────────────────────────────
st.set_page_config(page_title="GCS PDF 요약 (디버그 모드)", layout="wide")
st.title("📂 GCS PDF 업로드 & 요약 (디버그)")

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
    st.write("DEBUG ▶ list_pdfs 호출")
    blobs = client.list_blobs(bucket, prefix="pdfs/")
    pdf_names: list[str] = []
    for blob in blobs:
        name = blob.name
        if name.endswith(".pdf"):
            pdf_names.append(name.split("/", 1)[1])
    st.write("DEBUG ▶ list_pdfs 반환:", pdf_names)
    return pdf_names

# ─── 유틸: 요약 목록 ────────────────────────────────────────────
def list_summaries() -> dict[str, storage.Blob]:
    st.write("DEBUG ▶ list_summaries 호출")
    blobs = client.list_blobs(bucket, prefix="summaries/")
    summary_map: dict[str, storage.Blob] = {}
    for blob in blobs:
        name = blob.name
        if name.endswith("_summary.txt"):
            key = name.split("/", 1)[1]
            summary_map[key] = blob
    st.write("DEBUG ▶ list_summaries 반환:", list(summary_map.keys()))
    return summary_map

# ─── 유틸: PDF 업로드 ────────────────────────────────────────────
def upload_pdf(pdf_name: str, data_bytes: bytes) -> None:
    st.write(f"DEBUG ▶ upload_pdf 호출: {pdf_name}, {len(data_bytes)} bytes")
    blob = bucket.blob(f"pdfs/{pdf_name}")
    blob.upload_from_file(io.BytesIO(data_bytes), content_type="application/pdf")
    st.write("DEBUG ▶ upload_pdf 완료")

# ─── 유틸: 다운로드 ──────────────────────────────────────────────
def download_pdf_bytes(path: str) -> bytes:
    st.write("DEBUG ▶ download_pdf_bytes 호출:", path)
    blob = bucket.blob(path)
    data = blob.download_as_bytes()
    st.write("DEBUG ▶ download_pdf_bytes 반환 크기:", len(data))
    return data

# ─── 유틸: 요약 업로드 ───────────────────────────────────────────
def upload_summary(name: str, text: str) -> None:
    st.write("DEBUG ▶ upload_summary 호출:", name)
    blob = bucket.blob(f"summaries/{name}")
    blob.upload_from_string(text, content_type="text/plain")
    st.write("DEBUG ▶ upload_summary 완료")

# ─── OpenAI 재시도 로직 ─────────────────────────────────────────
@retry(reraise=True, stop=stop_after_attempt(5), wait=wait_exponential(min=1, max=10))
def summarize_with_retry(prompt: str) -> str:
    st.write("DEBUG ▶ summarize_with_retry 호출")
    resp = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )
    text = resp.choices[0].message.content.strip()
    st.write("DEBUG ▶ summarize_with_retry 반환:", text[:50], "...")
    return text

# ─── 요약 생성/조회 (캐시 제거, 디버그 모드) ───────────────────────
def get_or_create_summary(pdf_name: str, existing: dict[str, storage.Blob]) -> str:
    st.write("DEBUG ▶ get_or_create_summary 호출:", pdf_name)
    summary_filename = pdf_name.replace(".pdf", "_summary.txt")

    # 이미 생성된 요약이 있으면 다운로드
    if summary_filename in existing:
        st.write("DEBUG ▶ 기존 요약 사용:", summary_filename)
        return existing[summary_filename].download_as_text()

    # 새로 생성: PDF 다운로드 → 텍스트 추출 → 요약 → 업로드
    pdf_bytes = download_pdf_bytes(f"pdfs/{pdf_name}")
    reader    = PdfReader(io.BytesIO(pdf_bytes))
    text      = "\n".join(page.extract_text() or "" for page in reader.pages)
    prompt    = f"다음 PDF를 5문장 이내로 요약해 주세요:\n\n{text[:2000]}"
    st.write("DEBUG ▶ 프롬프트:", prompt[:100], "...")

    # 실제 호출 및 예외 노출
    summary = summarize_with_retry(prompt)

    upload_summary(summary_filename, summary)
    return summary

# ─── 사이드바: PDF 업로드 UI ────────────────────────────────────
st.sidebar.header("📤 PDF 업로드")
uploaded = st.sidebar.file_uploader("PDF 선택", type="pdf")
if uploaded:
    data = uploaded.read()
    upload_pdf(uploaded.name, data)
    st.sidebar.success("✅ 업로드 완료! 페이지를 새로침하세요.")

# ─── 메인: PDF 목록 및 요약 표시 ─────────────────────────────────
st.header("📑 저장된 PDF 및 요약")
pdfs      = list_pdfs()
summaries = list_summaries()

if not pdfs:
    st.info("버킷의 pdfs/ 폴더에 PDF 파일을 업로드해 보세요.")
else:
    for pdf_name in sorted(pdfs):
        st.subheader(pdf_name)
        summary = get_or_create_summary(pdf_name, summaries)
        st.markdown(f"**요약:** {summary}")
        st.markdown("---")
