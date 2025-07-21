import io, json, base64
import streamlit as st
import openai
from PyPDF2 import PdfReader
from google.cloud import storage

# 페이지 & 시크릿
st.set_page_config(page_title="GCS PDF 요약", layout="wide")
openai.api_key      = st.secrets["OPENAI_API_KEY"]

# GCS 인증
b64                = st.secrets["GCS_SA_KEY_B64"]
sa_info            = json.loads(base64.b64decode(b64))
client             = storage.Client.from_service_account_info(sa_info)
bucket             = client.bucket(st.secrets["GCS_BUCKET_NAME"])

# 유틸: 버킷 내 PDF 리스트
def list_pdfs():
    return [blob.name for blob in client.list_blobs(bucket, prefix="pdfs/") if blob.name.endswith(".pdf")]

# 유틸: 요약 파일 리스트(name→blob)
def list_summaries():
    return {blob.name.replace("summaries/",""): blob for blob in client.list_blobs(bucket, prefix="summaries/")}

# 유틸: PDF 업로드
def upload_pdf(pdf_name, data_bytes):
    blob = bucket.blob(f"pdfs/{pdf_name}")
    blob.upload_from_file(io.BytesIO(data_bytes), content_type="application/pdf")

# 유틸: PDF 다운로드
def download_pdf_bytes(name):
    blob = bucket.blob(name)
    return blob.download_as_bytes()

# 유틸: 요약 업로드
def upload_summary(name, text):
    blob = bucket.blob(f"summaries/{name}")
    blob.upload_from_string(text, content_type="text/plain")

# 요약 생성/조회
@st.cache_data
def get_or_create_summary(pdf_name, existing):
    summ_name = pdf_name.replace(".pdf","_summary.txt")
    if summ_name in existing:
        return existing[summ_name].download_as_text()
    # 새로 생성
    pdf_bytes = download_pdf_bytes(f"pdfs/{pdf_name}")
    text      = "\n".join(p.extract_text() or "" for p in PdfReader(io.BytesIO(pdf_bytes)).pages)
    resp      = openai.chat.completions.create(
                   model="gpt-4o-mini",
                   messages=[{"role":"user","content":f"다음 PDF를 5문장 이내로 요약해 주세요:\n\n{text[:2000]}"}]
               )
    summary   = resp.choices[0].message.content.strip()
    upload_summary(pdf_name.replace(".pdf","_summary.txt"), summary)
    return summary

# 사이드바: PDF 업로드
st.sidebar.header("📤 PDF 업로드")
uploaded = st.sidebar.file_uploader("PDF 선택", type="pdf")
if uploaded:
    data = uploaded.read()
    upload_pdf(uploaded.name, data)
    st.sidebar.success("업로드 완료! 새로고침하세요.")

# 메인: PDF 리스트 & 요약
st.header("📑 저장된 PDF 및 요약")
pdfs      = [name.split("/")[-1] for name in list_pdfs()]
summaries = list_summaries()

if not pdfs:
    st.info("버킷에 pdfs/ 폴더를 만들고 PDF를 업로드해 보세요.")
else:
    for pdf_name in sorted(pdfs):
        st.subheader(pdf_name)
        summary = get_or_create_summary(pdf_name, summaries)
        st.markdown(f"**요약:** {summary}")
        st.markdown("---")
