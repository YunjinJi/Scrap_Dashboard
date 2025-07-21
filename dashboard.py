import io
import json
import base64
import streamlit as st
import openai
from PyPDF2 import PdfReader
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseUpload, MediaIoBaseDownload
from googleapiclient.errors import HttpError

# ─── 페이지 설정 ────────────────────────────────────────────────
st.set_page_config(page_title="Google Drive PDF 요약", layout="wide")
st.title("📂 Google Shared Drive PDF 업로드 & 요약")

# ─── 시크릿 로드 ───────────────────────────────────────────────
openai.api_key = st.secrets["OPENAI_API_KEY"]
b64 = st.secrets["GDRIVE_SA_KEY_B64"]
sa_info = json.loads(base64.b64decode(b64))
folder_id = st.secrets["GDRIVE_FOLDER_ID"]

# ─── Drive 클라이언트 인증 ─────────────────────────────────────
creds = Credentials.from_service_account_info(
    sa_info,
    scopes=["https://www.googleapis.com/auth/drive.file"]
)
drive = build("drive", "v3", credentials=creds)

# ─── 헬퍼 함수들 ─────────────────────────────────────────────────

def list_pdfs_in_folder():
    try:
        resp = drive.files().list(
            q=f"'{folder_id}' in parents and mimeType='application/pdf'",
            includeItemsFromAllDrives=True,
            supportsAllDrives=True,
            fields="files(id,name)"
        ).execute()
        return resp.get("files", [])
    except HttpError as e:
        detail = e.error_details or (e.content.decode() if e.content else "<no detail>")
        st.error(f"PDF 목록 조회 에러 {e.status_code}: {detail}")
        return []

def list_summaries_in_folder():
    try:
        resp = drive.files().list(
            q=f"'{folder_id}' in parents and name contains '_summary.txt'",
            includeItemsFromAllDrives=True,
            supportsAllDrives=True,
            fields="files(id,name)"
        ).execute()
        return {f["name"]: f["id"] for f in resp.get("files", [])}
    except HttpError as e:
        detail = e.error_details or (e.content.decode() if e.content else "<no detail>")
        st.error(f"요약 목록 조회 에러 {e.status_code}: {detail}")
        return {}

def download_file_bytes(file_id):
    fh = io.BytesIO()
    downloader = MediaIoBaseDownload(fh, drive.files().get_media(fileId=file_id))
    done = False
    while not done:
        _, done = downloader.next_chunk()
    fh.seek(0)
    return fh.read()

def upload_summary_file(filename, summary_text):
    summary_name = f"{filename}_summary.txt"
    media = MediaIoBaseUpload(io.BytesIO(summary_text.encode("utf-8")), mimetype="text/plain")
    try:
        drive.files().create(
            body={"name": summary_name, "parents":[folder_id]},
            media_body=media,
            supportsAllDrives=True
        ).execute()
    except HttpError as e:
        detail = e.error_details or (e.content.decode() if e.content else "<no detail>")
        st.error(f"요약 업로드 에러 {e.status_code}: {detail}")

@st.cache_data(show_spinner=False)
def get_or_create_summary(pdf_meta, existing_summaries):
    name = pdf_meta["name"]
    file_id = pdf_meta["id"]
    summary_name = f"{name}_summary.txt"

    if summary_name in existing_summaries:
        data = download_file_bytes(existing_summaries[summary_name])
        return data.decode("utf-8")

    pdf_bytes = download_file_bytes(file_id)
    text = "\n".join(p.extract_text() or "" for p in PdfReader(io.BytesIO(pdf_bytes)).pages)

    resp = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{
            "role": "user",
            "content": f"다음 PDF를 5문장 이내로 요약해 주세요:\n\n{text[:2000]}"
        }],
        temperature=0.3
    )
    summary = resp.choices[0].message.content.strip()
    upload_summary_file(name, summary)
    return summary

# ─── 사이드바: PDF 업로드 ───────────────────────────────────────
st.sidebar.header("📤 PDF 업로드")
uploaded = st.sidebar.file_uploader("새 PDF 업로드", type="pdf")
if uploaded:
    try:
        data = uploaded.read()
        media = MediaIoBaseUpload(io.BytesIO(data), mimetype="application/pdf")
        res = drive.files().create(
            body={"name": uploaded.name, "parents":[folder_id]},
            media_body=media,
            supportsAllDrives=True,
            fields="id"
        ).execute()
        st.sidebar.success(f"✅ 업로드 완료 (fileId: {res.get('id')})\n페이지를 새로고침하세요.")
    except HttpError as e:
        detail = e.error_details or (e.content.decode() if e.content else "<no detail>")
        st.sidebar.error(f"업로드 에러 {e.status_code}: {detail}")

# ─── 메인: PDF 목록 및 요약 표시 ─────────────────────────────────
st.header("📑 저장된 PDF 및 요약")
pdfs = list_pdfs_in_folder()
summaries = list_summaries_in_folder()

if not pdfs:
    st.info("폴더에 PDF가 없습니다. 사이드바에서 업로드해 주세요.")
else:
    for pdf in pdfs:
        st.subheader(pdf["name"])
        summary_text = get_or_create_summary(pdf, summaries)
        st.markdown(f"**요약:**  {summary_text}")
        st.markdown("---")
