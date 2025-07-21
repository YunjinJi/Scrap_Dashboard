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
# OpenAI API 키
openai.api_key = st.secrets["OPENAI_API_KEY"]

# 서비스 계정 JSON (Base64) → 디코딩
b64       = st.secrets["GDRIVE_SA_KEY_B64"]
sa_info   = json.loads(base64.b64decode(b64))

# Shared Drive 폴더 ID
folder_id = st.secrets["GDRIVE_FOLDER_ID"]

# ─── Google Drive 클라이언트 인증 ─────────────────────────────
creds = Credentials.from_service_account_info(
    sa_info,
    scopes=["https://www.googleapis.com/auth/drive.file"]
)
drive = build("drive", "v3", credentials=creds)

# ─── (선택) 접근 가능한 Shared Drives 확인 ────────────────────
# drives = drive.drives().list(fields="drives(id,name)").execute().get("drives", [])
# st.write("🔍 Shared Drives:", drives)

# ─── 헬퍼 함수들 ─────────────────────────────────────────────────

def list_pdfs_in_folder():
    """공유 드라이브 폴더 내 PDF 목록 조회"""
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
    """공유 드라이브 폴더 내 요약 텍스트 목록 조회"""
    try:
        resp = drive.files().list(
            q=f"'{folder_id}' in parents and name contains '_summary.txt'",
            includeItemsFromAllDrives=True,
            supportsAllDrives=True,
            fields="files(id,name)"
        ).execute()
        return {f['name']: f['id'] for f in resp.get("files", [])}
    except HttpError as e:
        detail = e.error_details or (e.content.decode() if e.content else "<no detail>")
        st.error(f"요약 목록 조회 에러 {e.status_code}: {detail}")
        return {}

def download_file_bytes(file_id):
    """Drive 파일을 바이너리로 다운로드"""
    fh = io.BytesIO()
    downloader = MediaIoBaseDownload(fh, drive.files().get_media(fileId=file_id))
    done = False
    while not done:
        _, done = downloader.next_chunk()
    fh.seek(0)
    return fh.read()

def upload_summary_file(filename, summary_text):
    """요약 텍스트를 공유 드라이브에 _summary.txt로 업로드"""
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
    """이미 생성된 요약이 있으면 다운로드, 없으면 생성→업로드→반환"""
    name = pdf_meta["name"]
    file_id = pdf_meta["id"]
    summary_name = f"{name}_summary.txt"

    if summary_name in existing_summaries:
        data = download_file_bytes(existing_summaries[summary_name])
        return data.decode("utf-8")

    pdf_bytes = download_file_bytes(file_id)
    text = "\n".join(p.extract_text() or "" for p in PdfReader(io.BytesIO(pdf_bytes)).pages)

    resp = openai.chat.completions.create(
        model="gp
