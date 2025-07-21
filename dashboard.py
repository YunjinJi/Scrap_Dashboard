import io, json, time
import streamlit as st
import openai
from PyPDF2 import PdfReader
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseUpload, MediaIoBaseDownload

# 페이지 설정
st.set_page_config(page_title="Google Drive PDF 요약", layout="wide")
st.title("📂 Google Drive PDF 업로드 & 요약")

# 1) OpenAI 키
openai.api_key = st.secrets["OPENAI_API_KEY"]

# 2) Drive API 클라이언트
sa_info   = json.loads(st.secrets["GDRIVE_SA_KEY"])
folder_id = st.secrets["GDRIVE_FOLDER_ID"]
creds     = Credentials.from_service_account_info(
    sa_info, scopes=["https://www.googleapis.com/auth/drive.file"])
drive     = build("drive", "v3", credentials=creds)

# ─── 헬퍼 함수 ────────────────────────────────────────────────────

def list_pdfs_in_folder():
    """폴더 내 PDF 목록(id, name) 반환"""
    resp = drive.files().list(
        q=f"'{folder_id}' in parents and mimeType='application/pdf'",
        fields="files(id,name)"
    ).execute()
    return resp.get("files", [])

def list_summaries_in_folder():
    """폴더 내 _summary.txt 목록(name→id 매핑) 반환"""
    resp = drive.files().list(
        q=f"'{folder_id}' in parents and name contains '_summary.txt'",
        fields="files(id,name)"
    ).execute()
    return {f['name']: f['id'] for f in resp.get('files', [])}

def download_file_bytes(file_id):
    """Drive 파일을 바이너리로 다운로드"""
    request = drive.files().get_media(fileId=file_id)
    fh = io.BytesIO()
    downloader = MediaIoBaseDownload(fh, request)
    done = False
    while not done:
        _, done = downloader.next_chunk()
    fh.seek(0)
    return fh.read()

def upload_summary_file(filename, summary_text):
    """요약 텍스트를 _summary.txt로 업로드"""
    summary_name = filename + "_summary.txt"
    media = MediaIoBaseUpload(
        io.BytesIO(summary_text.encode("utf-8")),
        mimetype="text/plain"
    )
    metadata = {"name": summary_name, "parents": [folder_id]}
    drive.files().create(body=metadata, media_body=media).execute()

@st.cache_data(show_spinner=False)
def get_or_create_summary(pdf_meta, existing_summaries):
    """요약 이미 있으면 다운로드, 없으면 생성→업로드→반환"""
    name = pdf_meta['name']
    file_id = pdf_meta['id']
    summary_name = name + '_summary.txt'

    # 이미 있으면 다운로드
    if summary_name in existing_summaries:
        data = download_file_bytes(existing_summaries[summary_name])
        return data.decode('utf-8')

    # PDF 내려받아 텍스트 추출
    pdf_bytes = download_file_bytes(file_id)
    text = "\n".join(page.extract_text() or "" for page in PdfReader(io.BytesIO(pdf_bytes)).pages)

    # OpenAI 요약 요청
    resp = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{
            "role": "user",
            "content": f"다음 PDF를 5문장 이내로 요약해 주세요:\n\n{text[:2000]}"
        }],
        temperature=0.3
    )
    summary = resp.choices[0].message.content.strip()

    # 요약 업로드
    upload_summary_file(name, summary)
    return summary

# ─── 사이드바: PDF 업로드 ────────────────────────────────────────

st.sidebar.header("📤 PDF 업로드")
uploaded = st.sidebar.file_uploader("새 PDF 업로드", type=["pdf"])
if uploaded:
    data = uploaded.read()
    meta = {"name": uploaded.name, "parents": [folder_id]}
    media = MediaIoBaseUpload(io.BytesIO(data), mimetype="application/pdf")
    drive.files().create(body=meta, media_body=media, fields="id").execute()
    st.sidebar.success("✅ Drive 업로드 완료! 페이지를 새로고침하세요.")

# ─── 메인 화면: PDF 및 요약 표시 ─────────────────────────────────

st.header("📑 저장된 PDF 목록 및 요약")
pdfs = list_pdfs_in_folder()
summaries = list_summaries_in_folder()

if not pdfs:
    st.info("폴더에 PDF가 없습니다. 사이드바에서 업로드해 주세요.")
else:
    for pdf in pdfs:
        st.subheader(pdf['name'])
        summary_text = get_or_create_summary(pdf, summaries)
        st.markdown(f"**요약:** {summary_text}")
        st.markdown("---")
