import os
import io
import sqlite3
import pandas as pd
import streamlit as st
from PyPDF2 import PdfReader
import openai

# ————— 설정 —————
st.set_page_config(page_title="PDF 요약 대시보드", layout="wide")
openai.api_key = st.secrets["OPENAI_API_KEY"]

# ————— DB 초기화 —————
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "ax_summaries.db")

def init_db(path):
    conn = sqlite3.connect(path)
    c = conn.cursor()
    # 1) 업로드된 PDF 저장용 테이블
    c.execute("""
        CREATE TABLE IF NOT EXISTS pdfs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT,
            content BLOB,
            upload_at TEXT
        )
    """)
    # 2) 요약 결과 저장용 테이블
    c.execute("""
        CREATE TABLE IF NOT EXISTS summaries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            pdf_id INTEGER,
            summary TEXT,
            summarized_at TEXT,
            FOREIGN KEY(pdf_id) REFERENCES pdfs(id)
        )
    """)
    conn.commit()
    conn.close()

init_db(DB_PATH)

# ————— 유틸 함수 —————
def save_pdf_to_db(path, filename, data: bytes):
    conn = sqlite3.connect(path)
    c = conn.cursor()
    c.execute(
        "INSERT INTO pdfs (filename, content, upload_at) VALUES (?, ?, datetime('now'))",
        (filename, sqlite3.Binary(data))
    )
    conn.commit()
    last_id = c.lastrowid
    conn.close()
    return last_id

def extract_text_from_pdf(data: bytes) -> str:
    reader = PdfReader(io.BytesIO(data))
    texts = []
    for page in reader.pages:
        txt = page.extract_text()
        if txt:
            texts.append(txt)
    return "\n".join(texts)

def summarize_with_openai(text: str) -> str:
    # 길면 앞뒤 일부만 잘라내기(비용 절감용)
    snippet = text[:1500] + "\n\n...(중략)...\n\n" + text[-1500:]
    resp = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": f"다음 PDF 내용을 5문장 이내로 요약해주세요:\n\n{snippet}"}],
        temperature=0.3
    )
    return resp.choices[0].message.content.strip()

def save_summary_to_db(path, pdf_id, summary: str):
    conn = sqlite3.connect(path)
    c = conn.cursor()
    c.execute(
        "INSERT INTO summaries (pdf_id, summary, summarized_at) VALUES (?, ?, datetime('now'))",
        (pdf_id, summary)
    )
    conn.commit()
    conn.close()

# ————— 사이드바 : PDF 업로드 폼 —————
st.sidebar.header("📤 PDF 업로드 & 요약")
uploaded = st.sidebar.file_uploader(
    "PDF 파일을 선택하세요", type=["pdf"], accept_multiple_files=False
)

if uploaded:
    with st.spinner("PDF 저장 중…"):
        data = uploaded.read()
        pdf_id = save_pdf_to_db(DB_PATH, uploaded.name, data)
    with st.spinner("텍스트 추출 및 요약 중…"):
        raw_text = extract_text_from_pdf(data)
        summary = summarize_with_openai(raw_text)
        save_summary_to_db(DB_PATH, pdf_id, summary)
    st.sidebar.success("✅ 업로드 및 요약 완료!")

# ————— 메인 화면 : 요약 리스트 —————
st.title("📑 업로드된 PDF & 요약 결과")
conn = sqlite3.connect(DB_PATH)
pdfs_df = pd.read_sql("SELECT * FROM pdfs ORDER BY id DESC", conn)
summ_df = pd.read_sql("SELECT * FROM summaries ORDER BY id DESC", conn)
conn.close()

if pdfs_df.empty:
    st.info("업로드된 PDF가 아직 없습니다.")
else:
    for _, pdf in pdfs_df.iterrows():
        st.markdown(f"### 📄 {pdf['filename']}  (업로드: {pdf['upload_at']})")
        # 해당 PDF의 요약 불러오기
        sum_rows = summ_df[summ_df['pdf_id'] == pdf['id']]
        if sum_rows.empty:
            st.write("> 아직 요약이 생성되지 않았습니다.")
        else:
            for _, s in sum_rows.iterrows():
                st.markdown(f"- **요약 ({s['summarized_at']}):**  \n  {s['summary']}")
        st.markdown("---")
