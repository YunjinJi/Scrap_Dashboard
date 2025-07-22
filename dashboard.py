import io, json, base64, streamlit as st
from PyPDF2 import PdfReader
from google.cloud import storage
from google.oauth2 import service_account
import google.generativeai as genai
from tenacity import retry, stop_after_attempt, wait_exponential

st.set_page_config(page_title="PDF → Gemini Flash 요약", layout="wide")
st.title("PDF → gemini-2.0-flash@001 3줄 요약")

# GCS 인증
gcs_b64   = st.secrets["GCS_SA_KEY_B64"]
gcs_info  = json.loads(base64.b64decode(gcs_b64))
gcs_creds = service_account.Credentials.from_service_account_info(gcs_info)
gcs_client= storage.Client(credentials=gcs_creds, project=gcs_info["project_id"])
bucket    = gcs_client.bucket(st.secrets["GCS_BUCKET_NAME"])

# Generative AI 인증
genai_b64   = st.secrets["GENAI_SA_KEY_B64"]
genai_info  = json.loads(base64.b64decode(genai_b64))
genai_creds = service_account.Credentials.from_service_account_info(genai_info)
genai.configure(api_key=None, credentials=genai_creds)

MODEL_ID = "gemini-2.0-flash@001"

@retry(reraise=True, stop=stop_after_attempt(4), wait=wait_exponential(min=1, max=5))
def summarize_with_gemini(prompt: str) -> str:
    # 버전 확인 후 둘 중 동작하는 쪽 주석 해제
    # === A ===
    # resp = genai.chat.completions.create(
    #     model=MODEL_ID,
    #     messages=[{"role":"user","content":prompt}],
    #     temperature=0.3,
    #     max_output_tokens=256,
    # )
    # === B ===
    resp = genai.chat_completions.create(
        model=MODEL_ID,
        messages=[{"role":"user","content":prompt}],
        temperature=0.3,
        max_output_tokens=256,
    )
    return resp.choices[0].message.content.strip()

def summarize_pdf_bytes(pdf_bytes: bytes) -> str:
    text = "".join(p.extract_text() or "" for p in PdfReader(io.BytesIO(pdf_bytes)).pages)[:1000]
    if not text.strip():
        return "PDF에서 텍스트를 추출할 수 없습니다."
    prompt = "다음 내용을 3줄 이내로 요약해 주세요:\n\n" + text
    return summarize_with_gemini(prompt)

# UI
uploaded = st.sidebar.file_uploader("PDF 파일 업로드", type="pdf")
if uploaded:
    bucket.blob(f"pdfs/{uploaded.name}").upload_from_file(io.BytesIO(uploaded.read()), content_type="application/pdf")
    st.sidebar.success(f"{uploaded.name} 업로드 완료")

st.header("저장된 PDF 목록 및 요약")
pdfs = [b.name.split("/",1)[1] for b in gcs_client.list_blobs(bucket.name, prefix="pdfs/") if b.name.endswith(".pdf")]
for name in pdfs:
    st.subheader(name)
    if st.button(f"요약: {name}", key=name):
        data = bucket.blob(f"pdfs/{name}").download_as_bytes()
        with st.spinner("요약 생성 중…"):
            try:
                summary = summarize_pdf_bytes(data)
                st.text_area("요약 결과", summary, height=200)
            except Exception as e:
                st.error(f"요약 실패: {e}")
