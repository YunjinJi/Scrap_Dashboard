import io, json, base64, streamlit as st
from PyPDF2 import PdfReader
from google.cloud import storage
from google.oauth2 import service_account
import google.generativeai as genai
from tenacity import retry, stop_after_attempt, wait_exponential

# ─── 페이지 설정 ────────────────────────────────────────────
st.set_page_config(page_title="PDF → Gemini 2.0 Flash 요약", layout="wide")
st.title("📄 PDF 업로드 & Gemini 2.0 Flash 요약 (3줄)")

# ─── GCS 인증 ────────────────────────────────────────────────
gcs_b64     = st.secrets["GCS_SA_KEY_B64"]
gcs_info    = json.loads(base64.b64decode(gcs_b64))
bucket_name = st.secrets["GCS_BUCKET_NAME"]
gcs_creds   = service_account.Credentials.from_service_account_info(gcs_info)
gcs_client  = storage.Client(credentials=gcs_creds, project=gcs_info["project_id"])
bucket      = gcs_client.bucket(bucket_name)

# ─── Generative AI 인증 ─────────────────────────────────────
genai_b64   = st.secrets["GENAI_SA_KEY_B64"]
genai_info  = json.loads(base64.b64decode(genai_b64))
genai_creds = service_account.Credentials.from_service_account_info(genai_info)
genai.configure(api_key=None, credentials=genai_creds)

MODEL_ID = "gemini-2.0-flash@001"

@retry(reraise=True, stop=stop_after_attempt(4), wait=wait_exponential(min=1, max=5))
def summarize_with_gemini(prompt: str) -> str:
    # chat completions API를 사용해서 3줄 요약
    resp = genai.chat.completions.create(
        model=MODEL_ID,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_output_tokens=256,
    )
    return resp.choices[0].message.content.strip()

def summarize_pdf_bytes(pdf_bytes: bytes) -> str:
    reader = PdfReader(io.BytesIO(pdf_bytes))
    full = "".join((p.extract_text() or "") for p in reader.pages)[:1000]
    if not full.strip():
        return "PDF에서 텍스트를 추출할 수 없습니다."
    prompt = "다음 내용을 3줄 이내로 요약해 주세요:\n\n" + full
    try:
        return summarize_with_gemini(prompt)
    except Exception as e:
        st.error(f"요약 실패: {type(e).__name__}: {e}")
        return "요약 생성 실패"

# ─── UI: 사이드바 – PDF 업로드 & GCS 저장 ────────────────────
st.sidebar.header("📤 PDF 업로드")
up = st.sidebar.file_uploader("PDF 파일 선택", type="pdf")
if up:
    bucket.blob(f"pdfs/{up.name}").upload_from_file(io.BytesIO(up.read()), content_type="application/pdf")
    st.sidebar.success(f"✅ GCS에 저장됨: {up.name}")

# ─── UI: 메인 – PDF 목록 & 요약 ───────────────────────────────
st.header("📑 GCS에 저장된 PDF 및 요약 (3줄)")
pdfs = [b.name.split("/",1)[1] for b in gcs_client.list_blobs(bucket_name, prefix="pdfs/") if b.name.endswith(".pdf")]

if not pdfs:
    st.info("GCS 버킷의 pdfs/ 폴더에 PDF를 업로드해 보세요.")
else:
    for name in sorted(pdfs):
        st.subheader(name)
        if st.button(f"요약: {name}", key=name):
            data = bucket.blob(f"pdfs/{name}").download_as_bytes()
            with st.spinner("요약 생성 중…"):
                summary = summarize_pdf_bytes(data)
            st.text_area("📝 요약 결과", summary, height=200)
        st.markdown("---")
