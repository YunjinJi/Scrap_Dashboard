# app.py

import io, json, base64, streamlit as st
from PyPDF2 import PdfReader
from google.cloud import storage, aiplatform_v1
from google.oauth2 import service_account
import google.generativeai as genai
from tenacity import retry, stop_after_attempt, wait_exponential

# ─── 페이지 설정 ────────────────────────────────────────────
st.set_page_config(page_title="PDF 요약 & 모델 조회", layout="wide")
st.title("📄 PDF 요약 & Generative AI 모델 조회")

# ─── 시크릿 로드 & GCS 인증 ─────────────────────────────────
gcs_b64     = st.secrets["GCS_SA_KEY_B64"]
gcs_info    = json.loads(base64.b64decode(gcs_b64))
bucket_name = st.secrets["GCS_BUCKET_NAME"]

gcs_creds   = service_account.Credentials.from_service_account_info(gcs_info)
gcs_client  = storage.Client(credentials=gcs_creds, project=gcs_info["project_id"])
bucket      = gcs_client.bucket(bucket_name)

# ─── Generative AI 인증 & 클라이언트 설정 ──────────────────
genai_b64   = st.secrets["GENAI_SA_KEY_B64"]
genai_info  = json.loads(base64.b64decode(genai_b64))
genai_creds = service_account.Credentials.from_service_account_info(genai_info)
genai.configure(api_key=None, credentials=genai_creds)

# Vertex AI Prediction 클라이언트 (text-bison 사용 시)
prediction_client = aiplatform_v1.PredictionServiceClient(credentials=genai_creds)
project_id = genai_info["project_id"]
location   = "us-central1"

# ─── 1) 모델 리스트 조회 ─────────────────────────────────────
st.subheader("1️⃣ 사용 가능한 Generative AI 모델")
models = genai.list_models()
model_names = [m.name for m in models]
selected_model = st.selectbox("모델 선택", model_names)

# 만약 text-bison 계열을 선택했다면 PredictionServiceClient 사용할 수 있도록 endpoint 구성
use_prediction = selected_model.startswith("text-") or selected_model.startswith("bison")
if use_prediction:
    endpoint = f"projects/{project_id}/locations/{location}/publishers/google/models/{selected_model}"

# ─── GCS PDF 목록 / 업로드 ────────────────────────────────────
st.subheader("2️⃣ PDF 업로드 & 목록")
uploaded = st.file_uploader("새 PDF 업로드", type="pdf")
if uploaded:
    blob = bucket.blob(f"pdfs/{uploaded.name}")
    blob.upload_from_file(io.BytesIO(uploaded.read()), content_type="application/pdf")
    st.success(f"✅ GCS에 저장됨: {uploaded.name}")

def list_pdfs():
    return [b.name.split("/",1)[1] for b in gcs_client.list_blobs(bucket_name, prefix="pdfs/") if b.name.endswith(".pdf")]

pdfs = list_pdfs()
selected_pdf = st.selectbox("요약할 PDF 선택", pdfs)

# ─── 3) 요약 실행 ───────────────────────────────────────────
st.subheader("3️⃣ 선택한 PDF 3줄 요약")
if st.button("요약 시작"):
    pdf_bytes = bucket.blob(f"pdfs/{selected_pdf}").download_as_bytes()
    # 텍스트 추출
    reader = PdfReader(io.BytesIO(pdf_bytes))
    txt = "".join(p.extract_text() or "" for p in reader.pages)[:1000]
    if not txt.strip():
        st.error("PDF에서 텍스트가 추출되지 않았습니다.")
    else:
        prompt = "다음 내용을 3줄 이내로 요약해 주세요:\n\n" + txt

        @retry(reraise=True, stop=stop_after_attempt(4), wait=wait_exponential(min=1, max=5))
        def summarize(prompt: str) -> str:
            if use_prediction:
                res = prediction_client.predict(
                    endpoint=endpoint,
                    instances=[{"content": prompt}],
                    parameters={"temperature":0.3, "maxOutputTokens":256},
                )
                return res.predictions[0].get("content","").strip()
            else:
                # 텍스트 생성 API
                resp = genai.text.completions.create(
                    model=selected_model,
                    prompt=prompt,
                    temperature=0.3,
                    max_output_tokens=256,
                )
                return resp.choices[0].text.strip()

        with st.spinner("요약 생성 중…"):
            try:
                summary = summarize(prompt)
                st.text_area("📝 요약 결과", summary, height=200)
            except Exception as e:
                st.error(f"요약 실패: {e}")
