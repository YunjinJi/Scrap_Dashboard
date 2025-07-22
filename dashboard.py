# list_models.py
import streamlit as st
import json
import base64
from google.oauth2 import service_account
import google.generativeai as genai

st.set_page_config(page_title="Available Generative AI Models", layout="wide")
st.title("📜 사용 가능한 Generative AI 모델 목록")

# 1) Secrets에 저장된 서비스 계정 키(Base64)를 불러옵니다.
genai_b64  = st.secrets["GENAI_SA_KEY_B64"]
genai_info = json.loads(base64.b64decode(genai_b64))

# 2) Credentials 객체 생성
creds = service_account.Credentials.from_service_account_info(genai_info)

# 3) google-generativeai SDK 초기화
genai.configure(api_key=None, credentials=creds)

# 4) 모델 목록 조회 및 출력
models = genai.list_models()
for m in models:
    st.write(f"- **{m.name}**  ({m.display_name})")
