import io
import streamlit as st
from PyPDF2 import PdfReader
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer

st.set_page_config(page_title="PDF 추출적 요약 (무료)", layout="wide")
st.title("📄 PDF 업로드 → 추출적 요약 (무료)")

# 사이드바: 요약 문장 수 선택
num_sentences = st.sidebar.slider("요약 문장 수", 1, 10, 5)

# PDF 업로드
uploaded = st.file_uploader("PDF 파일 올리기", type="pdf")
if not uploaded:
    st.info("좌측에서 PDF를 올려 주세요.")
    st.stop()

# 텍스트 추출
reader = PdfReader(io.BytesIO(uploaded.read()))
full_text = []
for page in reader.pages:
    txt = page.extract_text()
    if txt:
        full_text.append(txt)
doc = "\n".join(full_text)

if not doc.strip():
    st.error("PDF에서 텍스트를 추출할 수 없습니다.")
    st.stop()

# 추출적 요약
parser = PlaintextParser.from_string(doc, Tokenizer("korean"))
summarizer = LexRankSummarizer()
summary_sentences = summarizer(parser.document, num_sentences)
summary = "\n".join(str(s).strip() for s in summary_sentences)

# 결과 출력
st.subheader("🔍 추출적 요약 결과")
st.write(summary)

with st.expander("원문 일부 보기"):
    st.write(doc[:2000] + "…")
