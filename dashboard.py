import os
import sqlite3
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# ————— 페이지 설정 —————
st.set_page_config(page_title="SK AX 흐름 대시보드", layout="wide")

# ————— DB 경로 & 초기화 —————
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
db_path = os.path.join(BASE_DIR, "ax_summaries.db")

# 파일이 없어도 connect() 시점에 빈 파일이 만들어지고, 아래에서 테이블이 생성됩니다.
conn = sqlite3.connect(db_path)
c = conn.cursor()
c.execute("""
    CREATE TABLE IF NOT EXISTS summaries (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        date TEXT,
        content TEXT
    )
""")
# (선택) 데이터가 하나도 없을 때만 샘플 삽입
c.execute("SELECT COUNT(*) FROM summaries")
if c.fetchone()[0] == 0:
    c.execute(
        "INSERT INTO summaries (date, content) VALUES (?, ?)",
        ("2025-07-11", "정부는 RE100 산업단지를 통해 AI 데이터센터를 유치하는 정책을 추진 중임.")
    )
conn.commit()
conn.close()

# ————— 데이터 로딩 (캐시 적용) —————
@st.cache_data(show_spinner=False)
def load_data(path):
    conn = sqlite3.connect(path)
    df = pd.read_sql("SELECT * FROM summaries", conn)
    conn.close()
    # 날짜 컬럼 파싱
    df['date'] = pd.to_datetime(df['date'])
    return df

df = load_data(db_path)

# ————— UI 그리기 —————
st.title("🧠 SK AX 신문스크랩 흐름 대시보드")
st.markdown("SK AX팀의 AX 기사 요약 흐름을 분석하고 시각화합니다.")

# 1) 날짜별 기사 수 추이
st.subheader("📅 날짜별 AX 기사 수")
daily_count = df.groupby('date').size()
st.line_chart(daily_count)

# 2) 키워드 클라우드
st.subheader("☁️ 주요 키워드 클라우드")
all_text = " ".join(df['content'])
wordcloud = WordCloud(width=1000, height=400).generate(all_text)
fig, ax = plt.subplots()
ax.imshow(wordcloud, interpolation='bilinear')
ax.axis("off")
st.pyplot(fig)

# 3) 요약 리스트 조회
st.subheader("📰 기사 요약 리스트")
selected_date = st.date_input("날짜 선택", df['date'].max())
mask = df['date'].dt.date == selected_date
for _, row in df[mask].iterrows():
    st.markdown(f"**🗓️ {row['date'].date()}**")
    st.text_area("요약 내용", row['content'], height=200)

# 4) (디버깅) 생성된 파일 확인
st.write("앱 디렉터리 파일들:", os.listdir(BASE_DIR))
