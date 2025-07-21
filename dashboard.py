import streamlit as st
import pandas as pd
import sqlite3
import matplotlib.pyplot as plt
from wordcloud import WordCloud

st.set_page_config(page_title="SK AX 흐름 대시보드", layout="wide")

# SQLite 연결
conn = sqlite3.connect("ax_summaries.db")
df = pd.read_sql("SELECT * FROM summaries", conn)
conn.close()

# 날짜 형식 처리
df['date'] = pd.to_datetime(df['date'])

st.title("🧠 SK AX 신문스크랩 흐름 대시보드")
st.markdown("SK AX팀의 AX 기사 요약 흐름을 분석하고 시각화합니다.")

# 날짜별 기사 수 추이
st.subheader("📅 날짜별 AX 기사 수")
daily_count = df.groupby('date').count()['content']
st.line_chart(daily_count)

# 키워드 클라우드 시각화
st.subheader("☁️ 주요 키워드 클라우드")
all_text = " ".join(df['content'].tolist())
wordcloud = WordCloud(width=1000, height=400).generate(all_text)
fig, ax = plt.subplots()
ax.imshow(wordcloud, interpolation='bilinear')
ax.axis("off")
st.pyplot(fig)

# 요약 리스트 확인
st.subheader("📰 기사 요약 리스트")
selected_date = st.date_input("날짜 선택", df['date'].max())
selected_rows = df[df['date'] == pd.to_datetime(selected_date)]

for _, row in selected_rows.iterrows():
    st.markdown(f"**🗓️ {row['date'].date()}**")
    st.text_area("요약 내용", row['content'], height=200)
