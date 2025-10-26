# app.py — 안정 UI + Glow 100% 적용
import os
import json
import re
import pandas as pd
import streamlit as st
import psycopg

from langchain_community.utilities import SQLDatabase
from langchain_openai import ChatOpenAI
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY")
DB_HOST = os.getenv("DB_HOST") or st.secrets.get("DB_HOST")
DB_NAME = os.getenv("DB_NAME") or st.secrets.get("DB_NAME", "postgres")
DB_USER = os.getenv("DB_USER") or st.secrets.get("DB_USER", "readonly")
DB_PASS = os.getenv("DB_PASS") or st.secrets.get("DB_PASS")
DB_PORT = int(os.getenv("DB_PORT") or st.secrets.get("DB_PORT", 5432))

client = OpenAI(api_key=OPENAI_API_KEY)

SQLALCHEMY_URI = (
    f"postgresql+psycopg://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    "?sslmode=require"
)

AGENT_PREFIX = """
당신은 PostgreSQL SQL 전문가다.
오직 SELECT만 작성.
"""

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

@st.cache_resource
def get_lc_db():
    return SQLDatabase.from_uri(SQLALCHEMY_URI)

def get_sql_agent():
    from langchain_community.agent_toolkits import create_sql_agent
    return create_sql_agent(
        llm=llm,
        db=get_lc_db(),
        verbose=False,
        prefix=AGENT_PREFIX,
    )

# -------- 유틸 --------
def _extract_first_select(text: str) -> str:
    cleaned = text.strip()
    m = re.search(r"(?i)select", cleaned)
    if not m:
        return cleaned
    begin = m.start()
    sql = cleaned[begin:]
    semi = re.search(r";", sql)
    if semi:
        sql = sql[:semi.start()]
    return sql.strip()

def _validate_sql(sql: str):
    if not sql.lower().startswith("select"):
        raise ValueError("SELECT only")

# -------- CSS --------
st.markdown("""
<style>
:root { --blue:#0064FF; }

html, body, [data-testid="stAppViewContainer"] {
  background: #ECEEF1 !important;
}

* { font-family:'Pretendard',sans-serif !important; }

.header { text-align:center; margin-top:40px; }
.title { font-size:32px; font-weight:900; }
.byline { color:#6b7280; font-size:13px; }

/* ✅ 커스텀 Input Shell */
.glow-wrap {
  background:white; padding:12px 18px;
  border-radius:999px;
  border:2px solid var(--blue);
  display:flex; justify-content:center;
  box-shadow:
    0 0 20px rgba(0,100,255,.55),
    0 0 40px rgba(0,100,255,.35);
  animation:glowPulse 2s infinite ease-in-out;
}

/* 🔥Glow 강력 효과 */
@keyframes glowPulse {
  50% {
    box-shadow:
      0 0 35px rgba(0,100,255,.85),
      0 0 60px rgba(0,100,255,.45);
  }
}

/* ✅ Shadow DOM input 숨기고 투명화 */
div[data-baseweb="input"] input {
  background:transparent !important;
  border:0 !important;
  outline:none !important;
  box-shadow:none !important;
  height:32px !important;
}

button {
  background:var(--blue) !important;
  border-radius:999px !important;
  font-size:17px !important;
  font-weight:700 !important;
  height:50px !important;
}
button:hover { opacity:.9 !important; }

/* ✅ 아이콘 색상 강제 */
.svgfix path {
  fill: var(--blue) !important;
}
</style>
""", unsafe_allow_html=True)

# ---------------- Header ----------------
st.markdown("""
<div class="header">
  <div class="title">보험사 경영공시 챗봇
    <svg class="svgfix" xmlns="http://www.w3.org/2000/svg" width="36" height="36" viewBox="0 0 24 24">
      <path d="M12 8V4H8V8H12Z"/>
      <path d="M16 8V4H12V8H16Z"/>
      <path d="M12 14V12H8V14H12Z"/>
      <path d="M16 14V12H12V14H16Z"/>
      <path d="M6 18H18V16H6V18Z"/>
      <path d="M6 12H4V10H6V12Z"/>
      <path d="M20 12H18V10H20V12Z"/>
    </svg>
  </div>
  <div class="byline">made by 태훈 · 현철</div>
</div>
""", unsafe_allow_html=True)

# -------- Input + Glow Wrapper --------
with st.container():
    st.markdown('<div class="glow-wrap">', unsafe_allow_html=True)
    q = st.text_input("question", label_visibility="collapsed",
                      placeholder="예) 2023년 농협생명 K-ICS비율 알려줘")
    st.markdown('</div>', unsafe_allow_html=True)

if st.button("실행", use_container_width=True):
    if not q:
        st.warning("질문을 입력하세요.")
    else:
        try:
            agent = get_sql_agent()
            res = agent.invoke({"input": q})
            text = res.get("output") or res.get("final_answer")
            sql = _extract_first_select(text)
            _validate_sql(sql)

            df = pd.read_sql_query(sql,
                psycopg.connect(host=DB_HOST, dbname=DB_NAME,
                                user=DB_USER, password=DB_PASS,
                                port=DB_PORT, sslmode="require")
            )

            st.dataframe(df, use_container_width=True)

            if not df.empty:
                preview = df.head(20).to_csv(index=False)
                msg = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role":"user","content":f"{preview}\n3문장 요약"}]
                )
                st.success(msg.choices[0].message.content.strip())
        except Exception as e:
            st.error(f"오류: {e}")
