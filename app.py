# app.py — Glow 완벽 적용 + 챗봇 아이콘 재작업
import os
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
오직 SELECT만 작성하라.
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

def _extract_first_select(text):
    m = re.search(r"(?i)select", text)
    if not m:
        return text.strip()
    sql = text[m.start():]
    semi = re.search(r";", sql)
    if semi:
        sql = sql[:semi.start()]
    return sql.strip()

def _validate_sql(sql):
    if not sql.lower().startswith("select"):
        raise ValueError("SELECT only")

st.set_page_config(page_title="보험사 경영공시 챗봇", page_icon="🤖", layout="centered")

# ✅ CSS 재설계 (input 박스 자체가 Glow)
st.markdown("""
<style>
:root { --blue:#0064FF; }

html, body, [data-testid="stAppViewContainer"] {
  background: #ECEEF1 !important;
}

.header { text-align:center; margin-top:40px; }
.title { font-size:32px; font-weight:900; }

.byline { color:#6b7280; font-size:13px; margin-bottom:30px; }

.glow {
  background:white;
  border-radius:999px;
  border:2px solid var(--blue);
  padding:3px 18px!important;
  display:flex;
  align-items:center;
  justify-content:center;
  height:55px;
  box-shadow:
    0 0 20px rgba(0,100,255,.55),
    0 0 40px rgba(0,100,255,.35);
  animation:glowPulse 2s infinite ease-in-out;
}

@keyframes glowPulse {
  50% {
    box-shadow:
      0 0 35px rgba(0,100,255,.9),
      0 0 60px rgba(0,100,255,.5);
  }
}

/* ✅ 진짜 Input 박스 완전 투명화 */
input {
  background:transparent!important;
  border:none!important;
  outline:none!important;
  box-shadow:none!important;
  width:100% !important;
  height:40px!important;
  font-size:17px!important;
  font-weight:500!important;
}

/* ✅ 챗봇 아이콘 진짜 챗봇으로 (새 SVG) */
.bot-icon svg path {
  stroke:var(--blue)!important;
  stroke-width:1.8!important;
  fill:none!important;
}
</style>
""", unsafe_allow_html=True)


# ---------------- Header ----------------
st.markdown("""
<div class="header">
  <div class="title">
    보험사 경영공시 챗봇
    <span class="bot-icon">
      <svg width="35" height="35" viewBox="0 0 24 24">
        <path d="M12 2 L15 6 H21 V16 H3 V6 H9 Z"/>
        <circle cx="9" cy="11" r="1.5"/>
        <circle cx="15" cy="11" r="1.5"/>
      </svg>
    </span>
  </div>
  <div class="byline">made by 태훈 · 현철</div>
</div>
""", unsafe_allow_html=True)


# ✅ Glow Wrapper + Input 일체화
st.markdown('<div class="glow">', unsafe_allow_html=True)
q = st.text_input("질문", placeholder="예) 2023년 농협생명 K-ICS비율 알려줘", label_visibility="collapsed")
st.markdown('</div>', unsafe_allow_html=True)

st.write("")  # 입력창과 실행버튼 간 간격

if st.button("실행", use_container_width=True):
    if not q:
        st.warning("질문을 입력하세요.")
    else:
        try:
            agent = get_sql_agent()
            res = agent.invoke({"input": q})
            sql = _extract_first_select(res.get("output") or res.get("final_answer"))
            _validate_sql(sql)
            df = pd.read_sql_query(sql,
                psycopg.connect(host=DB_HOST, dbname=DB_NAME,
                                user=DB_USER, password=DB_PASS,
                                port=DB_PORT, sslmode="require")
            )

            st.dataframe(df, use_container_width=True)

            if not df.empty:
                preview = df.head(20).to_csv(index=False)
                m = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role":"user","content":f"{preview}\n3문장 요약"}]
                )
                st.success(m.choices[0].message.content.strip())
        except Exception as e:
            st.error(f"오류: {e}")
