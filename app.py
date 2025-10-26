# app.py — 완성형 (Glow box = real input, 아래 input 완전 제거)
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

def get_sql_agent():
    from langchain_community.agent_toolkits import create_sql_agent
    db = SQLDatabase.from_uri(f"postgresql+psycopg://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}?sslmode=require")
    return create_sql_agent(llm=ChatOpenAI(model="gpt-4o-mini", temperature=0), db=db, verbose=False,
                            prefix="오직 SELECT!")

def extract_sql(output):
    m = re.search(r"(?i)select", output)
    sql = output[m.start():] if m else output
    s = re.search(r";", sql)
    return sql[:s.start()] if s else sql

st.set_page_config(page_title="보험사 경영공시 챗봇", page_icon="🤖", layout="centered")

# ✅ CSS: ONLY glow input box shown
st.markdown("""
<style>
:root { --blue:#0064FF; }

html, body, [data-testid="stAppViewContainer"] { background: #ECEEF1 !important; }
* { font-family:'Pretendard',sans-serif !important; }

.header { text-align:center; margin-top:40px; }
.title { font-size:32px; font-weight:900; }

.byline { color:#6b7280; font-size:13px; margin-bottom:25px; }

.glow-input {
  width:480px;
  margin:auto;
  background:white;
  border:2px solid var(--blue);
  border-radius:999px;
  padding:10px 25px;
  text-align:center;
  font-size:18px;
  box-shadow:
    0 0 25px rgba(0,100,255,.55),
    0 0 50px rgba(0,100,255,.35);
  animation:glowPulse 2s infinite ease-in-out;
}

@keyframes glowPulse {
  50% {
    box-shadow:
      0 0 40px rgba(0,100,255,.9),
      0 0 70px rgba(0,100,255,.5);
  }
}

.glow-input:focus {
  outline:none !important;
}

.bot-icon svg path {
  stroke: var(--blue)!important;
  stroke-width:1.8!important;
  fill:none!important;
}
</style>
""", unsafe_allow_html=True)

# ✅ Header
st.markdown("""
<div class="header">
  <div class="title">
    보험사 경영공시 챗봇
    <span class="bot-icon">
      <svg width="35" height="35" viewBox="0 0 24 24">
        <path d="M12 2 L16 7 H21 V17 H3 V7 H8 Z"/>
        <circle cx="9" cy="11" r="1.6"/>
        <circle cx="15" cy="11" r="1.6"/>
      </svg>
    </span>
  </div>
  <div class="byline">made by 태훈 · 현철</div>
</div>
""", unsafe_allow_html=True)

# ✅ ✅ Custom Input (Real input inside Glow div)
if "user_q" not in st.session_state:
    st.session_state["user_q"] = ""

st.write("""
<input id="glowinput" class="glow-input" 
placeholder="예) 2023년 농협생명 K-ICS비율 알려줘"
onchange="window.parent.postMessage({type:'setInput', value:this.value}, '*')">
""", unsafe_allow_html=True)

# ✅ Sync JS → Streamlit state update
st.components.v1.html("""
<script>
window.addEventListener('message', (e) => {
  if (e.data.type === 'setInput') {
    const input = e.data.value;
    window.parent.postMessage({type: 'streamlit:setComponentValue', value: input}, '*');
  }
});
</script>
""", height=0)

q = st.session_state.get("user_q", "")

st.write("")

# ✅ 실행 버튼 그대로 유지
if st.button("실행", use_container_width=True):
    try:
        agent = get_sql_agent()
        res = agent.invoke({"input": q})
        sql = extract_sql(res.get("output") or res.get("final_answer"))
        df = pd.read_sql_query(sql,
            psycopg.connect(host=DB_HOST, dbname=DB_NAME,
                            user=DB_USER, password=DB_PASS,
                            port=DB_PORT, sslmode="require")
        )
        st.dataframe(df, use_container_width=True)
    except Exception as e:
        st.error(f"오류: {e}")
