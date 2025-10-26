# ==========================================
# app.py — 보험사 경영공시 챗봇 (최종 완성본)
# - "데이터" 제거
# - 타이틀 위 15pt 여백
# - 챗봇 아이콘 40% 확대 & 글자 붙임
# - 입력창 네온 블루 글로우 + 반짝 애니메이션
# - 실행 버튼 50%폭 중앙
# - 실행 결과가 타이틀과 입력창 사이에 표시
# ==========================================

import os
import json
import re
import pandas as pd
import streamlit as st
import psycopg

from langchain_community.utilities import SQLDatabase
try:
    from langchain_community.agent_toolkits import create_sql_agent
except ImportError:
    try:
        from langchain_community.agent_toolkits.sql.base import create_sql_agent
    except ImportError:
        from langchain.agents.agent_toolkits import create_sql_agent
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

if not OPENAI_API_KEY:
    st.error("OPENAI_API_KEY 설정이 필요합니다")
    st.stop()

client = OpenAI(api_key=OPENAI_API_KEY)

SQLALCHEMY_URI = (
    f"postgresql+psycopg://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}?"
    "sslmode=require"
)

AGENT_PREFIX = """
당신은 PostgreSQL SQL 전문가다.
...
(👉 기존에 사용한 AGENT_PREFIX 내용 그대로 유지)
""".strip()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=OPENAI_API_KEY)

@st.cache_resource(show_spinner=False)
def get_lc_db():
    return SQLDatabase.from_uri(SQLALCHEMY_URI)

def get_sql_agent():
    return create_sql_agent(
        llm=llm,
        db=get_lc_db(),
        agent_type="openai-tools",
        verbose=False,
        prefix=AGENT_PREFIX,
    )

def _strip_code_fences(text: str) -> str:
    t=text.strip()
    t=re.sub(r"^```[a-zA-Z]*\s*","",t)
    t=re.sub(r"\s*```$","",t)
    return t.strip()

def _remove_sql_comments(sql:str)->str:
    sql=re.sub(r"/\*.*?\*/","",sql,flags=re.S)
    sql=re.sub(r"^\s*--.*?$","",sql,flags=re.M)
    return sql.strip()

def _extract_first_select(text:str)->str:
    cleaned=_remove_sql_comments(_strip_code_fences(text))
    m=re.search(r"(?is)\bselect\b",cleaned)
    if not m: return cleaned.strip()
    tail=cleaned[m.start():]
    semi=re.search(r";",tail)
    return (tail[:semi.start()] if semi else tail).strip()

def _validate_sql_is_select(sql:str):
    if not re.match(r"(?is)^\s*select\b",sql): raise ValueError("SELECT만 허용")
    if sql.count(";")>1: raise ValueError("다중 문장 금지")

st.set_page_config(page_title="보험사 경영공시 챗봇", page_icon="🤖", layout="centered")

# ---------------- CSS 적용 ----------------
st.markdown("""
<style>
:root { --blue:#0064FF; }

.header{
 padding:39px 20px 12px 20px;
 border-bottom:1px solid #eef2f7;
 text-align:center;
}

/* 아이콘 확대 + 글자에 붙임 */
.title-row{
 display:flex; align-items:center; justify-content:center;
 gap:4px !important;
 flex-wrap:nowrap;
}
.header svg{
 width:36px !important; height:36px !important;
}

/* 입력창 네온 글로우 */
.input-like .stTextInput > div > div > input{
 height:56px; font-size:17px;
 padding:0 22px; border-radius:9999px;
 border:2px solid #0080ff;
 background:#fff;
 animation:flash 1.2s infinite ease-in-out alternate;
}
@keyframes flash{
 from{ box-shadow:
     0 0 8px #0080ff,
     0 0 20px rgba(0,128,255,.4),
     0 0 40px rgba(0,128,255,.25);
 }
 to{ box-shadow:
     0 0 15px #0080ff,
     0 0 45px rgba(0,128,255,.65),
     0 0 75px rgba(0,128,255,.45);
 }
}

/* 실행 버튼 50% 폭 → columns 로 중앙 위치 */
</style>
""", unsafe_allow_html=True)

# ---------------- Header ----------------
st.markdown("""
<div class="header container-card">
  <div class="title-row">
    <h1>보험사 경영공시 챗봇</h1>
    <svg fill="none" stroke="#0064FF" stroke-width="2"
         stroke-linecap="round" stroke-linejoin="round"
         viewBox="0 0 24 24">
      <path d="M12 8V4H8V8H12Z" />
      <!-- 기존 SVG 동일 -->
    </svg>
  </div>
  <div class="byline">made by 태훈 · 현철</div>
</div>
""", unsafe_allow_html=True)

# ✅ 결과 표시 영역 (타이틀 아래)
result_area = st.container()

# ✅ 입력 UI
st.markdown('<div class="section input-like">', unsafe_allow_html=True)
q = st.text_input(
    "질문",
    placeholder="예) 2023년 농협생명 K-ICS 지급여력비율 추이",
    label_visibility="collapsed",
    key="q_input"
)
st.markdown('</div>', unsafe_allow_html=True)

# ✅ 실행 버튼 (중앙 50%)
c1,c2,c3 = st.columns([1,2,1])
with c2:
    run_btn = st.button("실행", use_container_width=True)

def run_sql(sql:str)->pd.DataFrame:
    with psycopg.connect(
        host=DB_HOST,dbname=DB_NAME,user=DB_USER,password=DB_PASS,
        port=DB_PORT,sslmode="require",
    ) as conn:
        return pd.read_sql_query(sql, conn)

def summarize(q,df):
    csv=df.head(20).to_csv(index=False)
    r=client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role":"user","content":f"질문:{q}\n{csv}\n3문장으로 요약"}]
    )
    return r.choices[0].message.content.strip()

if run_btn:
    if not q:
        with result_area: st.warning("질문 입력하세요")
    else:
        try:
            agent=get_sql_agent()
            res=agent.invoke({"input":q})
            text=res.get("output") if isinstance(res,dict) else str(res)
            sql=_extract_first_select(text)
            _validate_sql_is_select(sql)
            df=run_sql(sql)
            st.session_state["df"]=df
            with result_area:
                st.markdown("### ✅ 실행 결과")
                st.dataframe(df, use_container_width=True)
            if not df.empty:
                summary=summarize(q,df)
                with result_area: st.success(summary)
        except Exception as e:
            with result_area: st.error(str(e))

df_prev=st.session_state.get("df")
if df_prev is not None and not df_prev.empty:
    if st.button("요약 다시 생성"):
        with result_area: st.success(summarize(q,df_prev))
