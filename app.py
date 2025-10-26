# app.py — 보험사 경영공시 챗봇 (기능 원복 + CSS 안정화)
import os
import json
import re
import pandas as pd
import streamlit as st
import psycopg

# ====== LangChain / OpenAI LLM ======
from langchain_community.utilities import SQLDatabase

try:
    from langchain_community.agent_toolkits import create_sql_agent
except ImportError:
    try:
        from langchain_community.agent_toolkits.sql.base import create_sql_agent
    except ImportError:
        from langchain.agents.agent_toolkits import create_sql_agent

from langchain_openai import ChatOpenAI
# ====================================

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# ----------------- 환경변수/시크릿 -----------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY")
DB_HOST = os.getenv("DB_HOST") or st.secrets.get("DB_HOST")
DB_NAME = os.getenv("DB_NAME") or st.secrets.get("DB_NAME", "postgres")
DB_USER = os.getenv("DB_USER") or st.secrets.get("DB_USER", "readonly")
DB_PASS = os.getenv("DB_PASS") or st.secrets.get("DB_PASS")
DB_PORT = int(os.getenv("DB_PORT") or st.secrets.get("DB_PORT", 5432))

if not OPENAI_API_KEY:
    st.error("OPENAI_API_KEY 설정이 되어 있지 않습니다.")
    st.stop()

client = OpenAI(api_key=OPENAI_API_KEY)

# ================== LangChain Agent ==================
SQLALCHEMY_URI = (
    f"postgresql+psycopg://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    "?sslmode=require"
)

AGENT_PREFIX = """
당신은 PostgreSQL SQL 전문가다. 다음 규칙을 반드시 지켜라.
- 오직 'SELECT'만 작성한다.
- 결과는 SQL만 내보낸다.
- 테이블: kics_solvency_data_flexible
"""

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

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

# ============== ✅ 유틸 함수 복원 =====================
def _strip_code_fences(text: str) -> str:
    t = text.strip()
    t = re.sub(r"^```[a-zA-Z]*\s*", "", t)
    t = re.sub(r"\s*```$", "", t)
    return t.strip()

def _remove_sql_comments(sql: str) -> str:
    sql = re.sub(r"/\*.*?\*/", "", sql, flags=re.S)
    sql = re.sub(r"^\s*--.*?$", "", sql, flags=re.M)
    return sql.strip()

def _extract_first_select(text: str) -> str:
    cleaned = _remove_sql_comments(_strip_code_fences(text))
    m = re.search(r"(?is)\bselect\b", cleaned)
    if not m:
        return cleaned.strip()
    start = m.start()
    tail = cleaned[start:]
    semi = re.search(r";", tail)
    return (tail[:semi.start()] if semi else tail).strip()

def _validate_sql_is_select(sql: str):
    if sql.count(";") > 1:
        raise ValueError("Multiple statements not allowed.")
    if not re.match(r"(?is)^\s*select\b", sql):
        raise ValueError("Only SELECT allowed.")

# ======================================================

st.set_page_config(page_title="보험사 경영공시 챗봇", page_icon="📊", layout="centered")

# ✅ CSS 안정화 + Glow 적용
st.markdown("""
<style>
:root {
  --blue:#0064FF;
  --bg:#F0F1F3;
}

html, body, [data-testid="stAppViewContainer"] {
  background: var(--bg) !important;
}

* { font-family: 'Pretendard', sans-serif !important; }

.header { padding: 32px 10px 12px; text-align:center; }
.title-row { display:flex; align-items:center; justify-content:center; gap:6px; }
.header h1 { font-size: clamp(24px, 6vw, 38px); font-weight: 900; }

.header .icon svg {
  width: 38px;
  height: 38px;
  fill: var(--blue) !important;
}

/* ✅ Glow를 Shadow DOM 밖 wrapper에 적용 */
.input-like div[data-testid="stTextInput"] {
  background:white !important;
  border-radius:999px !important;
  border:1px solid var(--blue) !important;
  padding:6px 12px !important;
  box-shadow:0 0 20px rgba(0,100,255,.55),
             0 0 40px rgba(0,100,255,.35) !important;
  animation:glowPulse 2s infinite ease-in-out !important;
}

@keyframes glowPulse {
  50% {
    box-shadow:0 0 30px rgba(0,100,255,.85),
               0 0 50px rgba(0,100,255,.45) !important;
  }
}

input {
  background:transparent !important;
  border:none !important;
  box-shadow:none !important;
}

</style>
""", unsafe_allow_html=True)

# ======================================================
# Header UI
# ======================================================
st.markdown("""
<div class="header">
  <div class="title-row">
    <h1>보험사 경영공시 챗봇</h1>
    <span class="icon">
      <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24">
        <path d="M12 8V4H8V8H12Z" />
        <path d="M16 8V4H12V8H16Z" />
        <path d="M12 14V12H8V14H12Z" />
        <path d="M16 14V12H12V14H16Z" />
        <path d="M6 18H18V16H6V18Z" />
        <path d="M6 12H4V10H6V12Z" />
        <path d="M20 12H18V10H20V12Z" />
        <path d="M6 8H4V6H6V8Z" />
        <path d="M20 8H18V6H20V8Z" />
        <path d="M10 22H14V20H10V22Z" />
        <path d="M4 4H2V2H4V4Z" />
        <path d="M22 4H20V2H22V4Z" />
      </svg>
    </span>
  </div>
  <div style="color:#6b7280;font-size:13px;">made by 태훈 · 현철</div>
</div>
""", unsafe_allow_html=True)

result_area = st.container()

# ============== INPUT ==============
st.markdown('<div class="input-like">', unsafe_allow_html=True)
q = st.text_input("질문", placeholder="예) 2023년 NH농협생명 자산 증가율 알려줘", label_visibility="collapsed")
st.markdown('</div>', unsafe_allow_html=True)

if st.button("실행", use_container_width=True):
    if not q:
        with result_area:
            st.warning("질문을 입력하세요.")
    else:
        try:
            agent = get_sql_agent()
            result = agent.invoke({"input": q})
            text = result.get("output") or result.get("final_answer")
            sql = _extract_first_select(text)
            _validate_sql_is_select(sql)

            df = pd.read_sql_query(sql,
                psycopg.connect(
                    host=DB_HOST, dbname=DB_NAME,
                    user=DB_USER, password=DB_PASS,
                    port=DB_PORT, sslmode="require"
                )
            )

            with result_area:
                st.write("✅ 실행 결과")
                st.dataframe(df, use_container_width=True)

                if not df.empty:
                    preview = df.head(20).to_csv(index=False)
                    prompt = f"한국어로 3문장 요약:\n{preview}"
                    r = client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[{"role":"user","content":prompt}],
                        temperature=0
                    )
                    st.success(r.choices[0].message.content.strip())

        except Exception as e:
            with result_area:
                st.error(f"오류 발생: {e}")
