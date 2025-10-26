# app.py — 보험사 경영공시 챗봇 (SQL 생성+실행 One-Click, 결과 상단 슬롯, 파란 글로우 입력창, 모바일 타이틀 1줄 고정)
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
- 오직 'SELECT'만 작성한다. (INSERT/UPDATE/DELETE/ALTER/DROP/CREATE/GRANT/REVOKE/TRUNCATE 금지)
- 결과는 SQL만 내보낸다. 백틱/설명/자연어/코드블록/주석 없이 SQL 한 문장만 출력한다.
- 대상 테이블: kics_solvency_data_flexible
- 시계열을 조회할 때는 항상 ORDER BY date를 포함한다.
- 한국어 질의의 의미를 스스로 판단해 컬럼/값을 매핑한다.
- SELECT * 대신 필요한 컬럼만 선택한다.
"""

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

# ======================================================
# ---------------------- CSS FIX -----------------------
# ======================================================
st.markdown("""
<style>
:root {
  --blue:#0064FF;
  --blue-dark:#0050CC;
  --bg:#F0F1F3;
  --text:#0f172a;
  --muted:#64748b;
  --card:#ffffff;
}

html, body, [data-testid="stAppViewContainer"] {
  background: var(--bg) !important;
}

* { font-family: 'Pretendard', sans-serif !important; }

.block-container { padding-top: 1.0rem !important; max-width: 860px !important; }

/* HEADER */
.header { padding: 39px 20px 12px !important; text-align:center !important; }
.title-row { display:flex !important; align-items:center !important; justify-content:center !important; gap:4px !important; flex-wrap:nowrap !important; }
.header h1 {
  margin: 0 !important;
  font-size: clamp(24px, 6vw, 38px) !important;
  font-weight: 900 !important;
  color: var(--text) !important;
  white-space: nowrap !important;
}
.header .icon svg {
  width: 40px !important;
  height: 40px !important;
  fill: var(--blue) !important;
  pointer-events:none !important;
  margin:0 !important;
  padding:0 !important;
}
.byline { color: #6b7280 !important; font-size:13px !important; }

/* ✅ Input Glow — 안정적 셀렉터 + 우선순위 극대화 */
.input-like [data-testid="stTextInput"] input {
  height: 56px !important;
  border-radius: 999px !important;
  border: 1px solid var(--blue) !important;
  box-shadow:
    0 0 18px rgba(0,100,255,.55),
    0 0 30px rgba(0,100,255,.35) !important;
  animation: glowPulse 2s infinite ease-in-out !important;
  background-color: #ffffff !important;
}
@keyframes glowPulse {
  50% {
    box-shadow:
      0 0 28px rgba(0,100,255,.85),
      0 0 45px rgba(0,100,255,.45) !important;
  }
}

/* Buttons */
.stButton>button {
  height:48px !important; font-weight:700 !important; font-size:16px !important;
  color:#fff !important; background:var(--blue) !important;
  border-radius:12px !important; border:0 !important;
}
.stButton>button:hover { background:var(--blue-dark) !important; }

/* Results */
.table-container .stDataFrame { border:1px solid #e5e7eb !important; border-radius:8px !important; }
.fadein { animation:fadeIn .4s ease !important; }
@keyframes fadeIn { from{opacity:0;} to{opacity:1;} }
</style>
""", unsafe_allow_html=True)

# ----------------- Header -----------------
st.markdown("""
<div class="header">
  <div class="title-row">
    <h1>보험사 경영공시 챗봇</h1>
    <span class="icon">
      ✅ SVG 그대로 유지…
    </span>
  </div>
  <div class="byline">made by 태훈 · 현철</div>
</div>
""", unsafe_allow_html=True)

result_area = st.container()

# ===================== INPUT =====================
st.markdown('<div class="input-like">', unsafe_allow_html=True)
q = st.text_input(
    label="질문",
    placeholder="예) 2023년 NH농협생명 매출 월별 추이 보여줘",
    label_visibility="collapsed"
)
st.markdown('</div>', unsafe_allow_html=True)

c1, c2, c3 = st.columns([1,2,1])
with c2:
    go_btn = st.button("실행", use_container_width=True)

def generate_sql(user_q):
    agent = get_sql_agent()
    result = agent.invoke({"input": user_q})
    text = result.get("output") or result.get("final_answer")
    sql = _extract_first_select(text)
    _validate_sql_is_select(sql)
    return sql

def run_sql(sql):
    with psycopg.connect(
        host=DB_HOST,
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASS,
        port=DB_PORT,
        sslmode="require"
    ) as conn:
        return pd.read_sql_query(sql, conn)

def summarize_answer(q, df):
    preview = df.head(20).to_csv(index=False)
    prompt = f"""질문: {q}
아래 CSV 일부 참고하여 한국어로 3문장 이내 요약:
{preview}"""
    r = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role":"user","content":prompt}],
        temperature=0.2
    )
    return r.choices[0].message.content.strip()

if go_btn:
    if not q:
        with result_area:
            st.warning("질문을 입력하세요.")
    else:
        try:
            sql = generate_sql(q)
            df = run_sql(sql)
            st.session_state["df"] = df
            with result_area:
                st.markdown("#### ✅ 실행 결과")
                if df.empty:
                    st.info("결과가 없습니다.")
                else:
                    st.markdown('<div class="table-container">', unsafe_allow_html=True)
                    st.dataframe(df, use_container_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)

            if not df.empty:
                with result_area:
                    summary = summarize_answer(q, df)
                    st.success(summary)
        except Exception as e:
            with result_area:
                st.error(f"오류 발생: {e}")
