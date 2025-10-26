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
  예: '매출/수익'→ metric='revenue', '자산'→ 'assets', '부채'→ 'liabilities', 'K-ICS/킥스'→ 'k_ics'
- 회사명/약칭/별칭 등은 사용자가 한국어로 적더라도 스스로 합리적 company_code를 추론한다. (모호하면 LIMIT 300으로 시작)
- SELECT * 대신 필요한 컬럼만 선택하고, where 절에 기간/회사/지표 필터를 상식적으로 건다.
- 첫 토큰은 반드시 SELECT, CTE/WITH/EXPLAIN 금지. 세미콜론은 최대 1개만 허용.
- 사용자가 'YYYY년 MM월'또는 '2024.12' 또는 'YY년 MM월'을 입력하면 반드시 'closing_ym = YYYYMM'으로 변환한다.
- 최근 연말로 추정하거나 자동 보정하지 않는다.
- 회사명은 "미래에셋생명,흥국화재,한화생명,한화손해,iM라이프생명,흥국생명,메리츠화재,KB생명,신한생명,DB생명,하나생명,BNP생명,푸본현대생명,ABL생명,DB손해,동양생명,농협생명,삼성화재,교보라이프플래닛생명,메트라이프생명,처브라이프생명보험,AIA생명,현대해상,교보생명,롯데손해,KDB생명,라이나생명,IBK생명,코리안리,KB손해,삼성생명,농협손보"로 DB에 저장되어있다.
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

# ----------------- 유틸 함수 -----------------
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
    banned = r"(?is)\b(insert|update|delete|drop|alter|create|grant|revoke|truncate|copy|into|explain|with)\b"
    if re.search(banned, sql):
        raise ValueError("Blocked SQL keyword detected.")

# ----------------- 페이지 설정 -----------------
st.set_page_config(page_title="보험사 경영공시 챗봇", page_icon="📊", layout="centered")

# ----------------- CSS Theme -----------------
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

html, body, [data-testid="stAppViewContainer"] { background: var(--bg) !important; }
* { font-family: 'Pretendard', sans-serif !important; }

.block-container { padding-top: 1.0rem; max-width: 860px; }

/* HEADER */
.header { padding: 39px 20px 12px; text-align:center; }
.title-row { display:flex; align-items:center; justify-content:center; gap:4px; flex-wrap:nowrap; }
.header h1 {
  margin: 0;
  font-size: clamp(24px, 6vw, 38px);
  font-weight: 900;
  color: var(--text);
  white-space: nowrap;
}
.header .icon svg {
  width: 40px;
  height: 40px;
  fill: #0064FF; /* ✅ 파란 실루엣 스타일 */
  pointer-events:none;
}
.byline { color: #6b7280; font-size:13px; }

/* Input Glow */
.input-like .stTextInput input {
  height: 56px;
  border-radius:999px;
  border:1px solid var(--blue);
  box-shadow:
    0 0 18px rgba(0,100,255,.55),
    0 0 30px rgba(0,100,255,.35);
  animation: glowPulse 2s infinite ease-in-out;
}
@keyframes glowPulse {
  50% {
    box-shadow:
      0 0 28px rgba(0,100,255,.85),
      0 0 45px rgba(0,100,255,.45);
  }
}

/* Buttons */
.stButton>button {
  height:48px; font-weight:700; font-size:16px;
  color:#fff; background:var(--blue);
  border-radius:12px; border:0;
}
.stButton>button:hover { background:var(--blue-dark); }

/* Results */
.table-container .stDataFrame { border:1px solid #e5e7eb; border-radius:8px; }
.fadein { animation:fadeIn .4s ease; }
@keyframes fadeIn { from{opacity:0;} to{opacity:1;} }
</style>
""", unsafe_allow_html=True)

# ----------------- Header -----------------
st.markdown("""
<div class="header">
  <div class="title-row">
    <h1>보험사 경영공시 챗봇</h1>
    <span class="icon">
      <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24"
          aria-hidden="true">
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

# SQL 생성 + 실행
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

# Execute
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

