# app.py — LangChain create_sql_agent 버전 (LLM 자율 매핑, 최종 SELECT만 반환)

import os
import json
import re
import pandas as pd
import streamlit as st
import psycopg
from dotenv import load_dotenv

# ▼ LangChain / OpenAI (LangChain용)
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit, create_sql_agent
from langchain_openai import ChatOpenAI
# (요약용) 기존 OpenAI SDK
from openai import OpenAI

load_dotenv()

# ----------------- 환경변수/시크릿 -----------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY")
DB_HOST = os.getenv("DB_HOST") or st.secrets.get("DB_HOST")         # e.g., abc.supabase.co
DB_NAME = os.getenv("DB_NAME") or st.secrets.get("DB_NAME", "postgres")
DB_USER = os.getenv("DB_USER") or st.secrets.get("DB_USER", "readonly")
DB_PASS = os.getenv("DB_PASS") or st.secrets.get("DB_PASS")
DB_PORT = int(os.getenv("DB_PORT") or st.secrets.get("DB_PORT", 5432))

if not OPENAI_API_KEY:
    st.stop()


# LangChain LLM (SQL 생성용)
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=OPENAI_API_KEY)
# 요약용 OpenAI SDK (스트림릿과 친화적으로 그대로 유지)
client = OpenAI(api_key=OPENAI_API_KEY)

# ----------------- LangChain: SQL 에이전트 구성 -----------------
# Supabase(Postgres) → SQLAlchemy URI
SQLALCHEMY_URI = f"postgresql+psycopg2://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}/"

# LangChain이 스키마/테이블 정보를 읽을 DB 핸들러
lc_db = SQLDatabase.from_uri(SQLALCHEMY_URI)

# Toolkit & Agent
toolkit = SQLDatabaseToolkit(db=lc_db, llm=llm)
agent = create_sql_agent(llm=llm, toolkit=toolkit, verbose=True)

# ----------------- 페이지/테마 -----------------
st.set_page_config(page_title="보험사 경영공시 데이터 챗봇", page_icon="📊", layout="wide")

# Pretendard + 글로벌 스타일
st.markdown("""
<link rel="preconnect" href="https://cdn.jsdelivr.net" />
<link rel="stylesheet" as="style" crossorigin
      href="https://cdn.jsdelivr.net/gh/orioncactus/pretendard@v1.3.9/dist/web/static/pretendard.min.css" />
<style>
:root { --blue:#0064FF; --blue-dark:#0050CC; --bg:#F0F1F3; --text:#0f172a; --muted:#64748b; --card:#ffffff; --ring:#93c5fd; }
html, body, [data-testid="stAppViewContainer"] { background: var(--bg) !important; }
* { font-family: 'Pretendard', -apple-system, BlinkMacSystemFont, system-ui, Roboto, 'Helvetica Neue',
     'Segoe UI', 'Apple SD Gothic Neo', 'Noto Sans KR', 'Malgun Gothic', sans-serif !important; }
.container-card { background: var(--card); border-radius: 16px; box-shadow: 0 2px 12px rgba(2, 6, 23, 0.06); border: 1px solid #eef2f7; }
.header { padding: 32px 32px 16px 32px; border-bottom: 1px solid #eef2f7; text-align: center; }
.header h1 { margin: 0; padding: 0; font-size: 28px; font-weight: 800; letter-spacing: -0.02em; color: var(--text); }
.header .byline { color: #6b7280; font-size: 13px; margin-top: 6px; opacity: .85; }
.section { padding: 24px 32px 28px 32px; }
.hint { text-align:center; color:#475569; font-size: 16px; margin-bottom: 14px; }
.input-like label { display:none!important; }
.input-like .stTextInput>div>div>input { height: 56px; font-size: 18px; padding: 0 18px; background:#f3f4f6; border:1px solid #e5e7eb; border-radius:12px; }
.input-like .stTextInput>div>div>input:focus { outline: none; border-color: var(--ring); box-shadow: 0 0 0 3px rgba(147,197,253,.45); }
.stButton>button { width:100%; height:54px; font-weight:700; font-size:18px; color:#fff; background: var(--blue);
  border-radius:12px; border:0; box-shadow: 0 2px 0 rgba(0,0,0,.03); }
.stButton>button:hover { background: var(--blue-dark); }
.stButton>button:disabled { background:#d1d5db !important; color:#fff !important; }
.kpi { display:flex; gap:12px; align-items:center; justify-content:center; margin-top:8px; color:#6b7280; font-size:14px; }
.badge { background:#eef2ff; color:#3730a3; padding:6px 10px; border-radius:999px; font-weight:600; font-size:12px; }
.card-subtitle { color:#334155; font-size:18px; margin: 0 0 10px; text-align:center; }
.table-container .stDataFrame { border-radius:12px; overflow:hidden; border: 1px solid #e5e7eb; }
hr.sep { border:none; border-top:1px solid #eef2f7; margin: 20px 0; }
.small-note { color:#64748b; font-size:13px; margin-top:4px;}
.footer-note { color:#64748b; font-size:12px; text-align:center; margin-top:16px; }
.fadein { animation: fadeIn .5s ease; } @keyframes fadeIn { from{opacity:0; transform: translateY(6px)} to{opacity:1; transform:none} }
pre, code { font-size: 13px !important; }
</style>
""", unsafe_allow_html=True)

# ----------------- 헤더 -----------------
st.markdown('<div class="container-card fadein">', unsafe_allow_html=True)
st.markdown(f"""
<div class="header">
  <div style="display:flex; gap:10px; align-items:center; justify-content:center;">
    <h1>보험사 경영공시 데이터 <span style="color:var(--text)">챗봇</span></h1>
    <svg xmlns="http://www.w3.org/2000/svg" width="28" height="28" viewBox="0 0 24 24"
         fill="none" stroke="#0064FF" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
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
  </div>
  <div class="byline">made by 태훈 · 정형 데이터(SQL) 전용</div>
  <div class="kpi">
    <span class="badge">DB 연결</span>
    <span>Host: <b>{DB_HOST or "-"}</b></span>
    <span>·</span>
    <span>User: <b>{DB_USER or "-"}</b></span>
  </div>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="section">', unsafe_allow_html=True)
st.markdown('<p class="hint">현재 보유데이터는 2022~2024년 가정변경효과 · K-ICS 비율</p>', unsafe_allow_html=True)

# ----------------- 사이드바 (상태 영역) -----------------
with st.sidebar:
    st.markdown("### 연결 상태")
    st.write(f"DB Host: {DB_HOST}")
    st.write(f"DB User: {DB_USER}")
    st.caption("좌측 버튼 흐름대로 진행하세요. (① SQL 생성 → ② 실행 → ③ 차트/요약)")

# ----------------- LLM 규칙: 저장키 기반 자율 매핑 -----------------
BASE_SQL_RULES = r"""
You are an agent that must return ONLY a single PostgreSQL SELECT query for the table:
  company_financials(company_code text, date date, metric text, value numeric)

Final output: only the SQL statement. No markdown/backticks/explanations.

WHAT YOU MAY DO BEFORE THE FINAL SQL (OPTIONAL, VIA TOOLS):
- You may inspect schema using available tools (e.g., get_table_info).
- You may run small discovery queries strictly for introspection, e.g.:
    SELECT DISTINCT metric FROM company_financials LIMIT 200;
    SELECT DISTINCT company_code FROM company_financials LIMIT 200;
  Use them ONLY to learn the stored keys actually present. Do not return these queries as final output.

HARD CONSTRAINTS FOR THE FINAL SQL:
- SELECT only (never INSERT/UPDATE/DELETE/ALTER/DROP/CREATE/GRANT/REVOKE/TRUNCATE).
- Prefer explicit columns over SELECT * (e.g., date, value, metric, company_code).
- If time series, include ORDER BY date (or ORDER BY period when grouped).
- If the user mentions “월별/분기별/연도별”, aggregate using DATE_TRUNC('month'|'quarter'|'year', date) AS period
  and use a sensible aggregation like AVG(value) or SUM(value). Always ORDER BY period ASC.
- If the user gives a year (e.g., "2023년"), interpret as:
    date >= 'YYYY-01-01' AND date < 'YYYY+1-01-01'.

MAPPING POLICY (NO HARDCODED SYNONYMS):
- Infer the correct metric/company_code by comparing the user's Korean terms to the ACTUAL stored keys you discovered.
- Resolve ambiguous/typoed terms (e.g., “킥스”, “K-ICS”, “건전성비율”, 한국어 회사명 등) by picking the closest matching stored keys (semantic/fuzzy),
  not fixed dictionaries.
- If multiple keys are plausible, use a conservative IN (...) filter with the best candidates.
- Keep any discovery step minimal and bounded with LIMIT.

SAFETY:
- The final answer must be a single SELECT that likely returns the intended result, using the keys that actually exist.
"""

def generate_sql_with_agent(user_question: str) -> str:
    prompt = (
        f"{BASE_SQL_RULES}\n\n"
        "USER QUESTION (Korean):\n"
        f"{user_question}\n\n"
        "Return only the final SELECT statement:"
    )
    try:
        st.markdown("LangChain 프롬프트 (SQL 생성)")
        st.code(prompt)
    except Exception:
        pass

    try:
        # create_sql_agent 는 툴 사용 가능한 ReAct agent.
        # 필요 시 DISTINCT 탐색을 잠깐 수행하고, 최종에는 단일 SELECT만 내놓도록 유도.
        if hasattr(agent, "invoke"):
            res = agent.invoke({"input": prompt})
            sql = res["output"] if isinstance(res, dict) and "output" in res else str(res)
        else:
            sql = agent.run(prompt)
    except Exception as e:
        raise RuntimeError(f"SQL 에이전트 오류: {e}")

    sql = (sql or "").strip()

    try:
        st.markdown("LangChain 응답 (SQL 생성)")
        st.code(sql, language="sql")
    except Exception:
        pass

    # 안전필터
    if not re.match(r"(?is)^\s*select\s", sql):
        raise ValueError("Only SELECT queries are allowed.")
    banned = r"(?is)\b(insert|update|delete|drop|alter|create|grant|revoke|truncate)\b"
    if re.search(banned, sql):
        raise ValueError("Blocked SQL keyword detected.")
    return sql

# ----------------- DB 실행 -----------------
def run_sql(sql: str) -> pd.DataFrame:
    with psycopg.connect(host=DB_HOST, dbname=DB_NAME, user=DB_USER, password=DB_PASS, port=DB_PORT) as conn:
        return pd.read_sql_query(sql, conn)

# ----------------- 요약 -----------------
def summarize_answer(q: str, df: pd.DataFrame) -> str:
    preview_csv = df.head(20).to_csv(index=False)
    prompt = f"""질문: {q}
아래 CSV 일부를 참고해서 3문장 이내로 한국어 요약을 써줘. 단위와 기간을 분명히 써.
CSV 미리보기(최대 20행):
{preview_csv}
"""
    try:
        st.markdown("OpenAI 프롬프트 (요약)")
        st.code(prompt, language="markdown")
    except Exception:
        pass
    r = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role":"user","content": prompt}],
        temperature=0.2
    )
    summary_text = r.choices[0].message.content.strip()
    try:
        st.markdown("OpenAI 응답 (요약)")
        st.code(summary_text)
    except Exception:
        pass
    return summary_text

# ----------------- 입력창 -----------------
st.markdown('<div class="input-like">', unsafe_allow_html=True)
q = st.text_input(
    label="질문",
    placeholder="예) 2023년 농협생명 킥스 월별 추이 보여줘",
    label_visibility="collapsed",
    key="q_input"
)
st.markdown('</div>', unsafe_allow_html=True)

# ----------------- 버튼 & 흐름 -----------------
c1, c2 = st.columns([1,1])
with c1:
    st.markdown('<p class="card-subtitle">① SQL 생성</p>', unsafe_allow_html=True)
    make_sql = st.button("SQL 만들기", use_container_width=True)
with c2:
    st.markdown('<p class="card-subtitle">② SQL 실행</p>', unsafe_allow_html=True)
    run_btn = st.button("실행", use_container_width=True)

# SQL 만들기
if make_sql:
    if not q:
        st.warning("질문을 입력하세요.")
    else:
        with st.spinner("LangChain 에이전트가 SQL을 생성 중..."):
            try:
                sql = generate_sql_with_agent(q)
                st.code(sql, language="sql")
                st.session_state["sql"] = sql
            except Exception as e:
                st.error(f"SQL 생성 오류: {e}")

st.markdown('<hr class="sep"/>', unsafe_allow_html=True)

# 실행
if run_btn:
    sql = st.session_state.get("sql")
    if not sql:
        st.warning("먼저 'SQL 만들기'를 클릭하세요.")
    else:
        with st.spinner("DB에서 데이터 조회 중..."):
            try:
                df = run_sql(sql)
                if df.empty:
                    st.info("결과가 없습니다.")
                else:
                    st.markdown('<div class="table-container">', unsafe_allow_html=True)
                    st.dataframe(df, use_container_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                    st.session_state["df"] = df
            except Exception as e:
                st.error(f"DB 실행 오류: {e}")

# ----------------- 차트 & 요약 -----------------
st.markdown('<div class="container-card section fadein">', unsafe_allow_html=True)
st.markdown('<p class="card-subtitle">③ 차트 & 요약</p>', unsafe_allow_html=True)

df_prev = st.session_state.get("df")
if df_prev is not None and not df_prev.empty:
    date_col = None
    for c in df_prev.columns:
        if str(c).lower() == "date":
            date_col = c
            break
    if date_col:
        try:
            df_plot = df_prev.copy()
            df_plot[date_col] = pd.to_datetime(df_plot[date_col], errors="coerce")
            df_plot = df_plot.dropna(subset=[date_col])
            y_col = None
            for cand in ["value", "amount", "val"]:
                if cand in df_plot.columns:
                    y_col = cand
                    break
            if y_col:
                st.line_chart(df_plot.set_index(date_col)[y_col])
        except Exception:
            pass

    col_a, col_b = st.columns([1,1])
    with col_a:
        gen_sum = st.button("요약 생성", use_container_width=True)
    with col_b:
        st.caption("차트 영역은 time-series일 때 자동 표시됩니다.")

    if gen_sum:
        with st.spinner("요약 생성 중..."):
            try:
                summary = summarize_answer(q, df_prev)
                st.success(summary)
            except Exception as e:
                st.error(f"요약 오류: {e}")
else:
    st.caption("실행 결과가 표시되면 차트와 요약을 볼 수 있습니다.")

st.markdown('</div>', unsafe_allow_html=True)  # container-card
st.markdown('</div>', unsafe_allow_html=True)  # 상단 container-card 종료
st.markdown('<p class="footer-note">UI 동일 · 수동 매핑 제거 · 저장키 기반 자율 매핑(‘킥스’ 포함) · 최종 SELECT만 반환.</p>', unsafe_allow_html=True)
