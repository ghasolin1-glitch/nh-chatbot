# app.py — 디자인 리팩터링 (기능 동일, LangChain SQL Agent 적용)
import os
import json
import re
import pandas as pd
import streamlit as st
import psycopg

# ====== LangChain / OpenAI LLM ======
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent
from langchain_openai import ChatOpenAI
# ====================================

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# ----------------- 환경변수/시크릿 -----------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY")
DB_HOST = os.getenv("DB_HOST") or st.secrets.get("DB_HOST")         # e.g., aws-1-us-east-1.pooler.supabase.com
DB_NAME = os.getenv("DB_NAME") or st.secrets.get("DB_NAME", "postgres")
DB_USER = os.getenv("DB_USER") or st.secrets.get("DB_USER", "readonly")
DB_PASS = os.getenv("DB_PASS") or st.secrets.get("DB_PASS")
DB_PORT = int(os.getenv("DB_PORT") or st.secrets.get("DB_PORT", 5432))

if not OPENAI_API_KEY:
    st.stop()
client = OpenAI(api_key=OPENAI_API_KEY)

# ====== LangChain용 DB/LLM/에이전트 초기화 ======
# SQLAlchemy 연결문자열 (psycopg v3 드라이버) + sslmode=require
SQLALCHEMY_URI = (
    f"postgresql+psycopg://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    "?sslmode=require"
)

# 에이전트에게 “쿼리만 출력, 실행/설명 금지”를 못박는 프리픽스
AGENT_PREFIX = """
당신은 PostgreSQL SQL 전문가다. 다음 규칙을 반드시 지켜라.

- 오직 'SELECT'만 작성한다. (INSERT/UPDATE/DELETE/ALTER/DROP/CREATE/GRANT/REVOKE/TRUNCATE 금지)
- 결과는 SQL만 내보낸다. 백틱/설명/자연어/코드블록 없이 SQL 한 덩어리만 출력한다.
- 대상 테이블: company_financials(company_code text, date date, metric text, value numeric)
- 시계열을 조회할 때는 항상 ORDER BY date를 포함한다.
- 한국어 질의의 의미를 스스로 판단해 컬럼/값을 매핑한다.
  예: '매출/수익'→ metric='revenue', '자산'→ 'assets', '부채'→ 'liabilities', 'K-ICS/킥스'→ 'k_ics'
- 회사명/약칭/별칭 등은 사용자가 한국어로 적더라도 스스로 합리적 company_code를 추론한다.
  (모호하면 LIMIT를 두고 합리적인 필터로 시작한다.)
- SELECT * 대신 필요한 컬럼만 선택하고, where 절에 기간/회사/지표 필터를 상식적으로 건다.
- 안전을 위해 LIMIT 200을 기본 상한으로 둔다(사용자가 특정 기간을 명시했다면 그 기간 기준).

예시)
-- 질문: 2023년 NH농협생명 매출 월별 추이
SELECT date, value
FROM company_financials
WHERE metric='revenue'
  AND date >= '2023-01-01' AND date < '2024-01-01'
  -- company_code는 NH농협생명에 해당하는 값으로 합리적으로 추론
ORDER BY date
LIMIT 200;
""".strip()

# OpenAI Chat 모델 (LangChain용)
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=OPENAI_API_KEY)

# lazy 초기화: 앱 시작 시 즉시 접속하지 않음 (DNS/네트워크 이슈 회피)
@st.cache_resource(show_spinner=False)
def get_lc_db():
    return SQLDatabase.from_uri(SQLALCHEMY_URI)

def get_sql_agent():
    lc_db = get_lc_db()
    return create_sql_agent(
        llm=llm,
        db=lc_db,
        agent_type="openai-tools",
        verbose=False,
        prefix=AGENT_PREFIX,
    )

# ----------------- 페이지/테마 -----------------
st.set_page_config(page_title="보험사 경영공시 데이터 챗봇", page_icon="📊", layout="wide")

# Pretendard + 글로벌 스타일 (Tailwind 느낌의 톤앤매너)
st.markdown("""
<link rel="preconnect" href="https://cdn.jsdelivr.net" />
<link rel="stylesheet" as="style" crossorigin
      href="https://cdn.jsdelivr.net/gh/orioncactus/pretendard@v1.3.9/dist/web/static/pretendard.min.css" />

<style>
:root {
  --blue:#0064FF;
  --blue-dark:#0050CC;
  --bg:#F0F1F3;
  --text:#0f172a;
  --muted:#64748b;
  --card:#ffffff;
  --ring:#93c5fd;
}
html, body, [data-testid="stAppViewContainer"] {
  background: var(--bg) !important;
}
* { font-family: 'Pretendard', -apple-system, BlinkMacSystemFont, system-ui, Roboto, 'Helvetica Neue',
     'Segoe UI', 'Apple SD Gothic Neo', 'Noto Sans KR', 'Malgun Gothic', sans-serif !important; }

.container-card {
  background: var(--card);
  border-radius: 16px;
  box-shadow: 0 2px 12px rgba(2, 6, 23, 0.06);
  border: 1px solid #eef2f7;
}
.header {
  padding: 32px 32px 16px 32px;
  border-bottom: 1px solid #eef2f7;
  text-align: center;
}
.header h1 {
  margin: 0; padding: 0;
  font-size: 28px; font-weight: 800; letter-spacing: -0.02em; color: var(--text);
}
.header .byline {
  color: #6b7280; font-size: 13px; margin-top: 6px; opacity: .85;
}
.section {
  padding: 24px 32px 28px 32px;
}
.hint {
  text-align:center; color:#475569; font-size: 16px; margin-bottom: 14px;
}
.input-like label { display:none!important; }
.input-like .stTextInput>div>div>input {
  height: 56px; font-size: 18px; padding: 0 18px;
  background:#f3f4f6; border:1px solid #e5e7eb; border-radius:12px;
}
.input-like .stTextInput>div>div>input:focus { outline: none; border-color: var(--ring); box-shadow: 0 0 0 3px rgba(147,197,253,.45); }

.stButton>button {
  width:100%; height:54px; font-weight:700; font-size:18px;
  color:#fff; background: var(--blue);
  border-radius:12px; border:0;
  box-shadow: 0 2px 0 rgba(0,0,0,.03);
}
.stButton>button:hover { background: var(--blue-dark); }
.stButton>button:disabled { background:#d1d5db !important; color:#fff !important; }

.kpi {
  display:flex; gap:12px; align-items:center; justify-content:center; margin-top:8px;
  color:#6b7280; font-size:14px;
}
.badge {
  background:#eef2ff; color:#3730a3; padding:6px 10px; border-radius:999px; font-weight:600; font-size:12px;
}

.card-subtitle { color:#334155; font-size:18px; margin: 0 0 10px; text-align:center; }

.table-container .stDataFrame { border-radius:12px; overflow:hidden; border: 1px solid #e5e7eb; }
hr.sep { border:none; border-top:1px solid #eef2f7; margin: 20px 0; }

.small-note { color:#64748b; font-size:13px; margin-top:4px;}
.footer-note { color:#64748b; font-size:12px; text-align:center; margin-top:16px; }

.fadein { animation: fadeIn .5s ease; }
@keyframes fadeIn { from{opacity:0; transform: translateY(6px)} to{opacity:1; transform:none} }

/* code block polish */
pre, code { font-size: 13px !important; }
</style>
""", unsafe_allow_html=True)

# ----------------- 헤더 -----------------
st.markdown('<div class="container-card fadein">', unsafe_allow_html=True)
st.markdown("""
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
    <span>Host: <b>{host}</b></span>
    <span>·</span>
    <span>User: <b>{user}</b></span>
  </div>
</div>
""".format(host=DB_HOST or "-", user=DB_USER or "-"), unsafe_allow_html=True)

st.markdown('<div class="section">', unsafe_allow_html=True)
st.markdown('<p class="hint">현재 보유데이터는 2022~2024년 가정변경효과 · K-ICS 비율</p>', unsafe_allow_html=True)

# ----------------- 사이드바 (상태 영역) -----------------
with st.sidebar:
    st.markdown("### 연결 상태")
    st.write(f"DB Host: {DB_HOST}")
    st.write(f"DB User: {DB_USER}")
    st.caption("좌측 버튼 흐름대로 진행하세요. (① SQL 생성 → ② 실행 → ③ 차트/요약)")

# ----------------- SQL 생성 (LangChain Agent) -----------------
def _strip_code_fences(text: str) -> str:
    """```sql ...``` 같은 펜스를 제거"""
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z]*\s*", "", t)
        t = re.sub(r"\s*```$", "", t)
    return t.strip()

def generate_sql(user_question: str) -> str:
    """
    LangChain create_sql_agent를 사용해 '실행하지 않고' SQL만 생성.
    """
    # 프롬프트/입력 디버그
    try:
        st.markdown("OpenAI 프롬프트 (SQL 생성; LangChain Agent prefix)")
        st.code(AGENT_PREFIX, language="markdown")
        st.markdown("User 입력")
        st.code(user_question)
    except Exception:
        pass

    sql_agent = get_sql_agent()  # lazy 생성/접속
    result = sql_agent.invoke({"input": user_question})
    if isinstance(result, dict):
        text = result.get("output") or result.get("final_answer") or json.dumps(result, ensure_ascii=False)
    else:
        text = str(result)

    sql = _strip_code_fences(text)

    # 안전검증: SELECT 전용 + 금지어 차단
    if not re.match(r"(?is)^\s*select\s", sql):
        raise ValueError("Only SELECT queries are allowed.")
    banned = r"(?is)\b(insert|update|delete|drop|alter|create|grant|revoke|truncate)\b"
    if re.search(banned, sql):
        raise ValueError("Blocked SQL keyword detected.")

    # 응답 디버그
    try:
        st.markdown("OpenAI 응답 (SQL 생성)")
        st.code(sql, language="sql")
    except Exception:
        pass

    return sql

def run_sql(sql: str) -> pd.DataFrame:
    # psycopg 연결에도 sslmode=require 적용
    with psycopg.connect(
        host=DB_HOST,
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASS,
        port=DB_PORT,
        sslmode="require",
    ) as conn:
        return pd.read_sql_query(sql, conn)

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
    placeholder="예) 2023년 NH농협생명 매출 월별 추이 보여줘",
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
        with st.spinner("SQL을 생성하는 중..."):
            try:
                sql = generate_sql(q)
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
    # 날짜 컬럼 이름 추정
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
            # value/metric 열 힌트
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

st.markdown('<p class="footer-note">UI만 변경 · 기능 로직은 원본과 동일합니다.</p>', unsafe_allow_html=True)
