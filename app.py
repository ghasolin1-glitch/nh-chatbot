# app.py — 보험사 경영공시 챗봇 (흰색 배경, 모바일 입력창 수정, 결과 하단 표시)
import os
import json
import re
import pandas as pd
import streamlit as st
import psycopg

# ====== LangChain / OpenAI LLM ======
from langchain_community.utilities import SQLDatabase

# create_sql_agent 경로 버전별 대응
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
DB_HOST = os.getenv("DB_HOST") or st.secrets.get("DB_HOST")      # e.g., aws-1-us-east-1.pooler.supabase.com
DB_NAME = os.getenv("DB_NAME") or st.secrets.get("DB_NAME", "postgres")
DB_USER = os.getenv("DB_USER") or st.secrets.get("DB_USER", "readonly")
DB_PASS = os.getenv("DB_PASS") or st.secrets.get("DB_PASS")
DB_PORT = int(os.getenv("DB_PORT") or st.secrets.get("DB_PORT", 5432))

if not OPENAI_API_KEY:
    st.error("OPENAI_API_KEY 설정이 되어 있지 않습니다.")
    st.stop()

client = OpenAI(api_key=OPENAI_API_KEY)

# ====== LangChain용 DB/LLM/에이전트 초기화 ======
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

# ----------------- 유틸: 출력 정리/검증 -----------------
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
        raise ValueError("Multiple statements are not allowed.")
    if not re.match(r"(?is)^\s*select\b", sql):
        raise ValueError("Only SELECT queries are allowed.")
    banned = r"(?is)\b(insert|update|delete|drop|alter|create|grant|revoke|truncate|copy|into|explain|with)\b"
    if re.search(banned, sql):
        raise ValueError("Blocked SQL keyword detected.")

# ----------------- 페이지/테마 -----------------
st.set_page_config(page_title="보험사 경영공시 챗봇", page_icon="📊", layout="centered")

# Pretendard + 글로벌 스타일 (모바일 타이틀 1줄 고정 포함)
st.markdown("""
<link rel="preconnect" href="https://cdn.jsdelivr.net" />
<link rel="stylesheet" as="style" crossorigin
      href="https://cdn.jsdelivr.net/gh/orioncactus/pretendard@v1.3.9/dist/web/static/pretendard.min.css" />

<style>
:root {
  --blue:#0064FF;
  --blue-dark:#0050CC;
  --bg:#ffffff;       /* ✅ 1. 배경 흰색으로 변경 */
  --text:#0f172a;
  --muted:#64748b;
  --card:#ffffff;
  --ring:#93c5fd;
}

html, body, [data-testid="stAppViewContainer"] { background: var(--bg) !important; }
* { font-family: 'Pretendard', -apple-system, BlinkMacSystemFont, system-ui, Roboto, 'Helvetica Neue',
      'Segoe UI', 'Apple SD Gothic Neo', 'Noto Sans KR', 'Malgun Gothic', sans-serif !important; }

.block-container { padding-top: 1.0rem; padding-bottom: 1.5rem; max-width: 860px; }
@media (max-width: 640px) { .block-container { padding-left: 0.8rem; padding-right: 0.8rem; max-width: 100%; } }

.container-card {
  background: var(--card);
  border-radius: 16px;
  box-shadow: 0 2px 12px rgba(2, 6, 23, 0.06);
  border: 1px solid #eef2f7;
}

/* ====== 헤더/타이틀 - 모바일 한 줄 고정 ====== */
.header {
  padding: 39px 20px 12px 20px;  /* ⬅ 상단 여백 +15px (약 15pt) */
  border-bottom: 1px solid #eef2f7;
  text-align: center;
}
.title-row {
  display: flex; align-items: center; justify-content: center; gap: 10px;
  flex-wrap: nowrap; max-width: 100%;
}
.header h1 {
  margin: 0; padding: 0;
  font-size: clamp(22px, 5.5vw, 36px);
  font-weight: 800; letter-spacing: -0.02em; color: var(--text);
  white-space: nowrap;      /* ✅ 한 줄 강제 */
  overflow: hidden;         /* ✅ 넘치면 숨김 */
  text-overflow: ellipsis;  /* ✅ 말줄임표 */
  max-width: 100%;
}
.header svg { flex-shrink: 0; }
.header .byline { color: #6b7280; font-size: 13px; margin-top: 6px; opacity: .85; }

/* ====== 섹션 ====== */
.section { padding: 18px 20px 22px 20px; }

/* ====== 입력창 강조: 파란빛 글로우 + 반짝효과 (pill) ====== */
.input-like label { display:none!important; }
.input-like .stTextInput>div>div>input {
  height: 56px; font-size: 17px; padding: 0 20px;
  background:#ffffff; border:1px solid #0064FF;
  border-radius: 9999px;  /* pill */
  box-shadow:
    0 0 10px rgba(0, 100, 255, 0.35),
    0 0 20px rgba(0, 100, 255, 0.20),
    0 0 30px rgba(0, 100, 255, 0.10);
  animation: glowPulse 2.2s infinite ease-in-out; /* ✨ 반짝반짝 */
  
  /* ✅ 2. 모바일 반응형 수정 */
  box-sizing: border-box; 
  max-width: 100%; 
}
.input-like .stTextInput>div>div>input:focus {
  outline: none;
  border-color: #4f9cff;
  box-shadow:
    0 0 12px rgba(0, 100, 255, 0.6),
    0 0 24px rgba(0, 100, 255, 0.35),
    0 0 32px rgba(0, 100, 255, 0.25);
  animation: glowPulseFast 1.4s infinite ease-in-out; /* 더 강한 반짝 */
}
/* ✨ 반짝이는 Keyframes */
@keyframes glowPulse {
  0%, 100% {
    box-shadow:
      0 0 10px rgba(0, 100, 255, 0.25),
      0 0 20px rgba(0, 100, 255, 0.15),
      0 0 30px rgba(0, 100, 255, 0.05);
  }
  50% {
    box-shadow:
      0 0 14px rgba(0, 100, 255, 0.45),
      0 0 28px rgba(0, 100, 255, 0.25),
      0 0 32px rgba(0, 100, 255, 0.18);
  }
}
@keyframes glowPulseFast {
  0%, 100% {
    box-shadow:
      0 0 12px rgba(0, 100, 255, 0.45),
      0 0 22px rgba(0, 100, 255, 0.25),
      0 0 28px rgba(0, 100, 255, 0.15);
  }
  50% {
    box-shadow:
      0 0 18px rgba(0, 100, 255, 0.75),
      0 0 34px rgba(0, 100, 255, 0.45),
      0 0 40px rgba(0, 100, 255, 0.30);
  }
}

/* 버튼 기본(전체폭) → 이번엔 절반폭으로 중앙 정렬은 컬럼으로 처리 */
.stButton>button {
  width:100%; height:48px; font-weight:700; font-size:16px;
  color:#fff; background: var(--blue);
  border-radius:12px; border:0; box-shadow: 0 2px 0 rgba(0,0,0,.03);
}
.stButton>button:hover { background: var(--blue-dark); }
.stButton>button:disabled { background:#d1d5db !important; color:#fff !important; }

/* 카드/표 등 */
.card-subtitle { color:#334155; font-size:17px; margin: 0 0 10px; text-align:center; }
.table-container .stDataFrame { border-radius:12px; overflow:hidden; border: 1px solid #e5e7eb; }
hr.sep { border:none; border-top:1px solid #eef2f7; margin: 18px 0; }
.small-note { color:#64748b; font-size:12px; margin-top:4px;}
.footer-note { color:#64748b; font-size:12px; text-align:center; margin-top:12px; }

.fadein { animation: fadeIn .5s ease; }
@keyframes fadeIn { from{opacity:0; transform: translateY(6px)} to{opacity:1; transform:none} }

pre, code { font-size: 13px !important; }

@media (max-width: 640px) {
  .card-subtitle { font-size: 16px; }
  .input-like .stTextInput>div>div>input { height: 54px; font-size: 16px; }
}
</style>
""", unsafe_allow_html=True)

# ----------------- 헤더 -----------------
st.markdown('<div class="container-card fadein">', unsafe_allow_html=True)
st.markdown("""
<div class="header">
  <div class="title-row">
    <h1>보험사 경영공시 챗봇</h1>
    <svg xmlns="http://www.w3.org/2000/svg" width="26" height="26" viewBox="0 0 24 24"
         fill="none" stroke="#0064FF" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
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
  <div class="byline">made by 태훈 · 현철</div>
</div>
""", unsafe_allow_html=True)

# ===== (위치 변경) 실행 결과 슬롯이 헤더 '아래' -> 입력창 '아래'로 이동 =====
# result_area = st.container()  <- 이 줄이 아래로 이동

# ===================== 입력 섹션 =====================
st.markdown('<div class="section">', unsafe_allow_html=True)

# ----------------- SQL 생성 (LangChain Agent) -----------------
def generate_sql(user_question: str) -> str:
    try:
        with st.expander("OpenAI 프롬프트 (SQL 생성; LangChain Agent prefix)", expanded=False):
            st.code(AGENT_PREFIX, language="markdown")
        st.caption("User 입력")
        st.code(user_question)
    except Exception:
        pass

    sql_agent = get_sql_agent()
    result = sql_agent.invoke({"input": user_question})

    if isinstance(result, dict):
        text = result.get("output") or result.get("final_answer") or json.dumps(result, ensure_ascii=False)
    else:
        text = str(result)

    sql = _extract_first_select(text)
    _validate_sql_is_select(sql)

    try:
        st.caption("OpenAI 응답 (SQL 생성)")
        st.code(sql, language="sql")
    except Exception:
        pass

    return sql

def run_sql(sql: str) -> pd.DataFrame:
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
        with st.expander("OpenAI 프롬프트 (요약)", expanded=False):
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
        st.caption("OpenAI 응답 (요약)")
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

# ----------------- 버튼: 절반 너비(가운데) + 원클릭 실행 -----------------
c1, c2, c3 = st.columns([1, 2, 1])   # 가운데 컬럼만 버튼 -> 전체 대비 50% 폭
with c2:
    go_btn = st.button("실행", use_container_width=True)

# ✅ 3. (위치 변경) 실행 결과가 들어갈 슬롯
# 입력창과 버튼 '아래', 로직 '위'로 이동
result_area = st.container()

# 클릭 시: 결과는 'result_area'에 그리기
if go_btn:
    if not q:
        with result_area:
            st.warning("질문을 입력하세요.")
    else:
        # 1) SQL 생성
        try:
            sql = generate_sql(q)
            st.session_state["sql"] = sql
        except Exception as e:
            with result_area:
                st.error(f"SQL 생성 오류: {e}")
            st.stop()

        # 2) 즉시 실행 + 상단 결과 슬롯에 렌더링
        try:
            df = run_sql(st.session_state["sql"])
            st.session_state["df"] = df
            with result_area:
                st.markdown('<div class="section" style="padding-top: 5px;">', unsafe_allow_html=True) # 섹션 패딩 조절
                st.markdown('#### 실행 결과')
                if df.empty:
                    st.info("결과가 없습니다.")
                else:
                    st.markdown('<div class="table-container">', unsafe_allow_html=True)
                    st.dataframe(df, use_container_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
        except Exception as e:
            with result_area:
                st.error(f"DB 실행 오류: {e}")
            st.stop()

        # 3) 자동 요약 생성 (있을 때만)
        df_prev = st.session_state.get("df")
        if df_prev is not None and not df_prev.empty:
            try:
                # 요약은 result_area에 이어서 표시
                with result_area:
                    with st.spinner("요약 생성 중..."):
                        summary = summarize_answer(q, df_prev)
                        st.success(summary)
                        st.session_state["summary"] = summary
            except Exception as e:
                with result_area:
                    st.error(f"요약 오류: {e}")

st.markdown('<hr class="sep"/>', unsafe_allow_html=True)

# 필요 시 요약 버튼(재생성 용도)
df_prev = st.session_state.get("df")
if df_prev is not None and not df_prev.empty:
    if st.button("요약 생성", use_container_width=True):
        with result_area:
            with st.spinner("요약 생성 중..."):
                try:
                    summary = summarize_answer(q, df_prev)
                    st.success(summary)
                    st.session_state["summary"] = summary
                except Exception as e:
                    st.error(f"요약 오류: {e}")
else:
    st.caption("실행 결과가 표시되면 요약을 볼 수 있습니다.")

st.markdown('</div>', unsafe_allow_html=True)  # section 종료
st.markdown('</div>', unsafe_allow_html=True)  # container-card 종료