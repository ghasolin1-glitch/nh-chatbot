# app.py — 버그 수정 (st.text_input + Glow CSS 적용)
import os
import json
import re
import pandas as pd
import streamlit as st
import psycopg

# ====== LangChain / OpenAI LLM (이전 코드 유지) ======
from langchain_community.utilities import SQLDatabase

# create_sql_agent 경로 버전별 대응 (이전 코드 유지)
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

# ----------------- 환경변수/시크릿 (이전 코드 유지) -----------------
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

# ====== LangChain용 DB/LLM/에이전트 초기화 (이전 코드 유지) ======
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
- 회사명은 "미래에셋생명,흥국화재,한화생명,한화손해,iM라이프생명,흥국생명,메리츠화재,KB생명,신한생명,DB생명,하나생명,BNP생명,푸본현대생명,ABL생명,DB손해,동양생명,농협생명,삼성화재,교보라이프플래닛생명,메트라이프생명,처브라이f생명보험,AIA생명,현대해상,교보생명,롯데손해,KDB생명,라이나생명,IBK생명,코리안리,KB손해,삼성생명,농협손보"로 DB에 저장되어있다.
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

# ----------------- 유틸: 출력 정리/검증 (이전 코드 유지) -----------------
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
st.set_page_config(page_title="보험사 경영공시 챗봇", page_icon="🤖", layout="centered")

# ✅ (수정) CSS: .glow-input 대신 st.text_input 위젯을 직접 타겟팅
st.markdown("""
<style>
:root {
    --blue:#0064FF;
    --blue-dark:#0050CC;
}

html, body, [data-testid="stAppViewContainer"] { background: #ECEEF1 !important; }
* { font-family:'Pretendard',sans-serif !important; }

.header { text-align:center; margin-top:40px; }
.title { font-size:32px; font-weight:900; }

.byline { color:#6b7280; font-size:13px; margin-bottom:25px; }

/* ✅ (수정) .glow-input 대신 Streamlit 위젯을 직접 스타일링 */
[data-testid="stTextInput"] {
    width: 480px;
    margin: auto;
}
[data-testid="stTextInput"] > div > div > input {
    background: white;
    border: 2px solid var(--blue);
    border-radius: 999px;
    padding: 10px 25px;
    text-align: center;
    font-size: 18px;
    box-shadow:
        0 0 25px rgba(0, 100, 255, .55),
        0 0 50px rgba(0, 100, 255, .35);
    animation: glowPulse 2s infinite ease-in-out;
}
[data-testid="stTextInput"] > div > div > input:focus {
    outline: none !important;
}

@keyframes glowPulse {
    50% {
        box-shadow:
            0 0 40px rgba(0, 100, 255, .9),
            0 0 70px rgba(0, 100, 255, .5);
    }
}

.bot-icon svg path {
    stroke: var(--blue) !important;
    stroke-width: 1.8 !important;
    fill: none !important;
}

.stButton>button {
    width: 100%; height: 48px; font-weight: 700; font-size: 16px;
    color: #fff; background: var(--blue);
    border-radius: 12px; border: 0; box-shadow: 0 2px 0 rgba(0, 0, 0, .03);
}
.stButton>button:hover { background: var(--blue-dark); }
.stButton>button:disabled { background: #d1d5db !important; color: #fff !important; }

.table-container .stDataFrame {
    border-radius: 12px; overflow: hidden; border: 1px solid #e5e7eb;
}
</style>
""", unsafe_allow_html=True)

# ----------------- 헤더 (신규 UI 유지) -----------------
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

# ----------------- SQL 생성 (이전 코드 유지) -----------------
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

# ----------------- SQL 실행 (이전 코드 유지) -----------------
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

# ----------------- 요약 생성 (이전 코드 유지) -----------------
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
# ✅ (수정) HTML/JS 입력창 대신 Streamlit 기본 st.text_input 사용
# CSS가 이 위젯에 Glow 스타일을 적용할 것입니다.
q = st.text_input(
    "질문",
    placeholder="예) 2023년 농협생명 K-ICS비율 알려줘",
    label_visibility="collapsed",
    key="user_q"  # session_state 키
)

st.write("") # 스페이서

# ----------------- 버튼: (신규 UI) 전체 너비 버튼 -----------------
go_btn = st.button("실행", use_container_width=True)

# ✅ (수정) 'q' 변수가 이제 st.text_input의 값이므로 로직이 정상 작동
if go_btn:
    if not q:
        st.warning("질문을 입력하세요.")
    else:
        # 1) SQL 생성 (이전 로직)
        try:
            sql = generate_sql(q)
            st.session_state["sql"] = sql
        except Exception as e:
            st.error(f"SQL 생성 오류: {e}")
            st.stop()

        # 2) 즉시 실행 + 하단 결과 렌더링 (이전 로직)
        try:
            df = run_sql(st.session_state["sql"])
            st.session_state["df"] = df
            st.markdown('#### 실행 결과')
            if df.empty:
                st.info("결과가 없습니다.")
            else:
                st.markdown('<div class="table-container">', unsafe_allow_html=True)
                st.dataframe(df, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
        except Exception as e:
            st.error(f"DB 실행 오류: {e}")
            st.stop()

        # 3) 자동 요약 생성 (이전 로직)
        df_prev = st.session_state.get("df")
        if df_prev is not None and not df_prev.empty:
            try:
                with st.spinner("요약 생성 중..."):
                    summary = summarize_answer(q, df_prev)
                    st.success(summary)
                    st.session_state["summary"] = summary
            except Exception as e:
                st.error(f"요약 오류: {e}")