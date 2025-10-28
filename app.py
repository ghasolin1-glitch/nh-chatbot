# app.py — 보험사 경영공시 챗봇 (요약만 표시 버전)
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

# ----------------- 환경변수 -----------------
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

# ====== LangChain DB/LLM/에이전트 ======
SQLALCHEMY_URI = (
    f"postgresql+psycopg://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}?sslmode=require"
)

AGENT_PREFIX = """
당신은 PostgreSQL SQL 전문가다.
- 오직 SELECT만 작성한다. (INSERT/UPDATE/DELETE 등 금지)
- 대상 테이블: kics_solvency_data_flexible
- 시계열 조회 시 ORDER BY date 포함.
- 한국어 질의 의미를 스스로 컬럼에 매핑한다.
- 회사명은 “농협생명, 삼성생명, 교보생명, 한화생명 ...” 등 DB 기준으로 추론한다.
""".strip()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=OPENAI_API_KEY)

@st.cache_resource(show_spinner=False)
def get_lc_db():
    return SQLDatabase.from_uri(SQLALCHEMY_URI)

def get_sql_agent():
    return create_sql_agent(
        llm=llm, db=get_lc_db(), agent_type="openai-tools",
        verbose=False, prefix=AGENT_PREFIX
    )

# ====== 유틸 ======
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

# ====== 테마 ======
st.set_page_config(page_title="보험사 경영공시 챗봇", page_icon="📊", layout="centered")

st.markdown("""
<link rel="stylesheet" href="https://cdn.jsdelivr.net/gh/orioncactus/pretendard@v1.3.9/dist/web/static/pretendard.min.css" />
<style>
* {font-family:'Pretendard',sans-serif!important;}
.block-container{padding-top:1.0rem;max-width:860px;}
.header{text-align:center;padding:40px 20px 20px 20px;}
.header h1{font-size:clamp(25px,7vw,48px);font-weight:800;}
.byline{color:#6b7280;font-size:13px;margin-top:6px;}
.input-like label{display:none!important;}
.input-like .stTextInput>div>div>input{
  height:56px;font-size:17px;padding:0 20px;
  border:1px solid #0064FF;border-radius:9999px;
  box-shadow:0 0 15px rgba(0,100,255,0.4);
  animation:glowPulse 2s infinite ease-in-out;
}
@keyframes glowPulse{0%,100%{box-shadow:0 0 10px rgba(0,100,255,0.3);}
50%{box-shadow:0 0 20px rgba(0,100,255,0.6);}}
.stButton>button{
  width:100%;height:48px;font-weight:700;font-size:16px;
  color:#fff;background:#0064FF;border-radius:12px;border:0;
}
.stButton>button:hover{background:#004fe0;}
</style>
""", unsafe_allow_html=True)

# ====== 헤더 ======
st.markdown("""
<div class="header">
  <svg xmlns="http://www.w3.org/2000/svg" width="30" height="30" fill="none"
       stroke="#0064FF" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
    <path d="M12 8V4H8V8H12Z" /><path d="M16 8V4H12V8H16Z" />
    <path d="M12 14V12H8V14H12Z" /><path d="M16 14V12H12V14H16Z" />
  </svg>
  <h1>보험사 경영공시 챗봇</h1>
  <div class="byline">made by 태훈 · 현철</div>
</div>
""", unsafe_allow_html=True)

# ====== 입력 ======
st.markdown('<div class="input-like">', unsafe_allow_html=True)
q = st.text_input("질문", placeholder="예) 2023년 NH농협생명 K-ICS비율 월별 추이 보여줘",
                  label_visibility="collapsed", key="q_input")
st.markdown('</div>', unsafe_allow_html=True)

c1, c2, c3 = st.columns([1, 1.5, 1])
with c2:
    go_btn = st.button("실행", use_container_width=True)

result_area = st.container()

# ====== 기능 ======
def generate_sql(user_question: str) -> str:
    sql_agent = get_sql_agent()
    result = sql_agent.invoke({"input": user_question})
    if isinstance(result, dict):
        text = result.get("output") or result.get("final_answer") or json.dumps(result, ensure_ascii=False)
    else:
        text = str(result)
    sql = _extract_first_select(text)
    _validate_sql_is_select(sql)
    return sql

def run_sql(sql: str) -> pd.DataFrame:
    with psycopg.connect(
        host=DB_HOST, dbname=DB_NAME, user=DB_USER,
        password=DB_PASS, port=DB_PORT, sslmode="require"
    ) as conn:
        return pd.read_sql_query(sql, conn)

def summarize_answer(q: str, df: pd.DataFrame) -> str:
    preview_csv = df.head(20).to_csv(index=False)
    prompt = f"""질문: {q}
너는 뛰어난 재무분석가야. 아래 CSV 일부를 참고해서 한국어 요약을 써줘.
단위와 기간을 분명히 써.
CSV:
{preview_csv}
"""
    r = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )
    return r.choices[0].message.content.strip()

# ====== 동작 ======
if go_btn:
    if not q:
        with result_area:
            st.warning("질문을 입력하세요.")
    else:
        try:
            sql = generate_sql(q)
            df = run_sql(sql)
            if df.empty:
                with result_area:
                    st.info("데이터가 없습니다.")
            else:
                with result_area:
                    with st.spinner("요약 생성 중..."):
                        summary = summarize_answer(q, df)
                        st.success(summary)
        except Exception as e:
            with result_area:
                st.error(f"오류: {e}")
