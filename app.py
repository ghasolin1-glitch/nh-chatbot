import os
import re
import pandas as pd
import streamlit as st
import psycopg
from openai import OpenAI

from dotenv import load_dotenv
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
client = OpenAI(api_key=OPENAI_API_KEY)

st.set_page_config(page_title="회사 데이터 챗봇(정형 데이터 전용)", page_icon="📈", layout="wide")
st.title("회사 데이터 챗봇 — 정형 데이터(SQL)만 사용")

with st.sidebar:
    st.markdown("### 연결 상태")
    st.write(f"DB Host: {DB_HOST}")
    st.write(f"DB User: {DB_USER}")

# ----------------- SQL 생성 시스템 프롬프트 -----------------
SQL_SYSTEM_PROMPT = """You are a PostgreSQL SQL generator.
Return ONLY a SQL query (no backticks). Rules:
- Use SELECT queries ONLY (no INSERT/UPDATE/DELETE/ALTER/DROP/CREATE/GRANT/REVOKE).
- Target table: company_financials(company_code text, date date, metric text, value numeric)
- Always include ORDER BY date when selecting time series.
- Examples:
  Q: 2023년 NH농협생명 revenue 월별
  A: SELECT date, value FROM company_financials
     WHERE company_code='NH' AND metric='revenue'
       AND date >= '2023-01-01' AND date < '2024-01-01'
     ORDER BY date;

- Map common Korean phrasing to fields logically (e.g., '매출' -> metric='revenue').
- When unsure, default to selecting limited rows with sensible filters, not *
"""

# (선택) 간단한 한글→코드/메트릭 매핑 힌트
COMPANY_MAP = {
    "농협생명": "NH",
    "NH농협생명": "NH",
    "한화생명": "HANWHA",
    "삼성생명": "SAMSUNG",
}
METRIC_MAP = {
    "매출": "revenue",
    "자산": "assets",
    "부채": "liabilities",
    "수익": "revenue",
    "solvency": "solvency_ratio",
    "k-ics": "k_ics",
}

def apply_simple_mapping(q: str) -> str:
    # 질문에서 회사/지표를 영문 코드로 유도하는 텍스트 힌트 생성
    hints = []
    for k, v in COMPANY_MAP.items():
        if k in q:
            hints.append(f"company_code should be '{v}' for '{k}'")
    for k, v in METRIC_MAP.items():
        if k.lower() in q.lower():
            hints.append(f"metric should be '{v}' for '{k}'")
    return ("\nHINTS:\n" + "\n".join(hints)) if hints else ""

def generate_sql(user_question: str) -> str:
    hints = apply_simple_mapping(user_question)
    messages = [
        {"role": "system", "content": SQL_SYSTEM_PROMPT + hints},
        {"role": "user", "content": user_question},
    ]
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0
    )
    sql = resp.choices[0].message.content.strip()

    # --- 안전장치 ---
    if not re.match(r"(?is)^\s*select\s", sql):
        raise ValueError("Only SELECT queries are allowed.")
    banned = r"(?is)\b(insert|update|delete|drop|alter|create|grant|revoke|truncate)\b"
    if re.search(banned, sql):
        raise ValueError("Blocked SQL keyword detected.")
    # 너무 광범위한 SELECT * 방지(권장): 필요시 주석 해제
    # if re.search(r"(?is)select\s+\*\s+from", sql):
    #     raise ValueError("SELECT * is blocked. Please select named columns.")
    return sql

def run_sql(sql: str) -> pd.DataFrame:
    with psycopg.connect(host=DB_HOST, dbname=DB_NAME, user=DB_USER, password=DB_PASS, port=DB_PORT) as conn:
        return pd.read_sql_query(sql, conn)

def summarize_answer(q: str, df: pd.DataFrame) -> str:
    # 결과 요약 멘트 (간단한 LLM 요약)
    preview_csv = df.head(20).to_csv(index=False)
    prompt = f"""질문: {q}
아래 CSV 일부를 참고해서 3문장 이내로 한국어 요약을 써줘. 단위와 기간을 분명히 써.
CSV 미리보기(최대 20행):
{preview_csv}
"""
    r = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role":"user","content": prompt}],
        temperature=0.2
    )
    return r.choices[0].message.content.strip()

# ----------------- UI -----------------
q = st.text_input("질문을 입력하세요 (예: 'NH농협생명 2023년 매출 월별 추이 보여줘')")

col1, col2 = st.columns(2)
with col1:
    st.subheader("① SQL 생성")
    if st.button("SQL 만들기"):
        if not q:
            st.warning("질문을 입력하세요.")
        else:
            try:
                sql = generate_sql(q)
                st.code(sql, language="sql")
                st.session_state["sql"] = sql
            except Exception as e:
                st.error(f"SQL 생성 오류: {e}")

with col2:
    st.subheader("② SQL 실행")
    if st.button("실행"):
        sql = st.session_state.get("sql")
        if not sql:
            st.warning("먼저 'SQL 만들기'를 클릭하세요.")
        else:
            try:
                df = run_sql(sql)
                if df.empty:
                    st.info("결과가 없습니다.")
                else:
                    st.dataframe(df, use_container_width=True)
                    st.session_state["df"] = df
            except Exception as e:
                st.error(f"DB 실행 오류: {e}")

st.markdown("---")
st.subheader("③ 차트 & 요약")

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
        except Exception as _:
            pass

    if st.button("요약 생성"):
        try:
            summary = summarize_answer(q, df_prev)
            st.write(summary)
        except Exception as e:
            st.error(f"요약 오류: {e}")
else:
    st.caption("실행 결과가 표시되면 차트와 요약을 볼 수 있습니다.")
