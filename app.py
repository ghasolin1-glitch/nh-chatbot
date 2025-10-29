# 2025102
# app.py — 보험사 경영공시 챗봇 (아이콘/버튼너비 수정)
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
- 'YY년 MM월' 패턴의 경우, YY가 00~24 → 2000~2024년으로, YY가 25~99 → 2000 + YY로 변환한다. (예: 25년 6월 → 202506)
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
  --bg:#ffffff;
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
  padding: 39px 20px 12px 20px;
  border-bottom: 1px solid #eef2f7;
  text-align: center;
}
/* ✅ 1. (수정) 아이콘/텍스트 세로(column) 정렬 */
.title-row {
  display: flex;
  flex-direction: column; /* 아이콘/텍스트 세로 배치 */
  align-items: center; 
  justify-content: center; 
  gap: 10px;
  max-width: 100%;
}
.header h1 {
  margin: 0; padding: 0;
  font-size: clamp(25px, 7vw, 48px);
  font-weight: 800; letter-spacing: -0.02em; color: var(--text);
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
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
  border-radius: 9999px;
  box-shadow:
    0 0 10px rgba(0, 100, 255, 0.35),
    0 0 20px rgba(0, 100, 255, 0.20),
    0 0 30px rgba(0, 100, 255, 0.10);
  animation: glowPulse 2.2s infinite ease-in-out;
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
  animation: glowPulseFast 1.4s infinite ease-in-out;
}
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

/* 버튼 */
.stButton>button {
  width:100%; height:48px; font-weight:700; font-size:16px;
  color:#fff; background: var(--blue);
  border-radius:12px; border:0; box-shadow: 0 2px 0 rgba(0,0,0,.03);
}
.stButton>button:hover { background: var(--blue-dark); }
.stButton>button:disabled { background:#d1d5db !important; color:#fff !important; }

/* 카드/표 등 */
.table-container .stDataFrame { border-radius:12px; overflow:hidden; border: 1px solid #e5e7eb; }
.fadein { animation: fadeIn .5s ease; }
@keyframes fadeIn { from{opacity:0; transform: translateY(6px)} to{opacity:1; transform:none} }
            
/* ✅ expander 내부에서 코드/텍스트 겹침 완전 방지 */
.streamlit-expanderContent {
  white-space: normal !important;
  overflow-wrap: anywhere !important;
  word-break: break-word !important;
}

.streamlit-expanderContent pre, 
.streamlit-expanderContent code {
  white-space: pre-wrap !important;
  word-break: break-word !important;
  overflow-x: auto !important;
  overflow-y: auto !important;
  display: block !important;
  max-width: 100% !important;
  box-sizing: border-box !important;
  font-size: 14px !important;
  line-height: 1.5em !important;
  background-color: #f9fafb !important;
  padding: 10px 12px !important;
  border-radius: 8px !important;
}

.streamlit-expanderHeader {
  font-weight: 600 !important;
  font-size: 16px !important;
}

</style>
""", unsafe_allow_html=True)

# ----------------- 헤더 -----------------
st.markdown('<div class="container-card fadein">', unsafe_allow_html=True)
# ✅ 1. (수정) SVG 아이콘을 h1 타이틀 위로 이동
st.markdown("""
<div class="header">
  <div class="title-row">
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
    <h1>보험사 경영공시 챗봇</h1>
  </div>
  <div class="byline">made by 태훈 · 현철</div>
</div>
""", unsafe_allow_html=True)

# ===================== 입력 섹션 =====================
st.markdown('<div class="section">', unsafe_allow_html=True)

# ----------------- SQL 생성 (LangChain Agent) -----------------
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
        host=DB_HOST,
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASS,
        port=DB_PORT,
        sslmode="require",
    ) as conn:
        return pd.read_sql_query(sql, conn)

# ----------------- 요약 생성 -----------------
def summarize_answer(q: str, df: pd.DataFrame) -> str:
    preview_csv = df.to_csv(index=False)
    prompt = f"""
질문: {q}
너는 뛰어난 재무분석가이자 데이터 시각화 전문가야.
다음 CSV 데이터를 기반으로, 트렌드를 분석해 **한국어로 요약**해줘.
- 수치의 단위와 기간을 반드시 명시해.
- 데이터 패턴(증가/감소, 최고점, 평균 등)을 설명해.
- 이후 Python 코드가 차트를 자동 생성할 것이므로, 시각화에 필요한 주요 컬럼 1~2개를 명시적으로 언급해.
예: 'closing_ym'을 X축으로, 'k_ics_ratio'를 Y축으로 사용하면 좋겠다.
CSV 일부 샘플:
{preview_csv}
"""
    r = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )
    return r.choices[0].message.content.strip()


# ----------------- 입력창 -----------------
st.markdown('<div class="input-like">', unsafe_allow_html=True)
q = st.text_input(
    label="질문",
    placeholder="예)23년12월 농협생명 K-ICS비율 알려줘",
    label_visibility="collapsed",
    key="q_input"
)
st.markdown('</div>', unsafe_allow_html=True)

# ----------------- 버튼: 60% 너비(가운데) + 원클릭 실행 -----------------
# ✅ 2. (수정) 컬럼 비율을 [1, 3, 1]로 변경 (중앙 3/5 = 60%)
c1, c2, c3 = st.columns([1, 1.5, 1])   # 가운데 컬럼만 버튼 -> 전체 대비 60% 폭
with c2:
    go_btn = st.button("실행", use_container_width=True)

# 실행 결과가 들어갈 슬롯
result_area = st.container()

# 클릭 시: 결과는 'result_area'에 그리기
if go_btn:
    if not q:
        with result_area:
            st.warning("질문을 입력하세요.")
    else:
        # ✅ 진행상황 표시 (최종 결과 후 자동 제거)
        status_placeholder = st.empty()  # 임시 공간

        with status_placeholder.container():
            with st.status("진행 중입니다...", expanded=True) as status:
                # 1) SQL 생성
                try:
                    status.write("① SQL 생성 중...")
                    sql = generate_sql(q)
                    st.session_state["sql"] = sql
                    status.update(label="SQL 생성 완료 ✅", state="running")
                except Exception as e:
                    status.update(label="SQL 생성 오류 ❌", state="error")
                    with result_area:
                        st.error(f"SQL 생성 오류: {e}")
                    st.stop()

                # 2) SQL 실행
                try:
                    status.write("② 데이터 조회 중...")
                    df = run_sql(st.session_state["sql"])
                    st.session_state["df"] = df
                    status.update(label="데이터 조회 완료 ✅", state="running")
                except Exception as e:
                    status.update(label="DB 실행 오류 ❌", state="error")
                    with result_area:
                        st.error(f"DB 실행 오류: {e}")
                    st.stop()

                # 3) 자동 요약 생성만 표시
                if df is not None and not df.empty:
                    try:
                        status.write("③ 요약 생성 중...")

                        with result_area:
                            with st.spinner("요약 생성 중..."):
                                summary = summarize_answer(q, df)

                                # ✅ 최종 요약 결과 표시
                                # ✅ 요약결과를 밝은 회색 카드로 표시
                                st.markdown(
                                    f"""
                                    <div style="
                                        background-color:#F5F6F8;
                                        color:#0F172A;
                                        padding:18px 22px;
                                        border-radius:12px;
                                        font-size:16px;
                                        line-height:1.6em;
                                        border:1px solid #E5E7EB;
                                        ">
                                        {summary}
                                    </div>
                                    """,
                                    unsafe_allow_html=True
                                )

                                st.session_state["summary"] = summary


                                # ✅ Altair 기반 시각화 (matplotlib 제거)
                                import altair as alt
                                alt.themes.enable('none')  # Streamlit 다크모드 테마 비활성화


                                try:
                                    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
                                    date_cols = [c for c in df.columns if re.search(r"(date|ym|month|year)", c, re.I)]
                                    cat_cols = [c for c in df.columns if re.search(r"(name|company|보험|사명)", c, re.I)]

                                    # --- 1️⃣ 회사별 분포 (막대그래프) ---
                                    if numeric_cols and cat_cols:
                                        x_col = cat_cols[0]
                                        y_col = numeric_cols[0]
                                        st.markdown("### 📊 데이터 분포 (회사별)")

                                        # 공통 옵션: 글자색 검정, 축색 검정
                                        chart = (
                                            alt.Chart(df)
                                            .mark_bar(color="#0064FF")
                                            .encode(
                                                x=alt.X(x_col, sort='-y', title=x_col, axis=alt.Axis(labelColor="#0F172A", titleColor="#0F172A")),
                                                y=alt.Y(y_col, title=y_col, axis=alt.Axis(labelColor="#0F172A", titleColor="#0F172A")),
                                                tooltip=[x_col, y_col]
                                            )
                                            .properties(width="container", height=400, background="#F5F6F8")  # 밝은 회색 배경
                                        )


                                        # ✅ 수치 라벨 추가 (Altair text layer)
                                        text = (
                                            alt.Chart(df)
                                            .mark_text(
                                                align='center',
                                                baseline='bottom',
                                                dy=-3,
                                                color="#0F172A",
                                                fontSize=10
                                            )
                                            .encode(x=x_col, y=y_col, text=alt.Text(y_col, format=".1f"))
                                        )

                                        st.altair_chart(chart + text, use_container_width=True)

                                    # --- 2️⃣ 시계열 추이 (선그래프) ---
                                    elif numeric_cols and date_cols:
                                        x_col = date_cols[0]
                                        y_col = numeric_cols[0]
                                        st.markdown("### 📈 시계열 추이")

                                        line = (
                                            alt.Chart(df)
                                            .mark_line(color="#0064FF", point=True)
                                            .encode(
                                                x=alt.X(x_col, title=x_col, axis=alt.Axis(labelColor="#0F172A", titleColor="#0F172A")),
                                                y=alt.Y(y_col, title=y_col, axis=alt.Axis(labelColor="#0F172A", titleColor="#0F172A")),
                                                tooltip=[x_col, y_col]
                                            )
                                            .properties(width="container", height=400, background="#F5F6F8")
                                        )


                                        st.altair_chart(line, use_container_width=True)

                                except Exception as e:
                                    st.info(f"차트를 생성할 수 없습니다: {e}")

                                # ✅ 요약 결과 아래에 SQL 쿼리 및 프롬프트/결과 보기 토글 추가
                                with st.expander("🔍 SQL 요청 및 결과 보기", expanded=False):
                                    st.markdown("### 🧩 생성된 SQL 문")
                                    st.code(st.session_state.get("sql", ""), language="sql")

                                    st.markdown("### 💬 SQL 생성 프롬프트")
                                    sql_prompt = AGENT_PREFIX.strip()
                                    st.code(sql_prompt, language="markdown")

                                    st.markdown("### 💬 요약 생성 프롬프트")
                                    if "df" in st.session_state:
                                        sample_preview = st.session_state["df"].head(3).to_csv(index=False)
                                        summary_prompt = f"""
                                        질문: {q}
                                        너는 뛰어난 재무분석가이자 데이터 시각화 전문가야.
                                        다음 CSV 데이터를 기반으로, 트렌드를 분석해 **한국어로 요약**해줘.
                                        - 수치의 단위와 기간을 반드시 명시해.
                                        - 데이터 패턴(증가/감소, 최고점, 평균 등)을 설명해.
                                        - 이후 Python 코드가 차트를 자동 생성할 것이므로, 시각화에 필요한 주요 컬럼 1~2개를 명시적으로 언급해.
                                        예: 'closing_ym'을 X축으로, 'k_ics_ratio'를 Y축으로 사용하면 좋겠다.
                                        CSV 일부 샘플:
                                {sample_preview}
                                """
                                        st.code(summary_prompt.strip(), language="markdown")

                                    st.markdown("### 📊 쿼리 결과(DataFrame)")
                                    st.dataframe(st.session_state.get("df"), use_container_width=True)




                        status.update(label="요약 완료 ✅", state="complete")

                    except Exception as e:
                        status.update(label="요약 오류 ❌", state="error")
                        with result_area:
                            st.error(f"요약 오류: {e}")
                else:
                    status.update(label="데이터 없음 ⚠️", state="error")
                    with result_area:
                        st.info("데이터가 없습니다. 다른 질문을 입력해보세요.")

        # ✅ 최종 결과가 나오면 진행상황 박스를 제거
        status_placeholder.empty()


st.markdown('</div>', unsafe_allow_html=True)  # section 종료
st.markdown('</div>', unsafe_allow_html=True)  # container-card 종료