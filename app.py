import streamlit as st
import pandas as pd
import time
import threading
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor

# API 키 세팅
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# 체크리스트 항목
CHECKLIST = {
    "역할": "프롬프트에 역할이 명시되어 있는가?",
    "대상": "프롬프트에 대상이 명시되어 있는가?",
    "정보": "배경 정보가 포함되어 있는가?",
    "작업": "명확한 작업이 명시되어 있는가?",
    "규칙": "금지 조건이 포함되어 있는가?",
    "스타일": "스타일 지시가 포함되어 있는가?",
    "제약사항": "분량/시간 등의 제약 조건이 있는가?",
    "형식/구조": "출력 형식이 포함되어 있는가?",
    "예시": "예시가 포함되어 있는가?",
    "프롬프트 테크닉": "few-shot 등의 고급 기법이 사용되었는가?",
}

# 평가 요청
@st.cache_data(show_spinner=False)
def evaluate_prompt(prompt: str) -> str:
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "너는 교사처럼 프롬프트를 평가하는 역할을 맡았어."},
                {"role": "user", "content": f"다음은 학생 프롬프트입니다:\n{prompt}\n위 프롬프트를 다음 10가지 항목으로 0/1 JSON으로 평가하고 마지막에 피드백 문장 1~2개 추가해주세요:\n{', '.join(CHECKLIST.keys())}"}
            ],
            temperature=0
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"❗ 평가 실패 ({str(e).split(':')[0]})"

# 안전한 평가 with retry
def safe_evaluate(prompt: str, delay: float = 10.0) -> str:
    for attempt in range(3):
        result = evaluate_prompt(prompt)
        if "RateLimit" in result:
            st.warning(f"RateLimitError 발생 - {int(delay)}초 대기 후 재시도합니다...")
            time.sleep(delay * (attempt + 1))
        else:
            return result
    return result

# 멀티 스레드 평가 실행
def evaluate_batch(prompts: list[str], max_workers: int = 2) -> list[str]:
    results = [None] * len(prompts)

    def run(index: int):
        results[index] = safe_evaluate(prompts[index])

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for i in range(len(prompts)):
            executor.submit(run, i)
    return results

# Streamlit UI
st.title("🧠 프롬프트 자동 채점 WebApp")

uploaded_file = st.file_uploader("📄 엑셀 파일 업로드 (.xlsx)", type=["xlsx"])

if uploaded_file is not None:
    df = pd.read_excel(uploaded_file)
    if "프롬프트" not in df.columns:
        st.error("⚠️ '프롬프트'라는 열이 필요합니다.")
        st.stop()

    df = df.astype(str)
    prompts = df["프롬프트"].fillna("").tolist()

    st.success(f"{len(prompts)}개의 프롬프트를 불러왔습니다.")
    max_threads = st.slider("🧵 동시에 평가할 최대 쓰레드 수", 1, 10, 2)

    if st.button("📊 평가 시작"):
        with st.spinner("평가 진행 중..."):
            results = evaluate_batch(prompts, max_threads)
        df['평가결과'] = results
        st.dataframe(df)

        st.download_button(
            "📥 평가 결과 다운로드 (CSV)",
            data=df.to_csv(index=False).encode('utf-8-sig'),
            file_name="프롬프트_평가결과.csv",
            mime='text/csv'
        )

