import streamlit as st
import pandas as pd
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
from openai import RateLimitError
import hashlib

# API 키 설정
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# 체크리스트 정의
CHECKLIST = {
    "역할": "프롬프트에 역할(예: 너는 선생님이다)이 명시되어 있는가?",
    "대상": "프롬프트에 대상(예: 중학생에게 설명해줘)이 명시되어 있는가?",
    "정보": "배경 정보 또는 설명이 포함되어 있는가?",
    "작업": "명확한 작업(예: 요약해줘, 표로 정리해줘)이 명시되어 있는가?",
    "규칙": "하지 말아야 할 금지 조건이 포함되어 있는가?",
    "스타일": "어조, 톤, 스타일 지시가 포함되어 있는가?",
    "제약사항": "분량, 시간 등의 제약 조건이 명시되어 있는가?",
    "형식/구조": "JSON, 표, 목록 등의 출력 형식이 포함되어 있는가?",
    "예시": "예시 또는 샘플이 포함되어 있는가?",
    "프롬프트 테크닉": "few-shot, chain-of-thought 등의 고급 기법이 사용되었는가?",
}

# 간단한 캐시 시스템
CACHE = {}

def prompt_hash(prompt):
    return hashlib.sha256(prompt.encode()).hexdigest()

# 평가 함수 정의
def evaluate_prompt(prompt):
    criteria_prompt = f"""
다음은 학생이 작성한 AI 프롬프트입니다:
{prompt}

이 프롬프트를 아래의 10가지 항목에 따라 0(아니다)/1(그렇다)로 평가해주세요.

{', '.join(CHECKLIST.keys())}

답변은 다음 형식의 JSON으로 출력해주세요:
{{
  "역할": 0 또는 1,
  "대상": 0 또는 1,
  ... 생략 ...
}}
그리고 마지막에 학생에게 줄 1~2문장 피드백을 써주세요.
"""

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "너는 교사처럼 프롬프트를 평가하는 역할을 맡았어."},
            {"role": "user", "content": criteria_prompt}
        ],
        temperature=0
    )

    return response.choices[0].message.content

# 안전한 평가 함수 + 캐시 + 재시도 포함
def safe_evaluate(prompt):
    h = prompt_hash(prompt)
    if h in CACHE:
        return CACHE[h]
    
    for wait in [10, 20, 30]:
        try:
            result = evaluate_prompt(prompt)
            CACHE[h] = result
            return result
        except RateLimitError:
            time.sleep(wait)
    return "❗ 평가 실패 (RateLimit)"

# 병렬 처리 평가 함수
def parallel_evaluate(prompts, max_threads=5):
    results = [None] * len(prompts)
    with ThreadPoolExecutor(max_workers=max_threads) as executor:
        future_to_index = {executor.submit(safe_evaluate, prompt): i for i, prompt in enumerate(prompts)}
        for future in as_completed(future_to_index):
            idx = future_to_index[future]
            try:
                results[idx] = future.result()
            except Exception as e:
                results[idx] = f"오류: {str(e)}"
    return results

# Streamlit UI 시작
st.title("⚡ 빠른 프롬프트 자동 채점기 (GPT-3.5 병렬)")

uploaded_file = st.file_uploader("📄 프롬프트 Excel 업로드 (.xlsx)", type=["xlsx"])

if uploaded_file is not None:
    df = pd.read_excel(uploaded_file)
    if "프롬프트" not in df.columns:
        st.error("❌ '프롬프트' 열이 필요합니다.")
    else:
        st.success(f"{len(df)}개의 프롬프트를 불러왔습니다.")
        max_threads = st.slider("🔀 동시에 평가할 최대 쓰레드 수", 1, 10, 5)
        prompts = df["프롬프트"].tolist()

        with st.spinner("⏳ 프롬프트 평가 중..."):
            results = parallel_evaluate(prompts, max_threads=max_threads)

        df['평가결과'] = results
        st.dataframe(df)

        st.download_button("📥 평가 결과 다운로드 (CSV)",
                           data=df.to_csv(index=False).encode('utf-8-sig'),
                           file_name="프롬프트_평가결과.csv",
                           mime='text/csv')

