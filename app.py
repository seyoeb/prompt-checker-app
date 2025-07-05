import streamlit as st
import pandas as pd
import time
import json
from openai import OpenAI
from openai import RateLimitError

# API 키 불러오기
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# 체크리스트 항목 정의
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

# 안전한 평가 함수 (자동 재시도 포함)
def safe_evaluate_batch(prompts, retries=3):
    for attempt in range(retries):
        try:
            return evaluate_prompt_batch(prompts)
        except RateLimitError:
            st.warning(f"RateLimitError 발생 – {10 * (attempt + 1)}초 대기 후 재시도합니다...")
            time.sleep(10 * (attempt + 1))
    return ["❗ 평가 실패 (RateLimit)"] * len(prompts)

# 평가 함수 (배치 처리)
def evaluate_prompt_batch(prompts):
    joined = "\n\n".join([
        f"[{i+1}] {prompt}" for i, prompt in enumerate(prompts)
    ])

    criteria_prompt = f"""
다음은 학생들이 작성한 AI 프롬프트입니다:
{joined}

각 프롬프트를 아래의 10가지 항목에 따라 0(아니다)/1(그렇다)로 평가해주세요.

{', '.join(CHECKLIST.keys())}

답변은 JSON 형식의 리스트로 출력해주세요. 예:
[
  {{ "역할": 1, "대상": 0, ..., "피드백": "..." }},
  {{ "역할": 0, "대상": 1, ..., "피드백": "..." }}
]
"""

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "너는 교사처럼 프롬프트를 평가하는 역할을 맡았어."},
            {"role": "user", "content": criteria_prompt}
        ],
        temperature=0
    )

    output = response.choices[0].message.content
    return json.loads(output)

# Streamlit UI
st.title("🧠 프롬프트 자동 채점 WebApp")
st.markdown("""
**설명**: 아래에서 학생들의 프롬프트 Excel 파일을 업로드하면, 각 프롬프트를 체크리스트 기반으로 자동 평가합니다.  
[체크리스트 기준: 역할, 대상, 정보, 작업, 규칙, 스타일, 제약조건, 형식/구조, 예시, 프롬프트 테크닉]
""")

uploaded_file = st.file_uploader("📄 엑셀 파일 업로드 (.xlsx)", type=["xlsx"])

if uploaded_file is not None:
    df = pd.read_excel(uploaded_file)
    st.success(f"{len(df)}개의 프롬프트를 불러왔습니다.")

    if "프롬프트" not in df.columns:
        st.error("⚠️ '프롬프트'라는 열이 필요합니다. 엑셀 파일에 '프롬프트' 열이 있는지 확인해주세요.")
    else:
        results = []
        BATCH_SIZE = st.number_input("🔢 한 번에 평가할 프롬프트 수", min_value=1, max_value=10, value=5)
        WAIT_SECONDS = st.slider("⏱️ 평가 간 대기 시간 (초)", min_value=0, max_value=60, value=10)

        for start in range(0, len(df), BATCH_SIZE):
            batch = df.iloc[start:start+BATCH_SIZE]
            prompts = batch['프롬프트'].tolist()
            with st.spinner(f"{start+1}~{start+len(batch)}번 프롬프트 평가 중..."):
                evaluations = safe_evaluate_batch(prompts)
                results.extend(evaluations)
            time.sleep(WAIT_SECONDS)

        for key in CHECKLIST.keys():
            df[key] = [e.get(key, None) for e in results]
        df['피드백'] = [e.get("피드백", "") for e in results]

        st.dataframe(df)

        st.download_button("📥 평가 결과 다운로드 (CSV)",
                           data=df.to_csv(index=False).encode('utf-8-sig'),
                           file_name="프롬프트_평가결과.csv",
                           mime='text/csv')

