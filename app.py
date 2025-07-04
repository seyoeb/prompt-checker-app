import streamlit as st
import pandas as pd
from openai import OpenAI

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
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "너는 교사처럼 프롬프트를 평가하는 역할을 맡았어."},
            {"role": "user", "content": criteria_prompt}
        ],
        temperature=0
    )

    return response.choices[0].message.content

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
        for i, row in df.iterrows():
            prompt = row['프롬프트']
            with st.spinner(f"{i+1}번 프롬프트 평가 중..."):
                evaluation = evaluate_prompt(prompt)
                results.append(evaluation)

        df['평가결과'] = results
        st.dataframe(df)

        st.download_button("📥 평가 결과 다운로드 (CSV)",
                           data=df.to_csv(index=False).encode('utf-8-sig'),
                           file_name="프롬프트_평가결과.csv",
                           mime='text/csv')
