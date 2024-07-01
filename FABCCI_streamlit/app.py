import streamlit as st
import pandas as pd
import os
import json
from statsmodels.tsa.api import VAR

# 현재 파일의 디렉토리를 기준으로 데이터 디렉토리 설정
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
SAMPLES_DIR = os.path.join(BASE_DIR, 'samples')
FACTOR_MAP_PATH = os.path.join(BASE_DIR, 'factor_map.json')
LOGO_PATH_SNUCEM = os.path.join(BASE_DIR, 'img/logo_snucem.png')
LOGO_PATH_HG = os.path.join(BASE_DIR, 'img/logo_hg.png')

# factor_map.json 파일 읽기
if os.path.exists(FACTOR_MAP_PATH):
    with open(FACTOR_MAP_PATH, 'r', encoding='cp949') as f:
        factor_map = json.load(f)
else:
    factor_map = {}

# 컬럼명 변환 함수
def rename_columns(df, factor_map):
    df = df.rename(columns=lambda x: factor_map.get(x, x))
    return df

# 1. 사용자로부터 CSV 파일을 업로드 받기 또는 내장된 데이터 사용
st.title("FAB 건설공사비지수 예측")
st.divider()
st.subheader("데이터 업로드 방식을 선택하세요", divider='grey')
data_choice = st.radio('데이터베이스에 내장된 데이터를 활용하거나, 새로운 데이터를 CSV 파일로 업로드할 수 있습니다',("내장된 데이터 사용 (최신: 2000.01. ~ 2024.04.)", "새로운 CSV 파일 업로드"))

if data_choice == "새로운 CSV 파일 업로드":
    # 샘플 파일 다운로드 버튼 생성
    st.divider()
    st.subheader("샘플 파일 다운로드", divider='grey')
    st.write("주의!: 예측하고자 하는 값은 꼭 마지막 컬럼으로 지정하여 작성해주세요")
    sample_files = [f for f in os.listdir(SAMPLES_DIR) if f.endswith('.csv')]
    for sample_file in sample_files:
        file_path = os.path.join(SAMPLES_DIR, sample_file)
        with open(file_path, 'rb') as f:
            st.download_button(label=sample_file, data=f, file_name=sample_file, mime='text/csv')

    st.divider()
    st.subheader("CSV 파일을 업로드해 주세요", divider='grey')
    uploaded_file = st.file_uploader("파일 드래그 및 파일 찾기로 업로드해 주세요", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file, encoding='cp949')
        df = rename_columns(df, factor_map)
        proceed = st.button("예측하기")
    else:
        proceed = False
else:
    options = {
        "통합": 1,
        "재료": 2,
        "노무": 3,
        "골조": 4,
        "마감": 5,
        "건축마감": 6,
        "설비마감": 7,
        "전기마감": 8
    }
    st.divider()
    st.subheader("예측하고자 하는 지수를 선택해 주세요", divider='grey')
    selected_option = st.selectbox("", options.keys())
    proceed = st.button("예측하기")
    if proceed:
        file_number = options[selected_option]
        file_path = os.path.join(DATA_DIR, f'tar{file_number}.csv')
        if os.path.exists(file_path):
            df = pd.read_csv(file_path, encoding='cp949')
            df = rename_columns(df, factor_map)
        else:
            st.error(f"파일을 찾을 수 없습니다: {file_path}")
            proceed = False

if proceed and 'df' in locals():
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)

    st.divider()
    st.subheader("업로드된 데이터", divider='grey')
    st.write(df)
    
    # 업로드된 데이터들 plot
    st.divider()
    st.subheader("시장 지수 및 실제 지수", divider='grey')
    st.line_chart(df)

    # AIC를 사용하여 VAR 모델의 최적 lag 값을 찾기
    model = VAR(df)
    results = model.select_order(maxlags=15)
    lag_order = results.aic

    st.divider()
    st.subheader("파라미터 최적화 결과", divider='grey')
    st.write(f"최적 지연(lag)= {lag_order}개월")

    # 해당 lag를 사용해서 모델을 학습
    var_model = model.fit(lag_order)

    # 마지막 컬럼의 향후 15개월 값을 예측
    forecast_steps = 15
    forecast = var_model.forecast(df.values[-lag_order:], steps=forecast_steps)
    forecast_index = pd.date_range(start=(df.index[-1] + pd.offsets.MonthBegin()).replace(day=1), periods=forecast_steps, freq='MS')
    forecast_df = pd.DataFrame(forecast, index=forecast_index, columns=df.columns)

    # 예측 값을 date와 함께 사용자에게 보여줌
    st.divider()
    st.subheader(f"{df.columns[-1]} 예측값", divider='grey')
    forecast_df_renamed = rename_columns(forecast_df[[df.columns[-1]]], factor_map)
    st.write(forecast_df_renamed)

    # 마지막 컬럼의 실제값과 예측값을 plot
    st.divider()
    st.subheader("실제값 및 예측값 비교 그래프", divider='grey')
    actual_pred_df = pd.concat([df.iloc[:, -1], forecast_df.iloc[:, -1]], axis=1)
    actual_pred_df.columns = ['Actual', 'Predicted']
    st.line_chart(actual_pred_df)

    # 예측값을 CSV 파일로 다운로드할 수 있게 만듬
    st.divider()
    st.subheader("예측값 다운로드", divider='grey')
    csv = forecast_df_renamed.to_csv().encode('cp949')
    st.download_button(label="예측값 다운로드(.csv)", data=csv, file_name=f'{df.columns[-1]}_예측값.csv', mime='text/csv')

# 두 이미지를 양 옆에 나란히 표시
st.divider()
col1, col2 = st.columns(2)
with col1:
    st.image(LOGO_PATH_SNUCEM, width=200)
with col2:
    st.image(LOGO_PATH_HG, width=200)



