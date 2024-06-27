import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.api import VAR
import os

# 현재 파일의 디렉토리를 기준으로 데이터 디렉토리 설정
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')

# 1. 사용자로부터 CSV 파일을 업로드 받기 또는 내장된 데이터 사용
st.title("FAB CCI 예측(VAR 모델)")

data_choice = st.radio("데이터 선택 방법을 고르세요", ("내장된 데이터 사용", "CSV 파일 업로드"))

if data_choice == "CSV 파일 업로드":
    uploaded_file = st.file_uploader("CSV 파일을 업로드해 주세요", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
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
    selected_option = st.selectbox("내장된 데이터 중 하나를 선택해 주세요", options.keys())
    proceed = st.button("예측하기")
    if proceed:
        file_number = options[selected_option]
        file_path = os.path.join(DATA_DIR, f'tar{file_number}.csv')
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
        else:
            st.error(f"파일을 찾을 수 없습니다: {file_path}")
            proceed = False

if proceed and 'df' in locals():
    # 2. 파일을 읽어서 데이터프레임으로 변환
    st.write("사용할 데이터:")
    st.write(df)
    
    st.divider()

    # 4. date 컬럼을 datetime 형식으로 변환
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)

    # 5. AIC를 사용하여 VAR 모델의 최적 lag 값을 찾기
    model = VAR(df)
    results = model.select_order(maxlags=15)
    lag_order = results.aic
    st.write(f"최적 Lag (AIC 기반): {lag_order}")

    st.divider()

    # 6. 해당 lag를 사용해서 모델을 학습
    var_model = model.fit(lag_order)

    # 7. 마지막 컬럼의 향후 15개월 값을 예측
    forecast_steps = 15
    forecast = var_model.forecast(df.values[-lag_order:], steps=forecast_steps)
    forecast_index = pd.date_range(start=(df.index[-1] + pd.offsets.MonthBegin()).replace(day=1), periods=forecast_steps, freq='MS')
    forecast_df = pd.DataFrame(forecast, index=forecast_index, columns=df.columns)

    st.divider()

    # 8. 예측 값을 date와 함께 사용자에게 보여줌
    st.write("예측값:")
    st.write(forecast_df[[df.columns[-1]]])

    st.divider()

    # 9. 마지막 컬럼의 실제값과 예측값을 plot
    plt.figure(figsize=(10, 6))
    plt.plot(df.index, df.iloc[:, -1], label='Actual', color='grey')
    plt.plot(forecast_df.index, forecast_df.iloc[:, -1], label='Predicted', color='red', linestyle='dashed')
    plt.xlabel('time')
    plt.ylabel(df.columns[-1])
    plt.title('Actual and Predicted Values')
    plt.legend()
    st.pyplot(plt)

    st.divider()

    # 10. 예측값을 CSV 파일로 다운로드할 수 있게 만듬
    csv = forecast_df[[df.columns[-1]]].to_csv().encode('utf-8')
    st.download_button(label="예측값 다운로드(.csv)", data=csv, file_name='forecast.csv', mime='text/csv')



