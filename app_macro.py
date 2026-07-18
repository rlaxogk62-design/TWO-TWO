import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import os

# ==========================================
# 1. 모델 로드 함수
# ==========================================
@st.cache_resource
def load_model():
    model_path = '/content/xgboost_btc_15m_3class_strict.pkl'
    if not os.path.exists(model_path):
        model_path = './data/model/xgboost_btc_15m_3class_strict.pkl'
    return joblib.load(model_path)

# ==========================================
# 2. 데이터 수집 및 전처리
# ==========================================
@st.cache_data(ttl=900) # 15분마다 캐시 갱신
def get_data():
    btc = yf.Ticker("BTC-USD")
    df = btc.history(period="60d", interval="15m")
    if df.index.tz is not None:
        df.index = df.index.tz_convert('Asia/Seoul').tz_localize(None)
        
    df['Returns'] = df['Close'].pct_change()
    df['SMA_7'] = df['Close'].rolling(window=7).mean()
    
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    df['RSI_14'] = 100 - (100 / (1 + gain / loss))
    
    df['SMA_1H'] = df['Close'].rolling(window=4).mean()
    df['SMA_4H'] = df['Close'].rolling(window=16).mean()
    df['Vol_4H'] = df['Returns'].rolling(window=16).std()
    df['SMA_24H'] = df['Close'].rolling(window=96).mean()
    
    df['BB_Std'] = df['Close'].rolling(window=20).std()
    df['BB_Width'] = (df['BB_Std'] * 4) / df['Close'].rolling(window=20).mean()
    
    df.dropna(inplace=True)
    return df

# ==========================================
# 3. 백테스트 로직 (이중 임계값 & RSI 강제청산)
# ==========================================
def run_backtest(df, entry_th, exit_th, leverage, invest_ratio, use_rsi_exit):
    balance = 10000.0
    position = 0
    avg_entry_price = 0.0
    invested_margin = 0.0
    position_size = 0.0
    fee_rate = 0.0004
    
    balance_history = []
    
    for i in range(len(df)):
        close_price = df['Close'].iloc[i]
        rsi = df['RSI_14'].iloc[i]
        prob = df['Max_Prob'].iloc[i]
        pred = df['Pred'].iloc[i]
        
        if balance <= 0:
            balance_history.append(0)
            continue
            
        net_profit = 0
        if position != 0:
            price_change_pct = (close_price - avg_entry_price) / avg_entry_price * position
            net_profit = (position_size * price_change_pct) - (position_size * fee_rate * 2)
            
            # 강제 청산 조건 (마진 콜)
            if net_profit <= -invested_margin:
                balance -= invested_margin
                position, invested_margin, position_size = 0, 0, 0
                balance_history.append(balance)
                continue
            
            # RSI 기반 과매수/과매도 강제 청산
            if use_rsi_exit:
                if (position == 1 and rsi >= 70) or (position == -1 and rsi <= 30):
                    balance += net_profit
                    position, invested_margin, position_size = 0, 0, 0
                    balance_history.append(max(balance, 0))
                    continue
                
        is_loss = (position == 1 and close_price < avg_entry_price) or (position == -1 and close_price > avg_entry_price)
        
        if position == 0:
            # 포지션 진입 (진입 임계점 기준, 모델 예측만 고려)
            if prob >= entry_th:
                if pred == 2:
                    position = 1
                    avg_entry_price, invested_margin = close_price, balance * invest_ratio
                    position_size = invested_margin * leverage
                elif pred == 0:
                    position = -1
                    avg_entry_price, invested_margin = close_price, balance * invest_ratio
                    position_size = invested_margin * leverage
        else:
            # 반대 예측 시: 스위칭이 아닌 '익절/손절(청산)'만 수행 (청산 임계점 기준)
            if (position == 1 and pred == 0) or (position == -1 and pred == 2):
                if prob >= exit_th:
                    balance += net_profit
                    position, invested_margin, position_size = 0, 0, 0
            # 동일 예측 시: 물타기 수행 (진입 임계점 기준)
            elif (position == 1 and pred == 2) or (position == -1 and pred == 0):
                if prob >= entry_th and is_loss and balance > 0:
                    add_margin = balance * invest_ratio
                    add_size = add_margin * leverage
                    total_size = position_size + add_size
                    avg_entry_price = (position_size * avg_entry_price + add_size * close_price) / total_size
                    invested_margin += add_margin
                    position_size = total_size
                    
        balance_history.append(max(balance + (net_profit if position != 0 else 0), 0))
        
    return balance_history

# ==========================================
# 4. Streamlit UI 구성
# ==========================================
st.set_page_config(layout="wide", page_title="BTC AI Trading Bot")
st.title("비트코인 AI 15분봉 자동매매 시뮬레이터 (이중 임계값 & RSI 강제청산)")

st.sidebar.header("⚙️ 투자 로직 파라미터")
entry_th = st.sidebar.slider("진입 임계점 (Entry Threshold)", min_value=0.3, max_value=0.9, value=0.45, step=0.01,
                             help="새로운 포지션에 진입하거나 물타기를 할 때 요구되는 모델의 확신도입니다.")
exit_th = st.sidebar.slider("청산 임계점 (Exit Threshold)", min_value=0.3, max_value=0.9, value=0.40, step=0.01,
                            help="보유 중인 포지션과 반대되는 예측이 나왔을 때, 포지션을 종료(익절/손절)하기 위해 요구되는 모델의 확신도입니다.")

st.sidebar.markdown("---")
leverage = st.sidebar.slider("레버리지 (Leverage)", 1, 50, 10)
invest_ratio = st.sidebar.slider("1회 진입 비중 (%)", 1, 50, 10) / 100.0
use_rsi_exit = st.sidebar.checkbox("RSI 강제청산 적용 (Long >= 70, Short <= 30)", value=True, help="보유 포지션 방향에 대해 지표가 과열/과매도에 도달하면 즉각 청산합니다.")

# 데이터 및 모델 로드
model = load_model()
df = get_data()

# 예측 수행
features = ['Open', 'High', 'Low', 'Close', 'Volume', 'SMA_7', 'RSI_14', 'SMA_1H', 'SMA_4H', 'Vol_4H', 'SMA_24H', 'BB_Width']
X = df[features]
probs = model.predict_proba(X)
df['Max_Prob'] = np.max(probs, axis=1)
df['Pred'] = np.argmax(probs, axis=1)

# 백테스트 실행
hist = run_backtest(df, entry_th, exit_th, leverage, invest_ratio, use_rsi_exit)
df['Balance'] = hist

# 결과 시각화
col1, col2, col3 = st.columns(3)
col1.metric("초기 자본금", "$10,000.00")
col2.metric("최종 자산", f"${hist[-1]:,.2f}", f"{(hist[-1]/10000 - 1)*100:.2f}%")

fig = go.Figure()
fig.add_trace(go.Scatter(x=df.index, y=df['Balance'], mode='lines', name='포트폴리오 가치', line=dict(color='cyan', width=2)))
fig.add_hline(y=10000, line_dash="dash", line_color="gray")
fig.update_layout(
    title="백테스트 누적 자산 변화",
    yaxis_title="Balance (USD)",
    xaxis_title="Date",
    template='plotly_dark',
    height=500
)
st.plotly_chart(fig, use_container_width=True)
