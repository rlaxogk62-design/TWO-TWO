import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import os
import datetime

# ==========================================
# 1. 모델 로드 함수
# ==========================================
@st.cache_resource
def load_model():
    model_path = 'xgboost_btc_15m_3class_strict.pkl'
    if not os.path.exists(model_path):
        model_path = '/content/xgboost_btc_15m_3class_strict.pkl'
    if not os.path.exists(model_path):
        model_path = './data/model/xgboost_btc_15m_3class_strict.pkl'
    return joblib.load(model_path)

# ==========================================
# 2. 데이터 수집 및 전처리
# ==========================================
@st.cache_data(ttl=900)
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
# 3. 백테스트 로직 (타점 기록 추가)
# ==========================================
def run_backtest(df, entry_th, exit_th, leverage, invest_ratio, use_rsi_exit):
    balance = 10000.0
    position = 0
    avg_entry_price = 0.0
    invested_margin = 0.0
    position_size = 0.0
    fee_rate = 0.0004
    
    balance_history = []
    trades = [] # 진입/청산 기록 저장
    
    for i in range(len(df)):
        close_price = df['Close'].iloc[i]
        rsi = df['RSI_14'].iloc[i]
        prob = df['Max_Prob'].iloc[i]
        pred = df['Pred'].iloc[i]
        date = df.index[i]
        
        if balance <= 0:
            balance_history.append(0)
            continue
            
        net_profit = 0
        if position != 0:
            price_change_pct = (close_price - avg_entry_price) / avg_entry_price * position
            net_profit = (position_size * price_change_pct) - (position_size * fee_rate * 2)
            
            # 강제 청산 조건 (마진 콜)
            if net_profit <= -invested_margin:
                trades.append({'date': date, 'type': 'Liquidated', 'price': close_price})
                balance -= invested_margin
                position, invested_margin, position_size = 0, 0, 0
                balance_history.append(balance)
                continue
            
            # RSI 기반 강제 청산
            if use_rsi_exit:
                if (position == 1 and rsi >= 70) or (position == -1 and rsi <= 30):
                    trades.append({'date': date, 'type': 'Exit', 'price': close_price})
                    balance += net_profit
                    position, invested_margin, position_size = 0, 0, 0
                    balance_history.append(max(balance, 0))
                    continue
                
        is_loss = (position == 1 and close_price < avg_entry_price) or (position == -1 and close_price > avg_entry_price)
        
        if position == 0:
            if prob >= entry_th:
                if pred == 2:
                    position = 1
                    avg_entry_price, invested_margin = close_price, balance * invest_ratio
                    position_size = invested_margin * leverage
                    trades.append({'date': date, 'type': 'Long Entry', 'price': close_price})
                elif pred == 0:
                    position = -1
                    avg_entry_price, invested_margin = close_price, balance * invest_ratio
                    position_size = invested_margin * leverage
                    trades.append({'date': date, 'type': 'Short Entry', 'price': close_price})
        else:
            if (position == 1 and pred == 0) or (position == -1 and pred == 2):
                if prob >= exit_th:
                    trades.append({'date': date, 'type': 'Exit', 'price': close_price})
                    balance += net_profit
                    position, invested_margin, position_size = 0, 0, 0
            elif (position == 1 and pred == 2) or (position == -1 and pred == 0):
                if prob >= entry_th and is_loss and balance > 0:
                    add_margin = balance * invest_ratio
                    add_size = add_margin * leverage
                    total_size = position_size + add_size
                    avg_entry_price = (position_size * avg_entry_price + add_size * close_price) / total_size
                    invested_margin += add_margin
                    position_size = total_size
                    trades.append({'date': date, 'type': 'Add Margin', 'price': close_price})
                    
        balance_history.append(max(balance + (net_profit if position != 0 else 0), 0))
        
    return balance_history, trades

# ==========================================
# 4. Streamlit UI 구성
# ==========================================
st.set_page_config(layout="wide", page_title="BTC AI Trading Bot")
st.title("비트코인 AI 15분봉 자동매매 시뮬레이터 (차트 & 기간 조절 포함)")

# 데이터 및 모델 로드 (캐싱)
model = load_model()
raw_df = get_data()

st.sidebar.header("📅 투자 기간 설정")
min_date = raw_df.index.min().date()
max_date = raw_df.index.max().date()
start_date = st.sidebar.date_input("모의투자 시작일", min_value=min_date, max_value=max_date, value=min_date)

st.sidebar.header("⚙️ 투자 로직 파라미터")
entry_th = st.sidebar.slider("진입 임계점 (Entry Threshold)", min_value=0.3, max_value=0.9, value=0.45, step=0.01)
exit_th = st.sidebar.slider("청산 임계점 (Exit Threshold)", min_value=0.3, max_value=0.9, value=0.40, step=0.01)

st.sidebar.markdown("---")
leverage = st.sidebar.slider("레버리지 (Leverage)", 1, 50, 10)
invest_ratio = st.sidebar.slider("1회 진입 비중 (%)", 1, 50, 10) / 100.0
use_rsi_exit = st.sidebar.checkbox("RSI 강제청산 적용 (Long >= 70, Short <= 30)", value=True)

# 선택한 날짜에 맞게 데이터 필터링
df = raw_df[raw_df.index >= pd.to_datetime(start_date)]

if df.empty:
    st.error("선택한 날짜 이후의 데이터가 없습니다. 시작일을 더 과거로 조정해주세요.")
else:
    # 예측 수행
    features = ['Open', 'High', 'Low', 'Close', 'Volume', 'SMA_7', 'RSI_14', 'SMA_1H', 'SMA_4H', 'Vol_4H', 'SMA_24H', 'BB_Width']
    X = df[features]
    probs = model.predict_proba(X)
    df = df.copy()
    df['Max_Prob'] = np.max(probs, axis=1)
    df['Pred'] = np.argmax(probs, axis=1)

    # 백테스트 실행 (타점 리스트 받아오기)
    hist, trades = run_backtest(df, entry_th, exit_th, leverage, invest_ratio, use_rsi_exit)
    df['Balance'] = hist

    # 상단 요약
    col1, col2, col3 = st.columns(3)
    col1.metric("초기 자본금", "$10,000.00")
    col2.metric("최종 자산", f"${hist[-1]:,.2f}", f"{(hist[-1]/10000 - 1)*100:.2f}%")
    col3.metric("총 거래 횟수", f"{len([t for t in trades if 'Entry' in t['type']])} 회 진입")

    # 자산 변화 차트
    st.subheader("💰 백테스트 누적 자산 변화")
    fig_bal = go.Figure()
    fig_bal.add_trace(go.Scatter(x=df.index, y=df['Balance'], mode='lines', name='포트폴리오 가치', line=dict(color='cyan', width=2)))
    fig_bal.add_hline(y=10000, line_dash="dash", line_color="gray")
    fig_bal.update_layout(template='plotly_dark', height=400, xaxis_title="Date", yaxis_title="Balance (USD)")
    st.plotly_chart(fig_bal, use_container_width=True)

    # 캔들 및 타점 차트
    st.subheader("📈 비트코인 15분봉 및 진입/청산 타점 시각화")
    fig_candle = go.Figure(data=[go.Candlestick(
        x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='BTC Price',
        increasing_line_color='green', decreasing_line_color='red'
    )])

    # 타점 그리기
    margin = (df['High'].max() - df['Low'].min()) * 0.02
    long_entries = [t for t in trades if t['type'] == 'Long Entry']
    short_entries = [t for t in trades if t['type'] == 'Short Entry']
    exits = [t for t in trades if t['type'] == 'Exit']
    liquidations = [t for t in trades if t['type'] == 'Liquidated']

    if long_entries:
        fig_candle.add_trace(go.Scatter(x=[t['date'] for t in long_entries], y=[t['price'] - margin for t in long_entries],
                                        mode='markers', marker=dict(symbol='triangle-up', size=12, color='lime', line=dict(width=1, color='darkgreen')), name='Long 진입'))
    if short_entries:
        fig_candle.add_trace(go.Scatter(x=[t['date'] for t in short_entries], y=[t['price'] + margin for t in short_entries],
                                        mode='markers', marker=dict(symbol='triangle-down', size=12, color='red', line=dict(width=1, color='darkred')), name='Short 진입'))
    if exits:
        fig_candle.add_trace(go.Scatter(x=[t['date'] for t in exits], y=[t['price'] for t in exits],
                                        mode='markers', marker=dict(symbol='x', size=10, color='yellow'), name='청산(종료)'))
    if liquidations:
        fig_candle.add_trace(go.Scatter(x=[t['date'] for t in liquidations], y=[t['price'] for t in liquidations],
                                        mode='markers', marker=dict(symbol='x', size=14, color='purple'), name='강제청산'))

    fig_candle.update_layout(template='plotly_dark', height=600, xaxis_rangeslider_visible=False, yaxis_title="Price (USD)")
    st.plotly_chart(fig_candle, use_container_width=True)
