import streamlit as st
import ccxt
import yfinance as yf
import pandas as pd
import numpy as np
import joblib
import ta
import plotly.graph_objects as go
import os
import datetime

# ==========================================
# 1. ver_2 모델 로드 함수
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

@st.cache_resource
def load_ver2_model():
    model_path = os.path.join(BASE_DIR, 'xgboost_btc_15m_v2_advanced.pkl')
    if not os.path.exists(model_path):
        model_path = os.path.join(BASE_DIR, 'ver_2', 'models', 'xgboost_btc_15m_v2_advanced.pkl')
    return joblib.load(model_path)

# ==========================================
# 2. 데이터 수집 (바이낸스 선물 우선, 실패시 yfinance 백업) 및 ver_2 피처 전처리
# ==========================================
@st.cache_data(ttl=900)
def get_candle_data():
    df = None
    
    # 1차 시도: 바이낸스 선물 API (ccxt)
    try:
        exchange = ccxt.binance({'options': {'defaultType': 'future'}, 'timeout': 5000})
        symbol = 'BTC/USDT'
        timeframe = '15m'
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=1500)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        df.index = df.index.tz_localize('UTC').tz_convert('Asia/Seoul').tz_localize(None)
    except Exception as e:
        # Streamlit Cloud 미국 IP 지오블록(ExchangeNotAvailable) 발생 시 yfinance로 자동 백업
        btc = yf.Ticker("BTC-USD")
        df = btc.history(period="60d", interval="15m")
        if df.index.tz is not None:
            df.index = df.index.tz_convert('Asia/Seoul').tz_localize(None)
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']]

    # ver_2 피처 엔지니어링 (정상성 변환, ta 지표, MTF 1시간봉 비율)
    df['Returns'] = df['Close'].pct_change()
    df['Body_Size'] = (df['Close'] - df['Open']) / df['Open']
    df['Upper_Shadow'] = (df['High'] - df[['Open', 'Close']].max(axis=1)) / df['Close']
    df['Lower_Shadow'] = (df[['Open', 'Close']].min(axis=1) - df['Low']) / df['Close']

    df['RSI_14'] = ta.momentum.rsi(df['Close'], window=14) / 100.0  # 0~1 스케일
    df['ATR_14'] = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'], window=14)
    df['ATR_Ratio'] = df['ATR_14'] / df['Close']

    sma_20 = ta.trend.sma_indicator(df['Close'], window=20)
    df['Close_vs_SMA20'] = df['Close'] / sma_20 - 1

    bb_high = ta.volatility.bollinger_hband(df['Close'], window=20)
    bb_low = ta.volatility.bollinger_lband(df['Close'], window=20)
    df['BB_Width'] = (bb_high - bb_low) / sma_20
    df['BB_Pos'] = (df['Close'] - bb_low) / (bb_high - bb_low + 1e-8)

    # MTF 1시간봉 지표
    df_1h = df['Close'].resample('1h').last().to_frame(name='Close_1H')
    sma_20_1h = ta.trend.sma_indicator(df_1h['Close_1H'], window=20)
    df_1h['Close_vs_SMA20_1H'] = df_1h['Close_1H'] / sma_20_1h - 1

    df = df.join(df_1h[['Close_vs_SMA20_1H']], how='left')
    df['Close_vs_SMA20_1H'] = df['Close_vs_SMA20_1H'].ffill()

    df.dropna(inplace=True)
    return df

# ==========================================
# 3. 백테스트 시뮬레이션 로직
# ==========================================
def run_backtest(df, entry_th, exit_th, leverage, invest_ratio, use_rsi_exit, rsi_long_th=90, rsi_short_th=10):
    balance = 10000.0
    position = 0
    avg_entry_price = 0.0
    invested_margin = 0.0
    position_size = 0.0
    fee_rate = 0.0004  # 바이낸스 선물 수수료 (0.04% Taker)

    balance_history = []
    trades = []

    for i in range(len(df)):
        close_price = df['Close'].iloc[i]
        rsi = df['RSI_14'].iloc[i] * 100.0  # 표시용 0~100 스케일
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

            if net_profit <= -invested_margin:
                trades.append({'date': date, 'type': '마진콜 청산', 'price': close_price, 'profit': -invested_margin})
                balance -= invested_margin
                position, invested_margin, position_size = 0, 0, 0
                balance_history.append(balance)
                continue

            if use_rsi_exit:
                if (position == 1 and rsi >= rsi_long_th) or (position == -1 and rsi <= rsi_short_th):
                    trades.append({'date': date, 'type': 'RSI 초과 포지션 종료', 'price': close_price, 'profit': net_profit})
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
                    trades.append({'date': date, 'type': 'Long 진입', 'price': close_price, 'profit': 0.0})
                elif pred == 0:
                    position = -1
                    avg_entry_price, invested_margin = close_price, balance * invest_ratio
                    position_size = invested_margin * leverage
                    trades.append({'date': date, 'type': 'Short 진입', 'price': close_price, 'profit': 0.0})
        else:
            if (position == 1 and pred == 0) or (position == -1 and pred == 2):
                if prob >= exit_th:
                    trades.append({'date': date, 'type': '신호 포지션 종료', 'price': close_price, 'profit': net_profit})
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
                    trades.append({'date': date, 'type': '물타기', 'price': close_price, 'profit': 0.0})

        balance_history.append(max(balance + (net_profit if position != 0 else 0), 0))

    return balance_history, trades

# ==========================================
# 4. Streamlit UI 구성
# ==========================================
st.set_page_config(layout="wide", page_title="BTC ver_2 AI 시뮬레이터", page_icon="📈")
st.title("📈 비트코인 ver_2 AI 선물 시뮬레이터")

model = load_ver2_model()
raw_df = get_candle_data()

st.sidebar.header("📅 투자 기간 설정")
min_date = raw_df.index.min().date()
max_date = raw_df.index.max().date()
start_date = st.sidebar.date_input("시뮬레이션 시작일", min_value=min_date, max_value=max_date, value=min_date)

st.sidebar.header("⚙️ ver_2 매매 파라미터")
entry_th = st.sidebar.slider("진입 임계점 (Entry Threshold)", min_value=0.3, max_value=0.9, value=0.45, step=0.01)
exit_th = st.sidebar.slider("청산 임계점 (Exit Threshold)", min_value=0.3, max_value=0.9, value=0.45, step=0.01)

st.sidebar.markdown("---")
leverage = st.sidebar.slider("레버리지 (Leverage)", 1, 50, 25)
invest_ratio = st.sidebar.slider("1회 진입 비중 (%)", 1, 50, 25) / 100.0

st.sidebar.markdown("---")
use_rsi_exit = st.sidebar.checkbox("RSI 초과 포지션 종료 적용", value=True)
if use_rsi_exit:
    rsi_long_th = st.sidebar.slider("RSI 롱(Long) 청산 수치", min_value=50, max_value=95, value=90, step=1)
    rsi_short_th = st.sidebar.slider("RSI 숏(Short) 청산 수치", min_value=5, max_value=50, value=10, step=1)
else:
    rsi_long_th, rsi_short_th = 90, 10

df = raw_df[raw_df.index >= pd.to_datetime(start_date)]

if df.empty:
    st.error("선택한 날짜 이후의 데이터가 없습니다. 시작일을 더 과거로 조정해주세요.")
else:
    features = ['Returns', 'Body_Size', 'Upper_Shadow', 'Lower_Shadow', 
                'RSI_14', 'ATR_Ratio', 'Close_vs_SMA20', 'BB_Width', 'BB_Pos', 'Close_vs_SMA20_1H']
    X = df[features]
    probs = model.predict_proba(X)
    df = df.copy()
    df['Max_Prob'] = np.max(probs, axis=1)
    df['Pred'] = np.argmax(probs, axis=1)

    hist, trades = run_backtest(df, entry_th, exit_th, leverage, invest_ratio, use_rsi_exit, rsi_long_th, rsi_short_th)
    df['Balance'] = hist

    col1, col2, col3 = st.columns(3)
    col1.metric("초기 자본금", "$10,000.00")
    col2.metric("최종 자산", f"${hist[-1]:,.2f}", f"{(hist[-1]/10000 - 1)*100:.2f}%")
    col3.metric("총 거래 횟수", f"{len([t for t in trades if '진입' in t['type']])} 회 진입")

    st.subheader("💰 백테스트 누적 자산 변화 (ver_2 모델)")
    fig_bal = go.Figure()
    fig_bal.add_trace(go.Scatter(x=df.index, y=df['Balance'], mode='lines', name='포트폴리오 가치', line=dict(color='cyan', width=2)))
    fig_bal.add_hline(y=10000, line_dash="dash", line_color="gray")
    fig_bal.update_layout(template='plotly_dark', height=400, xaxis_title="Date", yaxis_title="Balance (USD)", dragmode='pan', hovermode='x unified')
    st.plotly_chart(fig_bal, use_container_width=True, config={'scrollZoom': True})

    st.subheader("📈 15분봉 및 ver_2 진입/청산 타점 시각화")
    fig_candle = go.Figure(data=[go.Candlestick(
        x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='BTC Price',
        increasing_line_color='green', decreasing_line_color='red'
    )])

    margin = (df['High'].max() - df['Low'].min()) * 0.02
    long_entries = [t for t in trades if t['type'] == 'Long 진입']
    short_entries = [t for t in trades if t['type'] == 'Short 진입']
    add_margins = [t for t in trades if t['type'] == '물타기']
    model_exits = [t for t in trades if t['type'] == '신호 포지션 종료']
    rsi_exits = [t for t in trades if t['type'] == 'RSI 초과 포지션 종료']
    liquidations = [t for t in trades if t['type'] == '마진콜 청산']

    if long_entries:
        fig_candle.add_trace(go.Scatter(x=[t['date'] for t in long_entries], y=[t['price'] - margin for t in long_entries],
                                        mode='markers', marker=dict(symbol='triangle-up', size=12, color='lime', line=dict(width=1, color='darkgreen')), name='Long 진입'))
    if short_entries:
        fig_candle.add_trace(go.Scatter(x=[t['date'] for t in short_entries], y=[t['price'] + margin for t in short_entries],
                                        mode='markers', marker=dict(symbol='triangle-down', size=12, color='red', line=dict(width=1, color='darkred')), name='Short 진입'))
    if add_margins:
        fig_candle.add_trace(go.Scatter(x=[t['date'] for t in add_margins], y=[t['price'] for t in add_margins],
                                        mode='markers', marker=dict(symbol='star', size=10, color='blue'), name='물타기'))
    if model_exits:
        fig_candle.add_trace(go.Scatter(x=[t['date'] for t in model_exits], y=[t['price'] for t in model_exits],
                                        mode='markers', marker=dict(symbol='x', size=10, color='yellow'), name='신호 포지션 종료'))
    if rsi_exits:
        fig_candle.add_trace(go.Scatter(x=[t['date'] for t in rsi_exits], y=[t['price'] for t in rsi_exits],
                                        mode='markers', marker=dict(symbol='x', size=12, color='orange'), name='RSI 초과 포지션 종료'))
    if liquidations:
        fig_candle.add_trace(go.Scatter(x=[t['date'] for t in liquidations], y=[t['price'] for t in liquidations],
                                        mode='markers', marker=dict(symbol='x', size=14, color='purple'), name='마진콜 강제청산'))

    fig_candle.update_layout(template='plotly_dark', height=600, xaxis_rangeslider_visible=False, yaxis_title="Price (USD)", dragmode='pan', hovermode='x unified')
    st.plotly_chart(fig_candle, use_container_width=True, config={'scrollZoom': True})

    st.subheader("📝 ver_2 상세 매매 일지")
    if trades:
        trades_df = pd.DataFrame(trades)
        trades_df.columns = ['시간', '구분', '체결가(USD)', '수익금(USD)']
        trades_df['시간'] = pd.to_datetime(trades_df['시간']).dt.strftime('%Y-%m-%d %H:%M')
        trades_df['체결가(USD)'] = trades_df['체결가(USD)'].apply(lambda x: f"${x:,.2f}")
        trades_df['수익금(USD)'] = trades_df['수익금(USD)'].apply(lambda x: f"${x:,.2f}" if x != 0 else "-")

        st.dataframe(trades_df, use_container_width=True, hide_index=True)
    else:
        st.info("해당 기간 동안 발생한 매매 내역이 없습니다.")
