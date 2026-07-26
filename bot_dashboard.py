import streamlit as st
import ccxt
import requests
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os
import joblib
import ta
from dotenv import load_dotenv

# 페이지 설정
st.set_page_config(layout="wide", page_title="BTC ver_2 AI 대시보드 및 시뮬레이터", page_icon="🤖")

# API 로드
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, '.env'))
API_KEY = os.getenv('BINANCE_API_KEY')
SECRET_KEY = os.getenv('BINANCE_SECRET_KEY')

# ver_2 모델 로드
@st.cache_resource
def load_ver2_model():
    model_path = os.path.join(BASE_DIR, 'xgboost_btc_15m_v2_advanced.pkl')
    if not os.path.exists(model_path):
        model_path = os.path.join(BASE_DIR, 'ver_2', 'models', 'xgboost_btc_15m_v2_advanced.pkl')
    return joblib.load(model_path)

model = load_ver2_model()

# 거래소 객체 초기화
@st.cache_resource
def get_exchange():
    if API_KEY and SECRET_KEY:
        return ccxt.binanceusdm({
            'apiKey': API_KEY,
            'secret': SECRET_KEY,
            'enableRateLimit': True,
            'options': {
                'defaultType': 'future',
                'warnOnFetchBalance': False
            }
        })
    return None

exchange = get_exchange()
SYMBOL = 'BTC/USDT'

# 도쿄 VPS API 또는 yfinance 백업 캔들 수집
@st.cache_data(ttl=300)
def get_candle_data():
    df = None
    try:
        url = 'http://149.28.23.225:5000/klines?symbol=BTC/USDT&timeframe=15m&limit=1500'
        res = requests.get(url, timeout=5)
        if res.status_code == 200:
            data = res.json()
            if data.get('status') == 'success':
                ohlcv = data['data']
                df = pd.DataFrame(ohlcv, columns=['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df.set_index('timestamp', inplace=True)
                df.index = df.index.tz_localize('UTC').tz_convert('Asia/Seoul').tz_localize(None)
    except Exception:
        df = None

    if df is None or df.empty:
        btc = yf.Ticker("BTC-USD")
        df = btc.history(period="60d", interval="15m")
        if df.index.tz is not None:
            df.index = df.index.tz_convert('Asia/Seoul').tz_localize(None)
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']]

    # ver_2 피처 엔지니어링
    df['Returns'] = df['Close'].pct_change()
    df['Body_Size'] = (df['Close'] - df['Open']) / df['Open']
    df['Upper_Shadow'] = (df['High'] - df[['Open', 'Close']].max(axis=1)) / df['Close']
    df['Lower_Shadow'] = (df[['Open', 'Close']].min(axis=1) - df['Low']) / df['Close']

    df['RSI_14'] = ta.momentum.rsi(df['Close'], window=14) / 100.0
    df['ATR_14'] = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'], window=14)
    df['ATR_Ratio'] = df['ATR_14'] / df['Close']

    sma_20 = ta.trend.sma_indicator(df['Close'], window=20)
    df['Close_vs_SMA20'] = df['Close'] / sma_20 - 1

    bb_high = ta.volatility.bollinger_hband(df['Close'], window=20)
    bb_low = ta.volatility.bollinger_lband(df['Close'], window=20)
    df['BB_Width'] = (bb_high - bb_low) / sma_20
    df['BB_Pos'] = (df['Close'] - bb_low) / (bb_high - bb_low + 1e-8)

    df_1h = df['Close'].resample('1h').last().to_frame(name='Close_1H')
    sma_20_1h = ta.trend.sma_indicator(df_1h['Close_1H'], window=20)
    df_1h['Close_vs_SMA20_1H'] = df_1h['Close_1H'] / sma_20_1h - 1

    df = df.join(df_1h[['Close_vs_SMA20_1H']], how='left')
    df['Close_vs_SMA20_1H'] = df['Close_vs_SMA20_1H'].ffill()

    df.dropna(inplace=True)
    return df

# 실시간 모니터링 데이터 수집
def fetch_live_monitoring():
    if not exchange:
        return None, None, None
    try:
        account_info = exchange.fapiPrivateV2GetAccount()
        usdt_total = float(account_info.get('totalWalletBalance', 0.0))
        usdt_free = float(account_info.get('availableBalance', 0.0))
        
        pos_data = None
        raw_positions = exchange.fapiPrivateV2GetPositionRisk({'symbol': 'BTCUSDT'})
        for p in raw_positions:
            amt = float(p.get('positionAmt', 0))
            if amt != 0:
                entry_price = float(p.get('entryPrice', 0))
                unrealized_pnl = float(p.get('unRealizedProfit', 0))
                leverage = int(p.get('leverage', 25))
                contracts = abs(amt)
                
                notional_value = contracts * entry_price
                initial_margin = notional_value / leverage if leverage > 0 else 0.0
                pnl_roe = (unrealized_pnl / initial_margin * 100.0) if initial_margin > 0 else 0.0
                liquidation_price = float(p.get('liquidationPrice', 0))

                pos_data = {
                    'symbol': 'BTC/USDT',
                    'side': 'LONG' if amt > 0 else 'SHORT',
                    'contracts': contracts,
                    'entryPrice': entry_price,
                    'unrealizedPnl': unrealized_pnl,
                    'pnlRoe': pnl_roe,
                    'leverage': leverage,
                    'initialMargin': initial_margin,
                    'notionalValue': notional_value,
                    'liquidationPrice': liquidation_price
                }
                break
        return usdt_total, usdt_free, pos_data
    except Exception as e:
        return None, None, None

# 사이드바 메뉴 선택
st.sidebar.title("📌 메뉴 선택")
mode = st.sidebar.radio("모드 전환", ["🤖 실시간 자동매매 모니터링", "📈 ver_2 백테스트 시뮬레이터"])

raw_df = get_candle_data()

if mode == "🤖 실시간 자동매매 모니터링":
    st.title("🤖 바이낸스 ver_2 AI 실시간 자동매매 모니터링")
    
    usdt_total, usdt_free, pos_data = fetch_live_monitoring()
    
    if raw_df is not None and not raw_df.empty:
        current_price = raw_df['Close'].iloc[-1]
        
        # 1. 상단 요약 카드
        col1, col2, col3, col4 = st.columns(4)
        if usdt_total is not None:
            col1.metric("총 보유 자산 (USDT)", f"${usdt_total:,.2f}")
            col2.metric("사용 가능 잔고 (USDT)", f"${usdt_free:,.2f}")
        else:
            col1.metric("총 보유 자산", "조회 불가 (API Key 확인)")
            col2.metric("사용 가능 잔고", "조회 불가")
            
        col3.metric("현재 비트코인 가격", f"${current_price:,.2f}")
        
        if pos_data:
            side = pos_data['side']
            unrealized_pnl = pos_data['unrealizedPnl']
            pnl_roe = pos_data['pnlRoe']
            leverage = pos_data['leverage']
            color = "normal" if unrealized_pnl >= 0 else "inverse"
            col4.metric(f"포지션: {side} ({leverage}x)", f"${unrealized_pnl:,.2f} ({pnl_roe:+.2f}%)", delta_color=color)
        else:
            col4.metric("현재 포지션", "없음 (대기중)")

        st.markdown("---")

        # 2. ver_2 AI 모델 예측 섹션
        st.subheader("🎯 ver_2 AI 모델 실시간 예측 분석")
        current_feat = raw_df.iloc[-1]
        features = ['Returns', 'Body_Size', 'Upper_Shadow', 'Lower_Shadow', 
                    'RSI_14', 'ATR_Ratio', 'Close_vs_SMA20', 'BB_Width', 'BB_Pos', 'Close_vs_SMA20_1H']
        X = current_feat[features].values.reshape(1, -1)
        
        probs = model.predict_proba(X)[0]
        pred_class = np.argmax(probs)
        
        p_short, p_neutral, p_long = probs[0]*100, probs[1]*100, probs[2]*100
        rsi_val = current_feat['RSI_14'] * 100
        
        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
        if pred_class == 2:
            m_col1.success("📈 AI 시그널: LONG (상승)")
        elif pred_class == 0:
            m_col1.error("📉 AI 시그널: SHORT (하락)")
        else:
            m_col1.warning("⏳ AI 시그널: HOLD (관망)")
            
        m_col2.metric("상승(Long) 확률", f"{p_long:.1f}%")
        m_col3.metric("하락(Short) 확률", f"{p_short:.1f}%")
        m_col4.metric("RSI (14)", f"{rsi_val:.1f}")
        
        prob_df = pd.DataFrame({
            '방향': ['하락 (Short)', '관망 (Hold)', '상승 (Long)'],
            '확률 (%)': [p_short, p_neutral, p_long]
        })
        fig_prob = go.Figure(go.Bar(
            x=prob_df['확률 (%)'],
            y=prob_df['방향'],
            orientation='h',
            marker_color=['#ff4b4b', '#ffa100', '#00c853'],
            text=[f"{p:.1f}%" for p in [p_short, p_neutral, p_long]],
            textposition='auto'
        ))
        fig_prob.update_layout(template='plotly_dark', height=180, margin=dict(l=0, r=0, t=10, b=10), xaxis=dict(range=[0, 100]))
        st.plotly_chart(fig_prob, use_container_width=True)

        st.markdown("---")

        # 3. 차트 & 포지션 상세
        col_chart, col_info = st.columns([2.8, 1.2])

        df_chart = raw_df.reset_index()
        df_chart['timestamp'] = df_chart['timestamp'].dt.tz_localize('UTC').dt.tz_convert('Asia/Seoul').dt.tz_localize(None) if df_chart['timestamp'].dt.tz is not None else df_chart['timestamp']

        with col_chart:
            st.subheader("📊 실시간 15분봉 차트")
            fig = go.Figure(data=[go.Candlestick(
                x=df_chart['timestamp'],
                open=df_chart['Open'], high=df_chart['High'], low=df_chart['Low'], close=df_chart['Close'],
                increasing_line_color='green', decreasing_line_color='red'
            )])
            if pos_data:
                line_color = "lime" if pos_data['side'] == "LONG" else "red"
                fig.add_hline(y=pos_data['entryPrice'], line_dash="dash", line_color=line_color, 
                              annotation_text=f"{pos_data['side']} 진입평단: ${pos_data['entryPrice']:,.2f}")
            fig.update_layout(template='plotly_dark', height=520, xaxis_rangeslider_visible=False, margin=dict(l=0, r=0, t=30, b=0))
            st.plotly_chart(fig, use_container_width=True)

        with col_info:
            st.subheader("💡 포지션 상세 정보")
            if pos_data:
                side_badge = "🔴 SHORT (하락 배팅)" if pos_data['side'] == "SHORT" else "🟢 LONG (상승 배팅)"
                st.markdown(f"#### {side_badge}")
                st.markdown(f"""
                * **레버리지:** `{pos_data['leverage']}x` (격리)
                * **계약 수량:** `{pos_data['contracts']:.3f} BTC`
                * **진입 평단가:** `${pos_data['entryPrice']:,.2f}`
                * **현재가:** `${current_price:,.2f}`
                * **총 포지션 가치:** `${pos_data['notionalValue']:,.2f} USDT`
                * **실제 투입 증거금:** `${pos_data['initialMargin']:,.2f} USDT`
                * **미실현 손익 (PnL):** `${pos_data['unrealizedPnl']:,.2f} USDT` (`{pos_data['pnlRoe']:+.2f}%`)
                * **추정 청산가:** `${pos_data['liquidationPrice']:,.2f}`
                """)
            else:
                st.warning("현재 진입한 포지션이 없습니다.\n\nAI가 45% 이상의 확실한 신호를 기다리고 있습니다.")

        if st.button("🔄 실시간 데이터 새로고침"):
            st.rerun()

elif mode == "📈 ver_2 백테스트 시뮬레이터":
    st.title("📈 비트코인 ver_2 AI 선물 백테스트 시뮬레이터")

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
    max_pyramid = st.sidebar.slider("최대 물타기 허용 횟수", 0, 5, 3)

    st.sidebar.markdown("---")
    use_rsi_exit = st.sidebar.checkbox("RSI 초과 포지션 종료 적용", value=True)
    if use_rsi_exit:
        rsi_long_th = st.sidebar.slider("RSI 롱(Long) 청산 수치", min_value=50, max_value=95, value=90, step=1)
        # RSI 숏 청산 수치 범위를 0~20까지 설정
        rsi_short_th = st.sidebar.slider("RSI 숏(Short) 청산 수치", min_value=0, max_value=20, value=10, step=1)
    else:
        rsi_long_th, rsi_short_th = 90, 10

    def run_backtest(df, entry_th, exit_th, leverage, invest_ratio, max_pyramid, use_rsi_exit, rsi_long_th, rsi_short_th):
        balance = 10000.0
        position = 0
        avg_entry_price = 0.0
        invested_margin = 0.0
        position_size = 0.0
        pyramid_count = 0
        fee_rate = 0.0004

        balance_history = []
        trades = []

        for i in range(len(df)):
            close_price = df['Close'].iloc[i]
            rsi = df['RSI_14'].iloc[i] * 100.0
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
                    position, invested_margin, position_size, pyramid_count = 0, 0, 0, 0
                    balance_history.append(balance)
                    continue

                if use_rsi_exit:
                    if (position == 1 and rsi >= rsi_long_th) or (position == -1 and rsi <= rsi_short_th):
                        trades.append({'date': date, 'type': 'RSI 초과 포지션 종료', 'price': close_price, 'profit': net_profit})
                        balance += net_profit
                        position, invested_margin, position_size, pyramid_count = 0, 0, 0, 0
                        balance_history.append(max(balance, 0))
                        continue

            is_loss = (position == 1 and close_price < avg_entry_price) or (position == -1 and close_price > avg_entry_price)

            if position == 0:
                if prob >= entry_th:
                    if pred == 2:
                        position = 1
                        avg_entry_price = close_price
                        invested_margin = balance * invest_ratio
                        position_size = invested_margin * leverage
                        pyramid_count = 0
                        trades.append({'date': date, 'type': 'Long 진입', 'price': close_price, 'profit': 0.0})
                    elif pred == 0:
                        position = -1
                        avg_entry_price = close_price
                        invested_margin = balance * invest_ratio
                        position_size = invested_margin * leverage
                        pyramid_count = 0
                        trades.append({'date': date, 'type': 'Short 진입', 'price': close_price, 'profit': 0.0})
            else:
                if (position == 1 and pred == 0) or (position == -1 and pred == 2):
                    if prob >= exit_th:
                        trades.append({'date': date, 'type': '신호 포지션 종료', 'price': close_price, 'profit': net_profit})
                        balance += net_profit
                        position, invested_margin, position_size, pyramid_count = 0, 0, 0, 0
                elif (position == 1 and pred == 2) or (position == -1 and pred == 0):
                    if prob >= entry_th and is_loss and balance > 0 and pyramid_count < max_pyramid:
                        add_margin = balance * invest_ratio
                        add_size = add_margin * leverage
                        total_size = position_size + add_size
                        avg_entry_price = (position_size * avg_entry_price + add_size * close_price) / total_size
                        invested_margin += add_margin
                        position_size = total_size
                        pyramid_count += 1
                        trades.append({'date': date, 'type': f'물타기 ({pyramid_count}/{max_pyramid}회)', 'price': close_price, 'profit': 0.0})

            balance_history.append(max(balance + (net_profit if position != 0 else 0), 0))

        return balance_history, trades

    df_sub = raw_df[raw_df.index >= pd.to_datetime(start_date)]

    if df_sub.empty:
        st.error("선택한 날짜 이후의 데이터가 없습니다.")
    else:
        features = ['Returns', 'Body_Size', 'Upper_Shadow', 'Lower_Shadow', 
                    'RSI_14', 'ATR_Ratio', 'Close_vs_SMA20', 'BB_Width', 'BB_Pos', 'Close_vs_SMA20_1H']
        X = df_sub[features]
        probs = model.predict_proba(X)
        df_sub = df_sub.copy()
        df_sub['Max_Prob'] = np.max(probs, axis=1)
        df_sub['Pred'] = np.argmax(probs, axis=1)

        hist, trades = run_backtest(df_sub, entry_th, exit_th, leverage, invest_ratio, max_pyramid, use_rsi_exit, rsi_long_th, rsi_short_th)
        df_sub['Balance'] = hist

        col1, col2, col3 = st.columns(3)
        col1.metric("초기 자본금", "$10,000.00")
        col2.metric("최종 자산", f"${hist[-1]:,.2f}", f"{(hist[-1]/10000 - 1)*100:.2f}%")
        col3.metric("총 거래 횟수", f"{len([t for t in trades if '진입' in t['type']])} 회 진입")

        st.subheader("💰 백테스트 누적 자산 변화 (ver_2 모델)")
        fig_bal = go.Figure()
        fig_bal.add_trace(go.Scatter(x=df_sub.index, y=df_sub['Balance'], mode='lines', name='포트폴리오 가치', line=dict(color='cyan', width=2)))
        fig_bal.add_hline(y=10000, line_dash="dash", line_color="gray")
        fig_bal.update_layout(template='plotly_dark', height=400, xaxis_title="Date", yaxis_title="Balance (USD)", dragmode='pan', hovermode='x unified')
        st.plotly_chart(fig_bal, use_container_width=True, config={'scrollZoom': True})

        st.subheader("📈 바이낸스 선물 15분봉 및 ver_2 진입/청산 타점 시각화")
        fig_candle = go.Figure(data=[go.Candlestick(
            x=df_sub.index, open=df_sub['Open'], high=df_sub['High'], low=df_sub['Low'], close=df_sub['Close'], name='BTC Price',
            increasing_line_color='green', decreasing_line_color='red'
        )])

        margin = (df_sub['High'].max() - df_sub['Low'].min()) * 0.02
        long_entries = [t for t in trades if t['type'] == 'Long 진입']
        short_entries = [t for t in trades if t['type'] == 'Short 진입']
        add_margins = [t for t in trades if '물타기' in t['type']]
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
