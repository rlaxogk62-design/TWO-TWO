import streamlit as st
import ccxt
import requests
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os
import joblib
import ta
import time
from datetime import datetime
from dotenv import load_dotenv

# 페이지 설정
st.set_page_config(layout="wide", page_title="BTC ver_2 AI 대시보드 및 시뮬레이터", page_icon="🤖")

# API 로드
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, '.env'))
API_KEY = os.getenv('BINANCE_API_KEY')
SECRET_KEY = os.getenv('BINANCE_SECRET_KEY')

# ver_2 실전 적용 파라미터 (고정)
LIVE_LEVERAGE = 25
LIVE_INVEST_RATIO = 0.25
LIVE_ENTRY_TH = 0.42
LIVE_EXIT_TH = 0.40
LIVE_MAX_PYRAMID = 3
LIVE_USE_RSI_EXIT = True
LIVE_RSI_LONG_TH = 90
LIVE_RSI_SHORT_TH = 10

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

# 도쿄 VPS API 또는 바이낸스 선물 직접 캔들 수집 (실시간 반영을 위해 ttl=10초 설정)
@st.cache_data(ttl=10)
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
        if exchange:
            try:
                ohlcv = exchange.fetch_ohlcv(SYMBOL, '15m', limit=1500)
                df = pd.DataFrame(ohlcv, columns=['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df.set_index('timestamp', inplace=True)
                df.index = df.index.tz_localize('UTC').tz_convert('Asia/Seoul').tz_localize(None)
            except Exception:
                df = None

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

# 실시간 바이낸스 모니터링 데이터 수집 (계좌 잔고 및 오픈 포지션)
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
    except Exception:
        return None, None, None

# 실제 바이낸스 체결 내역 및 실전 자산 추이 생성 (미실현 손익 반영)
def fetch_real_trades_and_equity(pos_data=None):
    if not exchange:
        return [], None
    try:
        raw_trades = exchange.fetch_my_trades('BTC/USDT', limit=100)
        account_info = exchange.fapiPrivateV2GetAccount()
        current_wallet_balance = float(account_info.get('totalWalletBalance', 0.0))

        formatted_trades = []
        if raw_trades:
            sorted_trades = sorted(raw_trades, key=lambda x: x['timestamp'])

            for t in sorted_trades:
                dt = pd.to_datetime(t['timestamp'], unit='ms')
                if dt.tz is None:
                    dt = dt.tz_localize('UTC').tz_convert('Asia/Seoul').tz_localize(None)
                else:
                    dt = dt.tz_convert('Asia/Seoul').tz_localize(None)

                side = t.get('side', '').upper()
                price = float(t.get('price', 0.0))
                amount = float(t.get('amount', 0.0))
                info = t.get('info', {})
                pnl = float(info.get('realizedPnl', 0.0))
                fee_info = t.get('fee', {})
                fee = float(fee_info.get('cost', 0.0)) if fee_info else float(info.get('commission', 0.0))

                trade_type = "🟢 매수 (BUY)" if side == "BUY" else "🔴 매도 (SELL)"
                if pnl != 0:
                    trade_type += f" [청산 PnL: ${pnl:+,.2f}]"

                formatted_trades.append({
                    'date': dt,
                    'type': trade_type,
                    'raw_side': side,
                    'price': price,
                    'amount': amount,
                    'pnl': pnl,
                    'fee': fee
                })

        # 실제 계좌 잔고 변화 추이 구성
        equity_records = []
        cum_pnl = sum([tr['pnl'] - tr['fee'] for tr in formatted_trades])
        start_balance = max(current_wallet_balance - cum_pnl, 0.0)

        running_balance = start_balance
        for tr in formatted_trades:
            running_balance += (tr['pnl'] - tr['fee'])
            equity_records.append({
                'date': tr['date'],
                'Balance': running_balance
            })

        # 현재 시점 (NOW)의 평가 자산 기록 추가 (미실현 손익 반영)
        now_dt = pd.Timestamp.now()
        unrealized_pnl = pos_data['unrealizedPnl'] if pos_data else 0.0
        current_total_equity = current_wallet_balance + unrealized_pnl

        equity_records.append({
            'date': now_dt,
            'Balance': current_total_equity
        })

        equity_df = pd.DataFrame(equity_records)
        if not equity_df.empty:
            equity_df.set_index('date', inplace=True)

        return formatted_trades, equity_df
    except Exception:
        return [], None

# 백테스트 시뮬레이션 전용 엔진
def run_backtest(df, entry_th, exit_th, leverage, invest_ratio, max_pyramid, use_rsi_exit, rsi_long_th, rsi_short_th):
    balance = 10000.0
    free_balance = 10000.0
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
        high_price = df['High'].iloc[i]
        low_price = df['Low'].iloc[i]
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
                free_balance = balance
                position, invested_margin, position_size, pyramid_count = 0, 0, 0, 0
                balance_history.append(balance)
                continue

            if use_rsi_exit:
                if (position == 1 and rsi >= rsi_long_th) or (position == -1 and rsi <= rsi_short_th):
                    trades.append({'date': date, 'type': 'RSI 초과 포지션 종료', 'price': close_price, 'profit': net_profit})
                    balance += net_profit
                    free_balance = balance
                    position, invested_margin, position_size, pyramid_count = 0, 0, 0, 0
                    balance_history.append(max(balance, 0))
                    continue

        is_loss = (position == 1 and low_price < avg_entry_price) or (position == -1 and high_price > avg_entry_price) or (position == 1 and close_price < avg_entry_price) or (position == -1 and close_price > avg_entry_price)

        if position == 0:
            if prob >= entry_th:
                if pred == 2:
                    position = 1
                    avg_entry_price = close_price
                    invested_margin = free_balance * invest_ratio
                    position_size = invested_margin * leverage
                    free_balance -= invested_margin
                    pyramid_count = 0
                    trades.append({'date': date, 'type': 'Long 신규진입', 'price': close_price, 'profit': 0.0})
                elif pred == 0:
                    position = -1
                    avg_entry_price = close_price
                    invested_margin = free_balance * invest_ratio
                    position_size = invested_margin * leverage
                    free_balance -= invested_margin
                    pyramid_count = 0
                    trades.append({'date': date, 'type': 'Short 신규진입', 'price': close_price, 'profit': 0.0})
        else:
            if (position == 1 and pred == 0) or (position == -1 and pred == 2):
                if prob >= exit_th:
                    trades.append({'date': date, 'type': '신호 포지션 종료', 'price': close_price, 'profit': net_profit})
                    balance += net_profit
                    free_balance = balance
                    position, invested_margin, position_size, pyramid_count = 0, 0, 0, 0
            elif (position == 1 and pred == 2) or (position == -1 and pred == 0):
                if prob >= entry_th and is_loss and free_balance > 0 and pyramid_count < max_pyramid:
                    add_margin = free_balance * invest_ratio
                    add_size = add_margin * leverage
                    total_size = position_size + add_size
                    avg_entry_price = (position_size * avg_entry_price + add_size * close_price) / total_size
                    invested_margin += add_margin
                    free_balance -= add_margin
                    position_size = total_size
                    pyramid_count += 1
                    trades.append({'date': date, 'type': f'물타기 ({pyramid_count}/{max_pyramid}회)', 'price': close_price, 'profit': 0.0})

        balance_history.append(max(balance + (net_profit if position != 0 else 0), 0))

    return balance_history, trades


# 실시간 바이낸스 모니터링 전용 차트 렌더링
def render_real_monitoring_charts(df_candle, real_trades, equity_df, pos_data=None, current_balance=100.0):
    # 1. 실제 계좌 누적 자산 변화 차트
    st.subheader("💰 실제 계좌 누적 자산 변화 (미실현 손익 반영 평가 자산)")
    fig_bal = go.Figure()

    if equity_df is not None and not equity_df.empty:
        fig_bal.add_trace(go.Scatter(
            x=equity_df.index, y=equity_df['Balance'], 
            mode='lines+markers', name='실제 계좌 잔고 (USDT)', 
            line=dict(color='cyan', width=2), marker=dict(size=6, color='cyan')
        ))
    else:
        fig_bal.add_trace(go.Scatter(
            x=[df_candle.index.min(), df_candle.index.max()], 
            y=[current_balance, current_balance], 
            mode='lines', name='실제 계좌 잔고 (USDT)', 
            line=dict(color='cyan', width=2, dash='dash')
        ))

    fig_bal.update_layout(
        template='plotly_dark', height=350, 
        xaxis_title="시간 (KST)", yaxis_title="잔고 (USDT)", 
        dragmode='pan', hovermode='x unified', margin=dict(l=0, r=0, t=30, b=10)
    )
    st.plotly_chart(fig_bal, use_container_width=True, config={'scrollZoom': True})

    # 2. 실제 바이낸스 15분봉 및 실전 매매 타점 시각화 차트
    st.subheader("📈 실제 바이낸스 선물 15분봉 및 실전 매매 타점 시각화")
    fig_candle = go.Figure(data=[go.Candlestick(
        x=df_candle.index, open=df_candle['Open'], high=df_candle['High'], low=df_candle['Low'], close=df_candle['Close'], name='BTC Price',
        increasing_line_color='green', decreasing_line_color='red'
    )])

    # 실전 포지션 라인 표시 (오픈 포지션 존재 시)
    if pos_data:
        side = pos_data['side']
        entry_p = pos_data['entryPrice']
        liq_p = pos_data['liquidationPrice']
        line_col = "lime" if side == "LONG" else "red"
        fig_candle.add_hline(y=entry_p, line_dash="dash", line_color=line_col, 
                             annotation_text=f"실전 {side} 진입평단: ${entry_p:,.2f}", annotation_position="top left")
        if liq_p > 0:
            fig_candle.add_hline(y=liq_p, line_dash="dot", line_color="orange", 
                                 annotation_text=f"추정 청산가: ${liq_p:,.2f}", annotation_position="bottom left")

    # 실제 바이낸스 체결 타점 마커
    if real_trades:
        buy_trades = [t for t in real_trades if t['raw_side'] == 'BUY']
        sell_trades = [t for t in real_trades if t['raw_side'] == 'SELL']
        pnl_trades = [t for t in real_trades if t['pnl'] != 0]

        margin = (df_candle['High'].max() - df_candle['Low'].min()) * 0.02

        if buy_trades:
            fig_candle.add_trace(go.Scatter(
                x=[t['date'] for t in buy_trades], y=[t['price'] - margin for t in buy_trades],
                mode='markers', marker=dict(symbol='triangle-up', size=14, color='lime', line=dict(width=1, color='darkgreen')),
                name='실제 매수 체결 (BUY)'
            ))
        if sell_trades:
            fig_candle.add_trace(go.Scatter(
                x=[t['date'] for t in sell_trades], y=[t['price'] + margin for t in sell_trades],
                mode='markers', marker=dict(symbol='triangle-down', size=14, color='red', line=dict(width=1, color='darkred')),
                name='실제 매도 체결 (SELL)'
            ))
        if pnl_trades:
            fig_candle.add_trace(go.Scatter(
                x=[t['date'] for t in pnl_trades], y=[t['price'] for t in pnl_trades],
                mode='markers', marker=dict(symbol='x', size=12, color='yellow'),
                name='실제 손익 확정 청산'
            ))

    fig_candle.update_layout(
        template='plotly_dark', height=580, 
        xaxis_rangeslider_visible=False, yaxis_title="Price (USD)", 
        dragmode='pan', hovermode='x unified', margin=dict(l=0, r=0, t=30, b=10)
    )
    st.plotly_chart(fig_candle, use_container_width=True, config={'scrollZoom': True})


# 백테스트 전용 차트 렌더링
def render_backtest_charts(df_input, balance_hist, trades_list):
    st.subheader("💰 백테스트 누적 자산 변화 (ver_2 모델)")
    fig_bal = go.Figure()
    fig_bal.add_trace(go.Scatter(x=df_input.index, y=balance_hist, mode='lines', name='포트폴리오 가치', line=dict(color='cyan', width=2)))
    fig_bal.add_hline(y=10000, line_dash="dash", line_color="gray", annotation_text="초기 자본금 ($10,000)")
    fig_bal.update_layout(template='plotly_dark', height=380, xaxis_title="Date", yaxis_title="Balance (USD)", dragmode='pan', hovermode='x unified', margin=dict(l=0, r=0, t=30, b=10))
    st.plotly_chart(fig_bal, use_container_width=True, config={'scrollZoom': True})

    st.subheader("📈 바이낸스 선물 15분봉 및 ver_2 백테스트 진입/청산 타점 시각화")
    fig_candle = go.Figure(data=[go.Candlestick(
        x=df_input.index, open=df_input['Open'], high=df_input['High'], low=df_input['Low'], close=df_input['Close'], name='BTC Price',
        increasing_line_color='green', decreasing_line_color='red'
    )])

    margin = (df_input['High'].max() - df_input['Low'].min()) * 0.02
    long_entries = [t for t in trades_list if t['type'] == 'Long 신규진입']
    short_entries = [t for t in trades_list if t['type'] == 'Short 신규진입']
    add_margins = [t for t in trades_list if '물타기' in t['type']]
    model_exits = [t for t in trades_list if t['type'] == '신호 포지션 종료']
    rsi_exits = [t for t in trades_list if t['type'] == 'RSI 초과 포지션 종료']
    liquidations = [t for t in trades_list if t['type'] == '마진콜 청산']

    if long_entries:
        fig_candle.add_trace(go.Scatter(x=[t['date'] for t in long_entries], y=[t['price'] - margin for t in long_entries],
                                        mode='markers', marker=dict(symbol='triangle-up', size=12, color='lime', line=dict(width=1, color='darkgreen')), name='Long 신규진입'))
    if short_entries:
        fig_candle.add_trace(go.Scatter(x=[t['date'] for t in short_entries], y=[t['price'] + margin for t in short_entries],
                                        mode='markers', marker=dict(symbol='triangle-down', size=12, color='red', line=dict(width=1, color='darkred')), name='Short 신규진입'))
    if add_margins:
        fig_candle.add_trace(go.Scatter(x=[t['date'] for t in add_margins], y=[t['price'] for t in add_margins],
                                        mode='markers', marker=dict(symbol='star', size=10, color='blue'), name='물타기 (추가진입)'))
    if model_exits:
        fig_candle.add_trace(go.Scatter(x=[t['date'] for t in model_exits], y=[t['price'] for t in model_exits],
                                        mode='markers', marker=dict(symbol='x', size=10, color='yellow'), name='신호 포지션 종료'))
    if rsi_exits:
        fig_candle.add_trace(go.Scatter(x=[t['date'] for t in rsi_exits], y=[t['price'] for t in rsi_exits],
                                        mode='markers', marker=dict(symbol='x', size=12, color='orange'), name='RSI 초과 포지션 종료'))
    if liquidations:
        fig_candle.add_trace(go.Scatter(x=[t['date'] for t in liquidations], y=[t['price'] for t in liquidations],
                                        mode='markers', marker=dict(symbol='x', size=14, color='purple'), name='마진콜 강제청산'))

    fig_candle.update_layout(template='plotly_dark', height=580, xaxis_rangeslider_visible=False, yaxis_title="Price (USD)", dragmode='pan', hovermode='x unified', margin=dict(l=0, r=0, t=30, b=10))
    st.plotly_chart(fig_candle, use_container_width=True, config={'scrollZoom': True})


# 사이드바 메뉴 선택
st.sidebar.title("📌 메뉴 선택")
mode = st.sidebar.radio("모드 전환", ["🤖 실시간 자동매매 모니터링", "📈 ver_2 백테스트 시뮬레이터"])

raw_df = get_candle_data()

# 실시간 모니터링 자동 새로고침 옵션
st.sidebar.markdown("---")
st.sidebar.subheader("⚡ 실시간 자동 새로고침")
auto_refresh = st.sidebar.checkbox("자동 새로고침 활성화", value=True)
refresh_sec = st.sidebar.selectbox("새로고침 주기 (초)", [5, 10, 15, 30, 60], index=0)

if mode == "🤖 실시간 자동매매 모니터링":

    fragment_kwargs = {"run_every": refresh_sec} if auto_refresh else {}

    @st.fragment(**fragment_kwargs)
    def render_live_monitoring():
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        col_title, col_time = st.columns([3, 1])
        with col_title:
            st.title("🤖 바이낸스 ver_2 AI 실시간 자동매매 모니터링")
        with col_time:
            st.caption(f"⏱️ 마지막 업데이트: `{now_str}`")
            if st.button("🔄 실시간 데이터 새로고침", use_container_width=True):
                st.rerun()

        # 실제 바이낸스 계좌 데이터 수집
        usdt_total, usdt_free, pos_data = fetch_live_monitoring()
        df_live = raw_df.copy() if raw_df is not None else pd.DataFrame()

        if df_live.empty:
            st.error("데이터를 불러오는 중입니다...")
            return

        current_price = df_live['Close'].iloc[-1]

        # 1. 핵심 지표 카드 (4 메트릭)
        m1, m2, m3, m4 = st.columns(4)
        if usdt_total is not None:
            m1.metric("총 보유 자산 (USDT)", f"${usdt_total:,.2f}")
            m2.metric("사용 가능 잔고 (USDT)", f"${usdt_free:,.2f}")
        else:
            m1.metric("총 보유 자산", "조회 불가 (API Key 확인)")
            m2.metric("사용 가능 잔고", "조회 불가")

        m3.metric("현재 비트코인 가격", f"${current_price:,.2f}")

        if pos_data:
            side = pos_data['side']
            unrealized_pnl = pos_data['unrealizedPnl']
            pnl_roe = pos_data['pnlRoe']
            leverage = pos_data['leverage']
            delta_color = "normal" if unrealized_pnl >= 0 else "inverse"
            m4.metric(f"포지션: {side} ({leverage}x)", f"${unrealized_pnl:,.2f} ({pnl_roe:+.2f}%)", delta_color=delta_color)
        else:
            m4.metric("현재 포지션", "없음 (대기중)")

        st.markdown("---")

        # 2. 실전 적용 파라미터 정보 배지
        st.subheader("⚙️ 현재 가동 중인 실전 매매 파라미터 (고정)")
        p_col1, p_col2, p_col3, p_col4, p_col5, p_col6 = st.columns(6)
        p_col1.info(f"**레버리지**\n\n`{LIVE_LEVERAGE}x (격리)`")
        p_col2.info(f"**1회 진입 비중**\n\n`{int(LIVE_INVEST_RATIO*100)}%`")
        p_col3.info(f"**진입 임계점**\n\n`{LIVE_ENTRY_TH:.2f}`")
        p_col4.info(f"**청산 임계점**\n\n`{LIVE_EXIT_TH:.2f}`")
        p_col5.info(f"**최대 물타기**\n\n`{LIVE_MAX_PYRAMID}회`")
        p_col6.info(f"**RSI 청산 조건**\n\n`Long>={LIVE_RSI_LONG_TH} / Short<={LIVE_RSI_SHORT_TH}`")

        st.markdown("---")

        # 3. ver_2 AI 모델 실시간 예측 분석
        st.subheader("🎯 ver_2 AI 모델 실시간 예측 분석")
        current_feat = df_live.iloc[-1]
        features = ['Returns', 'Body_Size', 'Upper_Shadow', 'Lower_Shadow', 
                    'RSI_14', 'ATR_Ratio', 'Close_vs_SMA20', 'BB_Width', 'BB_Pos', 'Close_vs_SMA20_1H']
        X = current_feat[features].values.reshape(1, -1)

        probs = model.predict_proba(X)[0]
        pred_class = np.argmax(probs)

        p_short, p_neutral, p_long = probs[0]*100, probs[1]*100, probs[2]*100
        rsi_val = current_feat['RSI_14'] * 100

        sig_col1, sig_col2, sig_col3, sig_col4 = st.columns(4)
        if pred_class == 2:
            sig_col1.success("📈 AI 시그널: LONG (상승)")
        elif pred_class == 0:
            sig_col1.error("📉 AI 시그널: SHORT (하락)")
        else:
            sig_col1.warning("⏳ AI 시그널: HOLD (관망)")

        sig_col2.metric("상승(Long) 확률", f"{p_long:.1f}%")
        sig_col3.metric("하락(Short) 확률", f"{p_short:.1f}%")
        sig_col4.metric("RSI (14)", f"{rsi_val:.1f}")

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
        fig_prob.update_layout(template='plotly_dark', height=160, margin=dict(l=0, r=0, t=10, b=10), xaxis=dict(range=[0, 100]))
        st.plotly_chart(fig_prob, use_container_width=True)

        st.markdown("---")

        # 4. 실제 바이낸스 체결 내역 및 실전 계좌 잔고 추이 시각화 (미실현 손익 포함)
        real_trades, equity_df = fetch_real_trades_and_equity(pos_data=pos_data)
        curr_bal = usdt_total if usdt_total is not None else 100.0

        render_real_monitoring_charts(df_live, real_trades, equity_df, pos_data=pos_data, current_balance=curr_bal)

        st.markdown("---")

        # 5. 실제 바이낸스 상세 체결 일지 표
        st.subheader("📝 실제 바이낸스 체결 일지")
        if real_trades:
            log_df = pd.DataFrame(real_trades)
            log_df['시간 (KST)'] = log_df['date'].dt.strftime('%Y-%m-%d %H:%M:%S')
            log_df['체결가 (USD)'] = log_df['price'].apply(lambda x: f"${x:,.2f}")
            log_df['체결수량 (BTC)'] = log_df['amount'].apply(lambda x: f"{x:.3f}")
            log_df['실현손익 (USDT)'] = log_df['pnl'].apply(lambda x: f"${x:+,.2f}" if x != 0 else "-")
            log_df['수수료 (USDT)'] = log_df['fee'].apply(lambda x: f"${x:,.4f}")

            show_df = log_df[['시간 (KST)', 'type', '체결가 (USD)', '체결수량 (BTC)', '실현손익 (USDT)', '수수료 (USDT)']]
            show_df.columns = ['시간 (KST)', '구분', '체결가 (USD)', '체결수량 (BTC)', '실현손익 (USDT)', '수수료 (USDT)']
            st.dataframe(show_df, use_container_width=True, hide_index=True)
        else:
            st.info("바이낸스 계좌에서 최근 체결된 거래 내역이 없거나 API Key가 조회되지 않습니다.")

    render_live_monitoring()

elif mode == "📈 ver_2 백테스트 시뮬레이터":
    st.title("📈 비트코인 ver_2 AI 선물 백테스트 시뮬레이터")

    st.sidebar.header("📅 투자 기간 설정")
    min_date = raw_df.index.min().date()
    max_date = raw_df.index.max().date()
    start_date = st.sidebar.date_input("시뮬레이션 시작일", min_value=min_date, max_value=max_date, value=min_date)

    st.sidebar.header("⚙️ ver_2 매매 파라미터")
    entry_th = st.sidebar.slider("진입 임계점 (Entry Threshold)", min_value=0.10, max_value=0.90, value=0.42, step=0.01)
    exit_th = st.sidebar.slider("청산 임계점 (Exit Threshold)", min_value=0.10, max_value=0.90, value=0.40, step=0.01)

    st.sidebar.markdown("---")
    leverage = st.sidebar.slider("레버리지 (Leverage)", 1, 50, 25)
    invest_ratio = st.sidebar.slider("1회 진입 비중 (%)", 1, 50, 25) / 100.0
    max_pyramid = st.sidebar.slider("최대 물타기 허용 횟수", 0, 5, 3)

    st.sidebar.markdown("---")
    use_rsi_exit = st.sidebar.checkbox("RSI 초과 포지션 종료 적용", value=True)
    if use_rsi_exit:
        rsi_long_th = st.sidebar.slider("RSI 롱(Long) 청산 수치", min_value=50, max_value=95, value=90, step=1)
        rsi_short_th = st.sidebar.slider("RSI 숏(Short) 청산 수치", min_value=0, max_value=20, value=10, step=1)
    else:
        rsi_long_th, rsi_short_th = 90, 10

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

        initial_entries = len([t for t in trades if '신규진입' in t['type']])
        pyramid_entries = len([t for t in trades if '물타기' in t['type']])

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("초기 자본금", "$10,000.00")
        col2.metric("최종 자산", f"${hist[-1]:,.2f}", f"{(hist[-1]/10000 - 1)*100:.2f}%")
        col3.metric("독립 신규 포지션 수", f"{initial_entries}회")
        col4.metric("총 매매 주문 수 (물타기 포함)", f"{initial_entries + pyramid_entries}회", f"추가 물타기 {pyramid_entries}회 포함")

        render_backtest_charts(df_sub, hist, trades)

        st.subheader("📝 ver_2 백테스트 상세 매매 일지")
        if trades:
            trades_df = pd.DataFrame(trades)
            trades_df.columns = ['시간', '구분', '체결가(USD)', '수익금(USD)']
            trades_df['시간'] = pd.to_datetime(trades_df['시간']).dt.strftime('%Y-%m-%d %H:%M')
            trades_df['체결가(USD)'] = trades_df['체결가(USD)'].apply(lambda x: f"${x:,.2f}")
            trades_df['수익금(USD)'] = trades_df['수익금(USD)'].apply(lambda x: f"${x:,.2f}" if x != 0 else "-")

            st.dataframe(trades_df, use_container_width=True, hide_index=True)
        else:
            st.info("해당 기간 동안 발생한 매매 내역이 없습니다.")
