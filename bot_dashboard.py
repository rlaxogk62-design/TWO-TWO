import streamlit as st
import ccxt
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os
import joblib
import ta
from dotenv import load_dotenv

# 페이지 설정
st.set_page_config(layout="wide", page_title="BTC ver_2 AI 대시보드", page_icon="🤖")
st.title("🤖 바이낸스 ver_2 AI 자동매매 모니터링")

# API 로드
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, '.env'))
API_KEY = os.getenv('BINANCE_API_KEY')
SECRET_KEY = os.getenv('BINANCE_SECRET_KEY')

if not API_KEY or not SECRET_KEY:
    st.error("⚠️ .env 파일에 API 키가 없습니다.")
    st.stop()

# ver_2 모델 로드
@st.cache_resource
def load_ver2_model():
    model_path = os.path.join(BASE_DIR, 'xgboost_btc_15m_v2_advanced.pkl')
    if not os.path.exists(model_path):
        model_path = os.path.join(BASE_DIR, 'ver_2', 'models', 'xgboost_btc_15m_v2_advanced.pkl')
    return joblib.load(model_path)

model = load_ver2_model()

# 거래소 객체 초기화 (캐싱하여 속도 향상)
@st.cache_resource
def get_exchange():
    return ccxt.binanceusdm({
        'apiKey': API_KEY,
        'secret': SECRET_KEY,
        'enableRateLimit': True,
        'options': {
            'defaultType': 'future',
            'warnOnFetchBalance': False
        }
    })

exchange = get_exchange()
SYMBOL = 'BTC/USDT'

# 실시간 데이터 가져오기 및 ver_2 피처 계산
def fetch_live_data():
    try:
        account_info = exchange.fapiPrivateV2GetAccount()
        usdt_total = float(account_info.get('totalWalletBalance', 0.0))
        usdt_free = float(account_info.get('availableBalance', 0.0))
        
        # 포지션 정밀 조회 (LONG / SHORT 정밀 판별)
        pos_data = None
        raw_positions = exchange.fapiPrivateV2GetPositionRisk({'symbol': 'BTCUSDT'})
        for p in raw_positions:
            amt = float(p.get('positionAmt', 0))
            if amt != 0:
                pos_data = {
                    'symbol': 'BTC/USDT',
                    'side': 'LONG' if amt > 0 else 'SHORT',
                    'contracts': abs(amt),
                    'entryPrice': float(p.get('entryPrice', 0)),
                    'unrealizedPnl': float(p.get('unRealizedProfit', 0)),
                    'leverage': int(p.get('leverage', 25))
                }
                break
                
        ohlcv = exchange.fetch_ohlcv(SYMBOL, '15m', limit=150)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        
        # ver_2 피처 계산
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
        df_chart = df.reset_index()
        df_chart['timestamp'] = df_chart['timestamp'].dt.tz_localize('UTC').dt.tz_convert('Asia/Seoul').dt.tz_localize(None)
        
        return usdt_total, usdt_free, pos_data, df, df_chart
    except Exception as e:
        st.error(f"데이터를 불러오는 중 오류가 발생했습니다: {e}")
        return None, None, None, None, None

# 데이터 로드
usdt_total, usdt_free, pos_data, df_calc, df_chart = fetch_live_data()

if df_chart is not None:
    current_price = df_chart['Close'].iloc[-1]

    # =========================================
    # 1. 상단 요약 카드 (Metrics)
    # =========================================
    col1, col2, col3, col4 = st.columns(4)
    
    col1.metric("총 보유 자산 (USDT)", f"${usdt_total:,.2f}")
    col2.metric("사용 가능 잔고 (USDT)", f"${usdt_free:,.2f}")
    col3.metric("현재 비트코인 가격", f"${current_price:,.2f}")
    
    if pos_data:
        side = pos_data['side'].upper()
        size = float(pos_data['contracts'])
        entry_price = float(pos_data['entryPrice'])
        unrealized_pnl = float(pos_data['unrealizedPnl'])
        leverage = pos_data['leverage']
        
        color = "normal" if unrealized_pnl >= 0 else "inverse"
        col4.metric(f"포지션: {side} ({leverage}x)", f"${unrealized_pnl:,.2f} 수익중", delta_color=color)
    else:
        col4.metric("현재 포지션", "없음 (대기중)")

    st.markdown("---")

    # =========================================
    # 2. ver_2 AI 모델 예측 섹션
    # =========================================
    st.subheader("🎯 ver_2 AI 모델 실시간 예측 분석")
    
    if model is not None and not df_calc.empty:
        current_feat = df_calc.iloc[-1]
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
        
        # 확률 바 차트
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
    else:
        st.info("ver_2 모델 로딩 중...")

    st.markdown("---")

    # =========================================
    # 3. 실시간 15분봉 차트 & 포지션 정보
    # =========================================
    col_chart, col_info = st.columns([3, 1])

    with col_chart:
        st.subheader("📊 실시간 15분봉 차트")
        fig = go.Figure(data=[go.Candlestick(
            x=df_chart['timestamp'],
            open=df_chart['Open'], high=df_chart['High'], low=df_chart['Low'], close=df_chart['Close'],
            increasing_line_color='green', decreasing_line_color='red'
        )])
        
        if pos_data:
            line_color = "lime" if side == "LONG" else "red"
            fig.add_hline(y=entry_price, line_dash="dash", line_color=line_color, 
                          annotation_text=f"{side} 진입가: ${entry_price:,.2f}")
            
        fig.update_layout(template='plotly_dark', height=450, xaxis_rangeslider_visible=False, margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig, use_container_width=True)

    with col_info:
        st.subheader("💡 포지션 상세")
        if pos_data:
            st.info(f"**방향:** {side}\n\n"
                    f"**수량:** {size} BTC\n\n"
                    f"**진입가:** ${entry_price:,.2f}\n\n"
                    f"**현재가:** ${current_price:,.2f}\n\n"
                    f"**레버리지:** {leverage}배\n\n"
                    f"**미실현 손익:** ${unrealized_pnl:,.2f}")
        else:
            st.warning("현재 진입한 포지션이 없습니다.\n\nAI가 45% 이상의 확실한 신호를 기다리고 있습니다.")
            
    if st.button("🔄 실시간 데이터 새로고침"):
        st.rerun()
