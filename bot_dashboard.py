import streamlit as st
import ccxt
import pandas as pd
import plotly.graph_objects as go
import os
import time
from dotenv import load_dotenv

# 페이지 설정
st.set_page_config(layout="wide", page_title="BTC 자동매매 모니터링", page_icon="📈")
st.title("📈 바이낸스 자동매매 실시간 대시보드")

# API 로드
load_dotenv()
API_KEY = os.getenv('BINANCE_API_KEY')
SECRET_KEY = os.getenv('BINANCE_SECRET_KEY')

if not API_KEY or not SECRET_KEY:
    st.error("⚠️ .env 파일에 API 키가 없습니다.")
    st.stop()

# 거래소 객체 초기화 (캐싱하여 속도 향상)
@st.cache_resource
def get_exchange():
    return ccxt.binanceusdm({
        'apiKey': API_KEY,
        'secret': SECRET_KEY,
        'enableRateLimit': True,
    })

exchange = get_exchange()
SYMBOL = 'BTC/USDT'

# 실시간 데이터 가져오기 함수
def fetch_live_data():
    try:
        # 1. 잔고 조회
        balance = exchange.fetch_balance()
        usdt_total = float(balance['total'].get('USDT', 0.0))
        usdt_free = float(balance['free'].get('USDT', 0.0))
        
        # 2. 포지션 조회
        positions = exchange.fetch_positions([SYMBOL])
        pos_data = None
        for p in positions:
            if p['symbol'] == SYMBOL and float(p['contracts']) > 0:
                pos_data = p
                break
                
        # 3. 최근 캔들 차트 데이터 (15분봉)
        ohlcv = exchange.fetch_ohlcv(SYMBOL, '15m', limit=100)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df['timestamp'] = df['timestamp'].dt.tz_localize('UTC').dt.tz_convert('Asia/Seoul').dt.tz_localize(None)
        
        return usdt_total, usdt_free, pos_data, df
    except Exception as e:
        st.error(f"데이터를 불러오는 중 오류가 발생했습니다: {e}")
        return None, None, None, None

# 데이터 로드
usdt_total, usdt_free, pos_data, df = fetch_live_data()

if df is not None:
    current_price = df['Close'].iloc[-1]

    # =========================================
    # 상단 요약 카드 (Metrics)
    # =========================================
    col1, col2, col3, col4 = st.columns(4)
    
    col1.metric("총 보유 자산 (USDT)", f"${usdt_total:,.2f}")
    col2.metric("사용 가능 잔고 (USDT)", f"${usdt_free:,.2f}")
    col3.metric("현재 비트코인 가격", f"${current_price:,.2f}")
    
    if pos_data:
        side = pos_data['side'].upper() # LONG or SHORT
        size = float(pos_data['contracts'])
        entry_price = float(pos_data['entryPrice'])
        unrealized_pnl = float(pos_data['unrealizedPnl'])
        leverage = pos_data['leverage']
        
        color = "normal" if unrealized_pnl >= 0 else "inverse"
        col4.metric(f"현재 포지션: {side} ({leverage}x)", f"${unrealized_pnl:,.2f} 수익중", delta_color=color)
    else:
        col4.metric("현재 포지션", "없음 (대기중)")

    st.markdown("---")

    # =========================================
    # 포지션 상세 정보 & 차트
    # =========================================
    col_chart, col_info = st.columns([3, 1])

    with col_chart:
        st.subheader("📊 실시간 15분봉 차트")
        fig = go.Figure(data=[go.Candlestick(
            x=df['timestamp'],
            open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'],
            increasing_line_color='green', decreasing_line_color='red'
        )])
        
        # 포지션이 있으면 진입가에 선 긋기
        if pos_data:
            line_color = "lime" if side == "LONG" else "red"
            fig.add_hline(y=entry_price, line_dash="dash", line_color=line_color, 
                          annotation_text=f"{side} 진입가: ${entry_price:,.2f}")
            
        fig.update_layout(template='plotly_dark', height=500, xaxis_rangeslider_visible=False, margin=dict(l=0, r=0, t=30, b=0))
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
            st.warning("현재 진입한 포지션이 없습니다.\n\n매매 봇이 다음 신호를 기다리고 있습니다.")
            
    # 새로고침 버튼
    if st.button("🔄 실시간 데이터 새로고침"):
        st.rerun()
