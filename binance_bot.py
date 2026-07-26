import sys
import ccxt
import pandas as pd
import numpy as np
import ta
import joblib
import os
import time
import schedule
from dotenv import load_dotenv

# stdout 라인 버퍼링 설정 (로그 파일 즉시 출력)
sys.stdout.reconfigure(line_buffering=True)

# 스크립트 기준 절대 경로 설정
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 환경 변수 로드 (.env 파일에 BINANCE_API_KEY, BINANCE_SECRET_KEY 저장 필요)
load_dotenv(os.path.join(BASE_DIR, '.env'))
API_KEY = os.getenv('BINANCE_API_KEY')
SECRET_KEY = os.getenv('BINANCE_SECRET_KEY')

# 사용자 설정 파라미터
SYMBOL = 'BTC/USDT'
TIMEFRAME = '15m'
LEVERAGE = 25
INVEST_RATIO = 0.25 # 25%
ENTRY_TH = 0.45
EXIT_TH = 0.45
RSI_LONG_EXIT = 90
RSI_SHORT_EXIT = 10

# ver_2 모델 로드
MODEL_PATH = os.path.join(BASE_DIR, 'xgboost_btc_15m_v2_advanced.pkl')
if not os.path.exists(MODEL_PATH):
    alt_path = os.path.join(BASE_DIR, 'ver_2', 'models', 'xgboost_btc_15m_v2_advanced.pkl')
    if os.path.exists(alt_path):
        MODEL_PATH = alt_path
    else:
        raise FileNotFoundError(f"{MODEL_PATH} 파일을 찾을 수 없습니다.")

model = joblib.load(MODEL_PATH)
print(f"✅ ver_2 모델 로드 완료: {MODEL_PATH}")

# 바이낸스 선물 객체 초기화
exchange = ccxt.binanceusdm({
    'apiKey': API_KEY,
    'secret': SECRET_KEY,
    'enableRateLimit': True,
    'options': {
        'defaultType': 'future'
    }
})

def set_leverage_and_margin():
    try:
        exchange.set_leverage(LEVERAGE, SYMBOL)
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 레버리지 {LEVERAGE}배 설정 완료")
    except Exception as e:
        print(f"레버리지 설정 예외/확인: {e}")

    try:
        exchange.set_margin_mode('ISOLATED', SYMBOL)
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 마진 모드: 격리(ISOLATED) 설정 완료")
    except Exception as e:
        print(f"마진 모드 설정 예외/확인: {e}")

def get_recent_data():
    ohlcv = exchange.fetch_ohlcv(SYMBOL, TIMEFRAME, limit=150)
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    
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
    
    # MTF 1시간봉 지표
    df_1h = df['Close'].resample('1h').last().to_frame(name='Close_1H')
    sma_20_1h = ta.trend.sma_indicator(df_1h['Close_1H'], window=20)
    df_1h['Close_vs_SMA20_1H'] = df_1h['Close_1H'] / sma_20_1h - 1
    
    df = df.join(df_1h[['Close_vs_SMA20_1H']], how='left')
    df['Close_vs_SMA20_1H'] = df['Close_vs_SMA20_1H'].ffill()

    df.dropna(inplace=True)
    return df

def get_current_position():
    positions = exchange.fetch_positions([SYMBOL])
    for p in positions:
        if p['symbol'] == SYMBOL:
            size = float(p['contracts']) if p['contracts'] else 0.0
            side = p['side']
            entry_price = float(p['entryPrice']) if p['entryPrice'] else 0.0
            if size > 0:
                if side == 'long':
                    return 1, size, entry_price
                elif side == 'short':
                    return -1, size, entry_price
    return 0, 0.0, 0.0

def get_usdt_balance():
    try:
        balance = exchange.fetch_balance()
        return float(balance['total'].get('USDT', 0.0))
    except Exception as e:
        print(f"잔고 조회 오류: {e}")
        return 0.0

def execute_trade():
    print(f"\n--- [{time.strftime('%Y-%m-%d %H:%M:%S')}] ver_2 모델 15분봉 체크 시작 ---")
    try:
        balance = get_usdt_balance()
        position, pos_size, avg_entry_price = get_current_position()
        
        pos_text = "없음"
        if position == 1:
            pos_text = f"LONG ({pos_size} BTC)"
        elif position == -1:
            pos_text = f"SHORT ({pos_size} BTC)"
            
        print(f"💰 현재 선물 잔액: {balance:.2f} USDT")
        print(f"📊 현재 포지션 상태: {pos_text}")
        print("-" * 40)

        df = get_recent_data()
        current_data = df.iloc[-1]
        
        features = ['Returns', 'Body_Size', 'Upper_Shadow', 'Lower_Shadow', 
                    'RSI_14', 'ATR_Ratio', 'Close_vs_SMA20', 'BB_Width', 'BB_Pos', 'Close_vs_SMA20_1H']
        X = current_data[features].values.reshape(1, -1)
        
        # ver_2 모델 예측
        probs = model.predict_proba(X)
        max_prob = np.max(probs, axis=1)[0]
        pred = np.argmax(probs, axis=1)[0] # 0: Short, 1: Hold, 2: Long
        
        rsi = current_data['RSI_14'] * 100.0
        close_price = current_data['Close']
        
        print(f"현재 가격: ${close_price:,.2f}, RSI: {rsi:.2f}")
        print(f"ver_2 AI 예측: {'Long' if pred==2 else 'Short' if pred==0 else 'Hold'} (확률: {max_prob*100:.1f}%)")
        
        position, pos_size, avg_entry_price = get_current_position()
        
        # 1. RSI 기반 강제 청산
        if position == 1 and rsi >= RSI_LONG_EXIT:
            print("🚨 RSI 롱 과매수 청산 조건 도달. 포지션을 종료합니다.")
            exchange.create_market_sell_order(SYMBOL, pos_size, params={'reduceOnly': True})
            return
        elif position == -1 and rsi <= RSI_SHORT_EXIT:
            print("🚨 RSI 숏 과매도 청산 조건 도달. 포지션을 종료합니다.")
            exchange.create_market_buy_order(SYMBOL, pos_size, params={'reduceOnly': True})
            return
            
        # 2. 신호에 의한 반대 포지션 종료
        if position != 0:
            if (position == 1 and pred == 0) or (position == -1 and pred == 2):
                if max_prob >= EXIT_TH:
                    print("🔄 반대 신호 감지. 기존 포지션을 종료합니다.")
                    if position == 1:
                        exchange.create_market_sell_order(SYMBOL, pos_size, params={'reduceOnly': True})
                    else:
                        exchange.create_market_buy_order(SYMBOL, pos_size, params={'reduceOnly': True})
                    position = 0
                    time.sleep(2)
            
            # 물타기(추가 진입)
            else:
                is_loss = (position == 1 and close_price < avg_entry_price) or (position == -1 and close_price > avg_entry_price)
                if (position == 1 and pred == 2) or (position == -1 and pred == 0):
                    if max_prob >= ENTRY_TH and is_loss:
                        balance = get_usdt_balance()
                        add_margin = balance * INVEST_RATIO
                        add_size = (add_margin * LEVERAGE) / close_price
                        
                        print(f"🌊 [물타기 조건 도달] 평단가: ${avg_entry_price:.2f}, 현재가: ${close_price:.2f}")
                        if position == 1:
                            exchange.create_market_buy_order(SYMBOL, add_size)
                            print("✅ Long 추가 진입 완료")
                        else:
                            exchange.create_market_sell_order(SYMBOL, add_size)
                            print("✅ Short 추가 진입 완료")
                        time.sleep(2)
                        return
        
        # 3. 신규 진입
        if position == 0:
            if max_prob >= ENTRY_TH and pred in [0, 2]:
                balance = get_usdt_balance()
                margin_to_use = balance * INVEST_RATIO
                target_size = (margin_to_use * LEVERAGE) / close_price
                
                print(f"🚀 ver_2 신규 진입 신호! 증거금: {margin_to_use:.2f} USDT, 수량: {target_size:.4f}")
                if pred == 2:
                    exchange.create_market_buy_order(SYMBOL, target_size)
                    print("✅ Long 진입 완료")
                elif pred == 0:
                    exchange.create_market_sell_order(SYMBOL, target_size)
                    print("✅ Short 진입 완료")

    except Exception as e:
        print(f"오류 발생: {e}")

if __name__ == "__main__":
    print("=== ver_2 비트코인 AI 선물 자동매매 봇 시작 ===")
    if not API_KEY or not SECRET_KEY:
        print("⚠️ 환경 변수(API_KEY, SECRET_KEY)가 설정되지 않았습니다. .env 파일을 확인해주세요.")
        exit(1)
        
    set_leverage_and_margin()
    execute_trade()
    
    schedule.every().hour.at(":00").do(execute_trade)
    schedule.every().hour.at(":15").do(execute_trade)
    schedule.every().hour.at(":30").do(execute_trade)
    schedule.every().hour.at(":45").do(execute_trade)
    
    while True:
        schedule.run_pending()
        time.sleep(1)
