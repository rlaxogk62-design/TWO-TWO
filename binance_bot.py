import ccxt
import pandas as pd
import numpy as np
import joblib
import os
import time
import schedule
from dotenv import load_dotenv

# 환경 변수 로드 (.env 파일에 BINANCE_API_KEY, BINANCE_SECRET_KEY 저장 필요)
load_dotenv()
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

# 모델 로드
MODEL_PATH = 'xgboost_btc_15m_3class_strict.pkl'
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"{MODEL_PATH} 파일을 찾을 수 없습니다. 스크립트를 같은 폴더에서 실행해주세요.")
model = joblib.load(MODEL_PATH)

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
    # 1. 레버리지 설정
    try:
        exchange.set_leverage(LEVERAGE, SYMBOL)
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 레버리지 {LEVERAGE}배 설정 완료")
    except Exception as e:
        print(f"레버리지 설정 오류 (이미 설정되었을 수 있음): {e}")

    # 2. 격리(ISOLATED) 마진 모드 설정
    try:
        exchange.set_margin_mode('ISOLATED', SYMBOL)
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 마진 모드: 격리(ISOLATED) 설정 완료")
    except Exception as e:
        # 이미 격리 모드로 설정되어 있으면 바이낸스 API가 에러를 반환하므로 예외 처리합니다.
        print(f"마진 모드 설정 완료 혹은 확인 필요 (이미 격리 상태일 수 있음): {e}")

def get_recent_data():
    # 24시간 이동평균(96개 캔들)을 위해 120개 정도의 캔들을 가져옵니다.
    ohlcv = exchange.fetch_ohlcv(SYMBOL, TIMEFRAME, limit=120)
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    
    # 지표 계산
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
        # USDT가 아예 없는 경우 KeyError가 발생하지 않도록 get() 사용
        return float(balance['total'].get('USDT', 0.0))
    except Exception as e:
        print(f"잔고 조회 오류: {e}")
        return 0.0

def execute_trade():
    print(f"\n--- [{time.strftime('%Y-%m-%d %H:%M:%S')}] 15분봉 체크 시작 ---")
    try:
        # 실시간 잔액 및 포지션 상태 조회 및 출력
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
        
        features = ['Open', 'High', 'Low', 'Close', 'Volume', 'SMA_7', 'RSI_14', 'SMA_1H', 'SMA_4H', 'Vol_4H', 'SMA_24H', 'BB_Width']
        X = current_data[features].values.reshape(1, -1)
        
        # 모델 예측
        probs = model.predict_proba(X)
        max_prob = np.max(probs, axis=1)[0]
        pred = np.argmax(probs, axis=1)[0] # 0: Short, 1: Hold, 2: Long
        
        rsi = current_data['RSI_14']
        close_price = current_data['Close']
        
        print(f"현재 가격: {close_price}, RSI: {rsi:.2f}")
        print(f"예측결과: {'Long' if pred==2 else 'Short' if pred==0 else 'Hold'} (확률: {max_prob:.2f})")
        
        position, pos_size, avg_entry_price = get_current_position()
        
        # 1. RSI 기반 강제 청산 (포지션이 있을 때)
        if position == 1 and rsi >= RSI_LONG_EXIT:
            print("🚨 RSI 롱 청산 조건 도달 (과매수). 포지션 종료합니다.")
            exchange.create_market_sell_order(SYMBOL, pos_size, params={'reduceOnly': True})
            return
        elif position == -1 and rsi <= RSI_SHORT_EXIT:
            print("🚨 RSI 숏 청산 조건 도달 (과매도). 포지션 종료합니다.")
            exchange.create_market_buy_order(SYMBOL, pos_size, params={'reduceOnly': True})
            return
            
        # 2. 신호에 의한 포지션 종료
        if position != 0:
            if (position == 1 and pred == 0) or (position == -1 and pred == 2):
                if max_prob >= EXIT_TH:
                    print("🔄 반대 신호 강도 도달. 기존 포지션을 종료합니다.")
                    if position == 1:
                        exchange.create_market_sell_order(SYMBOL, pos_size, params={'reduceOnly': True})
                    else:
                        exchange.create_market_buy_order(SYMBOL, pos_size, params={'reduceOnly': True})
                    position = 0 # 청산 후 새로운 진입 여부 체크
                    time.sleep(2)
            
            # 2.5 물타기 (추가 진입) 로직
            else:
                is_loss = (position == 1 and close_price < avg_entry_price) or (position == -1 and close_price > avg_entry_price)
                if (position == 1 and pred == 2) or (position == -1 and pred == 0):
                    if max_prob >= ENTRY_TH and is_loss:
                        balance = get_usdt_balance()
                        # 물타기 증거금 (총 잔고의 25%)
                        add_margin = balance * INVEST_RATIO
                        # 실제 추가 주문 수량 계산 (레버리지 반영)
                        add_size = (add_margin * LEVERAGE) / close_price
                        
                        print(f"🌊 [물타기 조건 도달] 평단가: ${avg_entry_price:.2f}, 현재가: ${close_price:.2f} (손실중)")
                        print(f"추가 증거금: {add_margin:.2f} USDT, 추가 수량: {add_size:.4f}")
                        
                        if position == 1: # Long 추매
                            exchange.create_market_buy_order(SYMBOL, add_size)
                            print("✅ Long 추가 진입(물타기) 완료")
                        else: # Short 추매
                            exchange.create_market_sell_order(SYMBOL, add_size)
                            print("✅ Short 추가 진입(물타기) 완료")
                        time.sleep(2)
                        return # 물타기를 한 차례 한 뒤에는 이번 15분봉 체크 종료
        
        # 3. 신규 진입 로직
        if position == 0:
            if max_prob >= ENTRY_TH:
                # 0: Short, 2: Long 일 때만 진입하고, 1: Hold 일 때는 진입하지 않음
                if pred in [0, 2]:
                    balance = get_usdt_balance()
                    # 진입 증거금 (사용자 설정 비율)
                    margin_to_use = balance * INVEST_RATIO
                    # 실제 매수 사이즈 (레버리지 적용)
                    target_size = (margin_to_use * LEVERAGE) / close_price
                    
                    print(f"🚀 신규 진입 조건 도달! 증거금: {margin_to_use:.2f} USDT, 주문 수량: {target_size:.4f}")
                    if pred == 2: # Long
                        exchange.create_market_buy_order(SYMBOL, target_size)
                        print("✅ Long 진입 완료")
                    elif pred == 0: # Short
                        exchange.create_market_sell_order(SYMBOL, target_size)
                        print("✅ Short 진입 완료")

    except Exception as e:
        print(f"오류 발생: {e}")

if __name__ == "__main__":
    print("=== 바이낸스 AI 15분봉 자동매매 봇 시작 ===")
    if not API_KEY or not SECRET_KEY:
        print("⚠️ 환경 변수(API_KEY, SECRET_KEY)가 설정되지 않았습니다. .env 파일을 확인해주세요.")
        exit(1)
        
    set_leverage_and_margin()
    
    # 즉시 1회 실행
    execute_trade()
    
    # 15분마다 정각 (00, 15, 30, 45분)에 실행하도록 설정
    # Binance의 15m 캔들 종가 마감 직후에 실행하기 위해 매시 00, 15, 30, 45분에 동작
    schedule.every().hour.at(":00").do(execute_trade)
    schedule.every().hour.at(":15").do(execute_trade)
    schedule.every().hour.at(":30").do(execute_trade)
    schedule.every().hour.at(":45").do(execute_trade)
    
    while True:
        schedule.run_pending()
        time.sleep(1)
