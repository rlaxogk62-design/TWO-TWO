import ccxt
import pandas as pd
import numpy as np
import time
import os
import json
import joblib
import schedule
from dotenv import load_dotenv

# 1. 환경 변수 및 설정 로드
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, '.env'))

API_KEY = os.getenv('BINANCE_API_KEY')
SECRET_KEY = os.getenv('BINANCE_SECRET_KEY')

# ver_3 최적 실전 매매 파라미터 (누수 0% 정직한 AI 모델)
LEVERAGE = 25          # 레버리지 25배 (격리)
INVEST_RATIO = 0.25    # 1회 진입 및 물타기 비중: 총액의 25%
ENTRY_TH = 0.40        # 최적 진입 임계점 0.40
EXIT_TH = 0.40         # 최적 청산 임계점 0.40
MAX_PYRAMID = 3        # 최대 물타기 허용 횟수 3회
RSI_LONG_EXIT = 85     # RSI 롱 고점 익절 수치 85
RSI_SHORT_EXIT = 20    # RSI 숏 저점 익절 수치 20

SYMBOL = 'BTC/USDT'
TIMEFRAME = '15m'
STATE_FILE = os.path.join(BASE_DIR, 'bot_state.json')

# 2. 거래소 객체 초기화
exchange = ccxt.binanceusdm({
    'apiKey': API_KEY,
    'secret': SECRET_KEY,
    'enableRateLimit': True,
    'options': {
        'defaultType': 'future',
        'warnOnFetchBalance': False
    }
})

# 3. ver_3 AI 모델 로드
MODEL_PATH = os.path.join(BASE_DIR, 'ver_3', 'models', 'xgboost_btc_15m_v3.pkl')
if not os.path.exists(MODEL_PATH):
    MODEL_PATH = os.path.join(BASE_DIR, 'xgboost_btc_15m_v3.pkl')

if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
    print(f"✅ ver_3 최적 AI 모델 로드 완료: {MODEL_PATH}")
else:
    print(f"❌ ver_3 모델 파일을 찾을 수 없습니다: {MODEL_PATH}")
    exit(1)

# 상태 저장 / 로드 함수
def load_state():
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, 'r') as f:
                return json.load(f)
        except Exception:
            return {'pyramid_count': 0}
    return {'pyramid_count': 0}

def save_state(state):
    try:
        with open(STATE_FILE, 'w') as f:
            json.dump(state, f)
    except Exception as e:
        print(f"상태 저장 실패: {e}")

# 레버리지 및 마진 모드 설정
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

# 4. ver_3 데이터 수집 및 피처 엔지니어링 (100% 누수 방지 비율 파이프라인)
def get_candle_data():
    ohlcv = exchange.fetch_ohlcv(SYMBOL, TIMEFRAME, limit=1500)
    now_ms = exchange.milliseconds()
    candle_duration_ms = 15 * 60 * 1000
    
    # 미마감(현재 진행 중인) 캔들 제거하여 100% 확정 종가만 사용
    if ohlcv and (ohlcv[-1][0] + candle_duration_ms > now_ms):
        ohlcv = ohlcv[:-1]

    df = pd.DataFrame(ohlcv, columns=['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    df.index = df.index.tz_localize('UTC').tz_convert('Asia/Seoul').tz_localize(None)

    # ver_3 피처 엔지니어링 (백테스트 시뮬레이터와 100% 동일)
    df['Returns'] = df['Close'].pct_change()
    df['Body_Size'] = (df['Close'] - df['Open']) / df['Open']
    df['Upper_Shadow'] = (df['High'] - df[['Open', 'Close']].max(axis=1)) / df['Close']
    df['Lower_Shadow'] = (df[['Open', 'Close']].min(axis=1) - df['Low']) / df['Close']

    vol_sma20 = df['Volume'].rolling(window=20).mean()
    df['Vol_Ratio'] = df['Volume'] / (vol_sma20 + 1e-8)

    df['Close_vs_SMA7'] = df['Close'] / df['Close'].rolling(window=7).mean() - 1

    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    df['RSI_14'] = 100 - (100 / (1 + gain / loss))

    df['Close_vs_SMA1H'] = df['Close'] / df['Close'].rolling(window=4).mean() - 1
    df['Close_vs_SMA4H'] = df['Close'] / df['Close'].rolling(window=16).mean() - 1
    df['Vol_4H'] = df['Returns'].rolling(window=16).std()
    df['Close_vs_SMA24H'] = df['Close'] / df['Close'].rolling(window=96).mean() - 1

    df['BB_Std'] = df['Close'].rolling(window=20).std()
    df['BB_Width'] = (df['BB_Std'] * 4) / df['Close'].rolling(window=20).mean()

    df.dropna(inplace=True)
    return df

# 현재 실제 바이낸스 포지션 조회
def get_current_position():
    try:
        raw_positions = exchange.fapiPrivateV2GetPositionRisk({'symbol': 'BTCUSDT'})
        for p in raw_positions:
            amt = float(p.get('positionAmt', 0))
            if amt != 0:
                entry_price = float(p.get('entryPrice', 0))
                side = 1 if amt > 0 else -1
                return side, abs(amt), entry_price
    except Exception as e:
        print(f"포지션 조회 오류: {e}")
    return 0, 0.0, 0.0

# 총 보유 자산 및 사용 가능 선물 잔액 조회
def get_account_balances():
    try:
        account_info = exchange.fapiPrivateV2GetAccount()
        total_equity = float(account_info.get('totalWalletBalance', 0.0))
        available_balance = float(account_info.get('availableBalance', 0.0))
        return total_equity, available_balance
    except Exception as e:
        print(f"잔액 조회 오류: {e}")
        return 0.0, 0.0

# 단일 프로세스 중복 가동 방지 Lock
def acquire_single_instance_lock():
    lock_file_path = os.path.join(BASE_DIR, 'binance_bot.lock')
    try:
        if os.name == 'posix':
            import fcntl
            lock_file = open(lock_file_path, 'w')
            fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return lock_file
        else:
            if os.path.exists(lock_file_path):
                try:
                    os.remove(lock_file_path)
                except Exception:
                    pass
            lock_file = open(lock_file_path, 'w')
            lock_file.write(str(os.getpid()))
            lock_file.flush()
            return lock_file
    except (IOError, OSError):
        print("❌ 이미 다른 binance_bot.py 프로세스가 가동 중입니다. 중복 가동 방지를 위해 종료합니다.")
        exit(0)

lock_handle = acquire_single_instance_lock()
last_processed_candle_time = None

# 5. 메인 매매 실행 함수 (ver_3 최적 알고리즘)
def execute_trade():
    global last_processed_candle_time
    # 바이낸스 API 서버의 캔들 종가 마감 정산 대기 (3초)
    time.sleep(3)
    print(f"\n--- [{time.strftime('%Y-%m-%d %H:%M:%S')}] ver_3 AI 모델 15분봉 체크 시작 ---")
    try:
        # 백테스트 시뮬레이터와 100% 동기화된 데이터 수집
        df = get_candle_data()
        current_candle_time = df.index[-1]

        # 동일 15분봉 캔들 중복 실행 방지
        if last_processed_candle_time == current_candle_time:
            print(f"⚠️ 캔들 ({current_candle_time})은 이미 매매 판단이 완료되었습니다. 중복 실행을 스킵합니다.")
            return
        last_processed_candle_time = current_candle_time

        total_equity, available_balance = get_account_balances()
        position, pos_size, avg_entry_price = get_current_position()
        state = load_state()
        pyramid_count = state.get('pyramid_count', 0)

        # 포지션이 없으면 물타기 카운트 리셋
        if position == 0:
            pyramid_count = 0
            save_state({'pyramid_count': 0})

        pos_text = "없음"
        if position == 1:
            pos_text = f"LONG ({pos_size} BTC) | 물타기 진행: {pyramid_count}/{MAX_PYRAMID}회"
        elif position == -1:
            pos_text = f"SHORT ({pos_size} BTC) | 물타기 진행: {pyramid_count}/{MAX_PYRAMID}회"

        print(f"💰 총 계좌 자산: {total_equity:.2f} USDT | 사용 가능 잔액: {available_balance:.2f} USDT")
        print(f"📊 현재 포지션 상태: {pos_text}")
        print("-" * 40)

        current_data = df.iloc[-1]

        features_v3 = ['Returns', 'Body_Size', 'Upper_Shadow', 'Lower_Shadow', 'Vol_Ratio',
                    'Close_vs_SMA7', 'RSI_14', 'Close_vs_SMA1H', 'Close_vs_SMA4H', 'Vol_4H', 'Close_vs_SMA24H', 'BB_Width']
        X = current_data[features_v3].values.reshape(1, -1)

        probs = model.predict_proba(X)
        max_prob = np.max(probs, axis=1)[0]
        pred = np.argmax(probs, axis=1)[0] # 0: Short, 1: Hold, 2: Long

        rsi = current_data['RSI_14']
        close_price = current_data['Close']

        print(f"현재 가격: ${close_price:,.2f}, RSI: {rsi:.2f}")
        print(f"ver_3 AI 예측: {'Long' if pred==2 else 'Short' if pred==0 else 'Hold'} (확률: {max_prob*100:.1f}%)")

        # 1. RSI 기반 강제 청산 조건
        if position == 1 and rsi >= RSI_LONG_EXIT:
            print(f"🚨 RSI 롱 과매수 청산 조건 도달 (RSI: {rsi:.1f} >= {RSI_LONG_EXIT}). 포지션을 종료합니다.")
            exchange.create_market_sell_order(SYMBOL, pos_size, params={'reduceOnly': True})
            save_state({'pyramid_count': 0})
            return
        elif position == -1 and rsi <= RSI_SHORT_EXIT:
            print(f"🚨 RSI 숏 과매도 청산 조건 도달 (RSI: {rsi:.1f} <= {RSI_SHORT_EXIT}). 포지션을 종료합니다.")
            exchange.create_market_buy_order(SYMBOL, pos_size, params={'reduceOnly': True})
            save_state({'pyramid_count': 0})
            return

        # 2. 반대 신호 감지 포지션 종료 (청산 임계점 EXIT_TH 이상)
        if position != 0:
            if (position == 1 and pred == 0) or (position == -1 and pred == 2):
                if max_prob >= EXIT_TH:
                    print(f"🔄 반대 신호 감지 (확률: {max_prob*100:.1f}% >= {EXIT_TH*100:.0f}%). 기존 포지션을 종료합니다.")
                    if position == 1:
                        exchange.create_market_sell_order(SYMBOL, pos_size, params={'reduceOnly': True})
                    else:
                        exchange.create_market_buy_order(SYMBOL, pos_size, params={'reduceOnly': True})
                    position = 0
                    save_state({'pyramid_count': 0})
                    time.sleep(2)

            # 3. 물타기(추가 진입) 로직 (총 자산의 25% 균등 물타기)
            else:
                is_loss = (position == 1 and close_price < avg_entry_price) or (position == -1 and close_price > avg_entry_price)
                if (position == 1 and pred == 2) or (position == -1 and pred == 0):
                    if max_prob >= ENTRY_TH and is_loss and available_balance > 0 and pyramid_count < MAX_PYRAMID:
                        margin_to_use = min(total_equity * INVEST_RATIO, available_balance)
                        add_size = (margin_to_use * LEVERAGE) / close_price

                        pyramid_count += 1
                        save_state({'pyramid_count': pyramid_count})

                        print(f"🌊 [ver_3 총액 물타기 {pyramid_count}/{MAX_PYRAMID}회 도달] 평단가: ${avg_entry_price:.2f}, 현재가: ${close_price:.2f}")
                        if position == 1:
                            exchange.create_market_buy_order(SYMBOL, add_size)
                            print("✅ Long 추가 진입 완료")
                        else:
                            exchange.create_market_sell_order(SYMBOL, add_size)
                            print("✅ Short 추가 진입 완료")
                        time.sleep(2)
                        return
                    elif pyramid_count >= MAX_PYRAMID:
                        print(f"⚠️ 물타기 최대 허용 횟수({MAX_PYRAMID}회)에 도달하여 추가 진입하지 않습니다.")

        # 4. 신규 진입 로직 (총 자산의 25% * 25배 레버리지)
        if position == 0:
            if max_prob >= ENTRY_TH and pred in [0, 2]:
                margin_to_use = min(total_equity * INVEST_RATIO, available_balance)
                target_size = (margin_to_use * LEVERAGE) / close_price

                print(f"🚀 ver_3 신규 진입 신호! 증거금: {margin_to_use:.2f} USDT, 수량: {target_size:.4f}")
                save_state({'pyramid_count': 0})
                if pred == 2:
                    exchange.create_market_buy_order(SYMBOL, target_size)
                    print("✅ Long 신규 진입 완료")
                elif pred == 0:
                    exchange.create_market_sell_order(SYMBOL, target_size)
                    print("✅ Short 신규 진입 완료")

    except Exception as e:
        print(f"오류 발생: {e}")

if __name__ == "__main__":
    print("=== ver_3 비트코인 AI 선물 자동매매 봇 시작 ===")
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
