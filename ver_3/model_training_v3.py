import ccxt
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os
import time
from datetime import datetime, timezone

# ==========================================
# 1. 바이낸스 15분봉 데이터 수집 (ccxt)
# (yfinance는 과거 60일 제한이 있어, 지정하신 25년 7월부터 데이터를 뽑기 위해 ccxt를 사용합니다)
# ==========================================
print(">>> 1. 바이낸스(Binance) 15분봉 데이터 수집 중 (2025-07-01 ~ 현재)...")
exchange = ccxt.binanceusdm()
symbol = 'BTC/USDT'
timeframe = '15m'

# 시작 시점: 2025-07-01 00:00:00 KST -> UTC 기준 2025-06-30 15:00:00
start_dt = datetime(2025, 6, 30, 15, 0, 0, tzinfo=timezone.utc)
since = int(start_dt.timestamp() * 1000)

all_ohlcv = []
while True:
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=1500)
        if not ohlcv:
            break
            
        all_ohlcv.extend(ohlcv)
        since = ohlcv[-1][0] + 1
        
        # 현재 시점까지 데이터를 다 가져왔으면 종료
        if ohlcv[-1][0] >= exchange.milliseconds() - 15 * 60 * 1000:
            break
            
        if len(all_ohlcv) % 6000 == 0:
            print(f"  ... 현재 {len(all_ohlcv)}개 캔들 수집 완료")
            
        time.sleep(0.2)
    except Exception as e:
        print("데이터 수집 중 오류 발생:", e)
        time.sleep(2)

btc_df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
btc_df['timestamp'] = pd.to_datetime(btc_df['timestamp'], unit='ms')
btc_df.set_index('timestamp', inplace=True)

# 시간대 처리 (UTC -> KST)
btc_df.index = btc_df.index.tz_localize('UTC').tz_convert('Asia/Seoul').tz_localize(None)
btc_df = btc_df[~btc_df.index.duplicated(keep='first')]

print(f"완료: 총 {len(btc_df)}개의 캔들 수집됨. (기간: {btc_df.index[0]} ~ {btc_df.index[-1]})")

# ==========================================
# 2. 다중 타임프레임(MTF) 및 기술적 피처 엔지니어링 (누수 없음 & 100% 비율 변환)
# ==========================================
print("\n>>> 2. 피처 엔지니어링 중 (누수 방지 직접연산 및 비율 변환 적용)...")

# 캔들 형태 비율 (원시 가격 대체)
btc_df['Returns'] = btc_df['Close'].pct_change()
btc_df['Body_Size'] = (btc_df['Close'] - btc_df['Open']) / btc_df['Open']
btc_df['Upper_Shadow'] = (btc_df['High'] - btc_df[['Open', 'Close']].max(axis=1)) / btc_df['Close']
btc_df['Lower_Shadow'] = (btc_df[['Open', 'Close']].min(axis=1) - btc_df['Low']) / btc_df['Close']

# 단기 이동평균 이격도 (비율)
btc_df['Close_vs_SMA7'] = btc_df['Close'] / btc_df['Close'].rolling(window=7).mean() - 1

# 거래량 비율 (단기 vs 20평균)
vol_sma20 = btc_df['Volume'].rolling(window=20).mean()
btc_df['Vol_Ratio'] = btc_df['Volume'] / (vol_sma20 + 1e-8)

# RSI (14) - v1 원본 수식 그대로 적용
delta = btc_df['Close'].diff()
gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
btc_df['RSI_14'] = 100 - (100 / (1 + gain / loss))

# --- 다중 타임프레임 (MTF) 지표 (누수 방지 직접 연산 + 비율 적용) ---
# 1시간 추세 이격도 (15분 * 4)
btc_df['Close_vs_SMA1H'] = btc_df['Close'] / btc_df['Close'].rolling(window=4).mean() - 1
# 4시간 추세 이격도 (15분 * 16)
btc_df['Close_vs_SMA4H'] = btc_df['Close'] / btc_df['Close'].rolling(window=16).mean() - 1
btc_df['Vol_4H'] = btc_df['Returns'].rolling(window=16).std() # 4시간 변동성 (이미 비율 기반)
# 24시간 추세 이격도 (15분 * 96)
btc_df['Close_vs_SMA24H'] = btc_df['Close'] / btc_df['Close'].rolling(window=96).mean() - 1

# 볼린저 밴드 너비 (비율 기반) - v1 수식 그대로 적용
btc_df['BB_Std'] = btc_df['Close'].rolling(window=20).std()
btc_df['BB_Width'] = (btc_df['BB_Std'] * 4) / btc_df['Close'].rolling(window=20).mean()

# ==========================================
# 3. 목표 변수(Label) 정의 (v1 로직 유지)
# ==========================================
THRESHOLD = 0.001

btc_df['Next_Return'] = btc_df['Close'].shift(-1) / btc_df['Close'] - 1

# 0: 하락(Down), 1: 횡보(Neutral), 2: 상승(Up)
conditions = [
    btc_df['Next_Return'] < -THRESHOLD,
    btc_df['Next_Return'] > THRESHOLD
]
choices = [0, 2]
btc_df['Target'] = np.select(conditions, choices, default=1)

# 결측치 제거
btc_df.dropna(inplace=True)

# ==========================================
# 4. 데이터 분할 (Train: 25년 7월 ~ 26년 6월 / Test: 26년 7월 ~ )
# ==========================================
features = ['Returns', 'Body_Size', 'Upper_Shadow', 'Lower_Shadow', 'Vol_Ratio',
            'Close_vs_SMA7', 'RSI_14', 'Close_vs_SMA1H', 'Close_vs_SMA4H', 'Vol_4H', 'Close_vs_SMA24H', 'BB_Width']

# 기간 분할 적용
train_df = btc_df[(btc_df.index >= '2025-07-01') & (btc_df.index < '2026-07-01')]
test_df = btc_df[btc_df.index >= '2026-07-01']

X_train, y_train = train_df[features], train_df['Target']
X_test, y_test = test_df[features], test_df['Target']

print(f"\n>>> 3. 데이터 분할 완료")
print(f"학습(Train) 기간: {train_df.index.min()} ~ {train_df.index.max()} ({len(X_train)}개)")
print(f"평가(Test) 기간: {test_df.index.min()} ~ {test_df.index.max()} ({len(X_test)}개)")
print(f"\n학습 데이터 타겟 분포: \n{y_train.value_counts(normalize=True).sort_index() * 100}")

# ==========================================
# 5. XGBoost 모델 학습
# ==========================================
print("\n>>> 4. XGBoost 다중 분류 모델 학습 시작 (v3)...")
model_xgb = XGBClassifier(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
    objective='multi:softprob'
)

model_xgb.fit(X_train, y_train)

# 폴더가 없으면 생성 후 저장
os.makedirs('ver_3/models', exist_ok=True)
joblib.dump(model_xgb, 'ver_3/models/xgboost_btc_15m_v3.pkl')
print("모델 저장 완료: ver_3/models/xgboost_btc_15m_v3.pkl")

# ==========================================
# 6. 26년 7월 테스트 데이터 평가
# ==========================================
print("\n>>> 5. 26년 7월 실전 데이터(Test) 평가 결과...")
y_pred = model_xgb.predict(X_test)

acc = accuracy_score(y_test, y_pred)
print(f"[테스트 세트 전체 정확도]: {acc * 100:.2f}%")
print(classification_report(y_test, y_pred, target_names=['하락(0)', '횡보(1)', '상승(2)']))
