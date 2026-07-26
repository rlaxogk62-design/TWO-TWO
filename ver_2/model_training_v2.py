import ccxt
import pandas as pd
import numpy as np
import ta
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.model_selection import TimeSeriesSplit
import joblib
import os
import time

# ==========================================
# 1 & 5. 바이낸스 선물 과거 데이터 대량 수집 (ccxt 사용)
# ==========================================
def fetch_binance_futures_data(symbol='BTC/USDT', timeframe='15m', limit=1500, total_candles=15000):
    print(f">>> {symbol} {timeframe} 바이낸스 선물 데이터 수집 시작 (목표: {total_candles}개)...")
    exchange = ccxt.binance({'options': {'defaultType': 'future'}})
    
    all_ohlcv = []
    since = exchange.milliseconds() - total_candles * 15 * 60 * 1000 
    
    while len(all_ohlcv) < total_candles:
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=limit)
            if not ohlcv:
                break
            all_ohlcv.extend(ohlcv)
            since = ohlcv[-1][0] + 1
            print(f"수집 진행 중... 현재 {len(all_ohlcv)}개")
            time.sleep(0.3)
        except Exception as e:
            print("데이터 수집 중 에러/완료:", e)
            break
            
    df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    df = df[~df.index.duplicated(keep='last')]
    return df.iloc[-total_candles:]

# ==========================================
# 2, 3, 4. 피처 엔지니어링 (정상성 변환, 올바른 지표, MTF)
# ==========================================
def feature_engineering(df):
    print(">>> 피처 엔지니어링 진행 중 (정상성, MTF, ta 패키지)...")
    df = df.copy()
    
    # 기초 비율 피처 (정상성 - Stationary 변환)
    df['Returns'] = df['Close'].pct_change()
    df['Body_Size'] = (df['Close'] - df['Open']) / df['Open']
    df['Upper_Shadow'] = (df['High'] - df[['Open', 'Close']].max(axis=1)) / df['Close']
    df['Lower_Shadow'] = (df[['Open', 'Close']].min(axis=1) - df['Low']) / df['Close']
    
    # 올바른 RSI (Wilder's Smoothing) & ATR (ta 패키지 사용)
    df['RSI_14'] = ta.momentum.rsi(df['Close'], window=14) / 100.0  # 0~1 스케일
    df['ATR_14'] = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'], window=14)
    df['ATR_Ratio'] = df['ATR_14'] / df['Close'] # 가격 독립적 변동성 비율
    
    # SMA 및 볼린저 밴드 (비율 기반)
    sma_20 = ta.trend.sma_indicator(df['Close'], window=20)
    df['Close_vs_SMA20'] = df['Close'] / sma_20 - 1
    
    bb_high = ta.volatility.bollinger_hband(df['Close'], window=20)
    bb_low = ta.volatility.bollinger_lband(df['Close'], window=20)
    df['BB_Width'] = (bb_high - bb_low) / sma_20
    df['BB_Pos'] = (df['Close'] - bb_low) / (bb_high - bb_low + 1e-8)
    
    # 다중 타임프레임 (MTF): 1시간봉 지표 계산 후 15분봉에 ffill 병합
    df_1h = df['Close'].resample('1h').last().to_frame(name='Close_1H')
    sma_20_1h = ta.trend.sma_indicator(df_1h['Close_1H'], window=20)
    df_1h['Close_vs_SMA20_1H'] = df_1h['Close_1H'] / sma_20_1h - 1
    
    df = df.join(df_1h[['Close_vs_SMA20_1H']], how='left')
    df['Close_vs_SMA20_1H'] = df['Close_vs_SMA20_1H'].ffill()

    return df

# ==========================================
# 6, 7, 8. 동적 임계값, 수수료 반영, 개선된 라벨링
# ==========================================
def create_labels(df, fee_slippage=0.0015, horizon=4):
    print(">>> 목표 변수 생성 (ATR 기반 동적 임계값, 수수료/슬리피지 반영)...")
    df = df.copy()
    
    # 4캔들(1시간) 뒤의 미래 가격
    df['Future_Close'] = df['Close'].shift(-horizon)
    
    # 동적 임계값: 현재 ATR의 0.5배와 (수수료+슬리피지) 중 큰 값
    df['Dynamic_Threshold'] = np.maximum(df['ATR_Ratio'] * 0.5, fee_slippage)
    
    # Target: 0 (하락-Short), 1 (관망-Neutral), 2 (상승-Long)
    future_return = df['Future_Close'] / df['Close'] - 1
    
    conditions = [
        future_return > df['Dynamic_Threshold'],
        future_return < -df['Dynamic_Threshold']
    ]
    choices = [2, 0]
    df['Target'] = np.select(conditions, choices, default=1)
    
    df.dropna(inplace=True)
    return df

def main():
    df = fetch_binance_futures_data(total_candles=15000)
    df = feature_engineering(df)
    df = create_labels(df)
    
    features = ['Returns', 'Body_Size', 'Upper_Shadow', 'Lower_Shadow', 
                'RSI_14', 'ATR_Ratio', 'Close_vs_SMA20', 'BB_Width', 'BB_Pos', 'Close_vs_SMA20_1H']
    
    X = df[features]
    y = df['Target']
    
    # 9. 클래스 불균형 (Class Imbalance) 가중치 계산
    sample_weights = compute_sample_weight(class_weight='balanced', y=y)
    print(f"타겟 분포:\n{y.value_counts(normalize=True)*100}")
    
    # 10. Walk-Forward 시계열 교차 검증
    print("\n>>> Walk-Forward 교차 검증 시작...")
    tscv = TimeSeriesSplit(n_splits=5)
    
    fold = 1
    model_xgb = XGBClassifier(
        n_estimators=200,
        learning_rate=0.03,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        objective='multi:softprob',
        n_jobs=-1
    )
    
    for train_index, test_index in tscv.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        sw_train = sample_weights[train_index]
        
        model_xgb.fit(X_train, y_train, sample_weight=sw_train)
        preds = model_xgb.predict(X_test)
        acc = accuracy_score(y_test, preds)
        print(f"Fold {fold} 정확도: {acc*100:.2f}%")
        fold += 1
        
    print("\n>>> 전체 데이터 모델 학습 및 저장...")
    model_xgb.fit(X, y, sample_weight=sample_weights)
    
    os.makedirs('ver_2/models', exist_ok=True)
    joblib.dump(model_xgb, 'ver_2/models/xgboost_btc_15m_v2_advanced.pkl')
    print("모델 저장 완료: ver_2/models/xgboost_btc_15m_v2_advanced.pkl")

if __name__ == "__main__":
    main()
