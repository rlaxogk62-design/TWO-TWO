import yfinance as yf
import pandas as pd
import numpy as np
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

def run_simulation():
    # 1. 모델 로드
    model_path = 'xgboost_btc_15m_3class_strict.pkl'
    model = joblib.load(model_path)

    # 2. 데이터 다운로드 (7월 17일부터 27일까지)
    print("데이터를 불러오는 중...")
    btc = yf.Ticker("BTC-USD")
    # yfinance 15m는 최대 60일, 넉넉히 7월 15일부터 가져와서 지표 계산 후 자름
    df = btc.history(start="2026-07-15", end="2026-07-28", interval="15m")
    if df.index.tz is not None:
        df.index = df.index.tz_convert('Asia/Seoul').tz_localize(None)

    # 3. 지표 계산
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

    # 4. 7월 17일 이후로 데이터 자르기
    df = df[df.index >= pd.to_datetime('2026-07-17 00:00:00')]

    if df.empty:
        print("데이터가 없습니다.")
        return

    # 5. 모델 예측
    features = ['Open', 'High', 'Low', 'Close', 'Volume', 'SMA_7', 'RSI_14', 'SMA_1H', 'SMA_4H', 'Vol_4H', 'SMA_24H', 'BB_Width']
    X = df[features]
    probs = model.predict_proba(X)
    df['Max_Prob'] = np.max(probs, axis=1)
    df['Pred'] = np.argmax(probs, axis=1)

    # 6. 백테스트 로직 및 변수 초기화
    entry_th = 0.45
    exit_th = 0.45
    leverage = 25
    invest_ratio = 0.25
    use_rsi_exit = True
    rsi_long_th = 90
    rsi_short_th = 10
    
    initial_balance = 100.0  # 100달러
    balance = initial_balance
    position = 0
    avg_entry_price = 0.0
    invested_margin = 0.0
    position_size = 0.0
    fee_rate = 0.0004 # 바이낸스 시장가 수수료 기준 대략 0.04%

    trades = []
    
    for i in range(len(df)):
        close_price = df['Close'].iloc[i]
        rsi = df['RSI_14'].iloc[i]
        prob = df['Max_Prob'].iloc[i]
        pred = df['Pred'].iloc[i]
        date = df.index[i]

        if balance <= 0:
            continue

        net_profit = 0
        if position != 0:
            price_change_pct = (close_price - avg_entry_price) / avg_entry_price * position
            net_profit = (position_size * price_change_pct) - (position_size * fee_rate * 2)

            # 강제 청산 (마진콜)
            if net_profit <= -invested_margin:
                trades.append({'date': date, 'type': '마진콜 청산', 'price': close_price, 'profit': -invested_margin})
                balance -= invested_margin
                position, invested_margin, position_size = 0, 0, 0
                continue

            # RSI 기반 강제 청산
            if use_rsi_exit:
                if (position == 1 and rsi >= rsi_long_th) or (position == -1 and rsi <= rsi_short_th):
                    trades.append({'date': date, 'type': 'RSI 청산', 'price': close_price, 'profit': net_profit})
                    balance += net_profit
                    position, invested_margin, position_size = 0, 0, 0
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

    # 마지막 날짜에 포지션이 남아있다면 종가로 계산
    if position != 0:
        price_change_pct = (df['Close'].iloc[-1] - avg_entry_price) / avg_entry_price * position
        net_profit = (position_size * price_change_pct) - (position_size * fee_rate * 2)
        balance += net_profit
        trades.append({'date': df.index[-1], 'type': '현재 포지션 종료(종가)', 'price': df['Close'].iloc[-1], 'profit': net_profit})

    print("-" * 50)
    print(f"💰 초기 자본금: ${initial_balance:,.2f}")
    print(f"💰 최종 자산: ${balance:,.2f}")
    print(f"📈 수익률: {((balance/initial_balance)-1)*100:.2f}%")
    print("-" * 50)
    
    entries = [t for t in trades if '진입' in t['type']]
    liquidations = [t for t in trades if t['type'] == '마진콜 청산']
    print(f"총 거래(진입) 횟수: {len(entries)}회")
    print(f"마진콜 발생 횟수: {len(liquidations)}회")
    print("\n[상세 거래 내역]")
    
    for t in trades:
        profit_str = f"수익: ${t['profit']:,.2f}" if t['profit'] != 0 else ""
        print(f"[{t['date']}] {t['type']} | BTC가: ${t['price']:,.2f} | {profit_str}")

if __name__ == "__main__":
    run_simulation()
