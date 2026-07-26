import ccxt
import pandas as pd
import numpy as np
import ta
import joblib
import os
import itertools
from concurrent.futures import ProcessPoolExecutor

def get_data_and_predictions():
    ex = ccxt.binance({'options': {'defaultType': 'future'}})
    ohlcv = ex.fetch_ohlcv('BTC/USDT', '15m', limit=8000)
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    
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
    
    model_path = 'xgboost_btc_15m_v2_advanced.pkl'
    if not os.path.exists(model_path):
        model_path = 'ver_2/models/xgboost_btc_15m_v2_advanced.pkl'
    model = joblib.load(model_path)
    
    features = ['Returns', 'Body_Size', 'Upper_Shadow', 'Lower_Shadow', 
                'RSI_14', 'ATR_Ratio', 'Close_vs_SMA20', 'BB_Width', 'BB_Pos', 'Close_vs_SMA20_1H']
    X = df[features]
    probs = model.predict_proba(X)
    
    df['Max_Prob'] = np.max(probs, axis=1)
    df['Pred'] = np.argmax(probs, axis=1)
    return df.copy()

def evaluate_params(args):
    df, entry_th, exit_th, leverage, invest_ratio, max_pyramid, rsi_long_th, rsi_short_th = args
    
    initial_balance = 10000.0
    balance = initial_balance
    position = 0
    avg_entry_price = 0.0
    invested_margin = 0.0
    position_size = 0.0
    pyramid_count = 0
    fee_rate = 0.0004

    equity_curve = [initial_balance]
    total_trades = 0
    win_trades = 0

    prices = df['Close'].values
    rsis = df['RSI_14'].values * 100.0
    probs = df['Max_Prob'].values
    preds = df['Pred'].values

    for i in range(len(df)):
        close_price = prices[i]
        rsi = rsis[i]
        prob = probs[i]
        pred = preds[i]

        if balance <= 0:
            equity_curve.append(0)
            continue

        net_profit = 0
        if position != 0:
            price_change_pct = (close_price - avg_entry_price) / avg_entry_price * position
            net_profit = (position_size * price_change_pct) - (position_size * fee_rate * 2)

            if net_profit <= -invested_margin:
                balance -= invested_margin
                position, invested_margin, position_size, pyramid_count = 0, 0, 0, 0
                equity_curve.append(balance)
                continue

            # RSI 초과 청산
            if (position == 1 and rsi >= rsi_long_th) or (position == -1 and rsi <= rsi_short_th):
                balance += net_profit
                total_trades += 1
                if net_profit > 0: win_trades += 1
                position, invested_margin, position_size, pyramid_count = 0, 0, 0, 0
                equity_curve.append(max(balance, 0))
                continue

        is_loss = (position == 1 and close_price < avg_entry_price) or (position == -1 and close_price > avg_entry_price)

        if position == 0:
            if prob >= entry_th:
                if pred in [0, 2]:
                    position = 1 if pred == 2 else -1
                    avg_entry_price = close_price
                    invested_margin = balance * invest_ratio
                    position_size = invested_margin * leverage
                    pyramid_count = 0
                    total_trades += 1
        else:
            if (position == 1 and pred == 0) or (position == -1 and pred == 2):
                if prob >= exit_th:
                    balance += net_profit
                    if net_profit > 0: win_trades += 1
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

        equity_curve.append(max(balance + (net_profit if position != 0 else 0), 0))

    eq_arr = np.array(equity_curve)
    peak = np.maximum.accumulate(eq_arr)
    drawdown = (eq_arr - peak) / (peak + 1e-8)
    mdd = np.min(drawdown) * 100.0
    ret_pct = ((balance - initial_balance) / initial_balance) * 100.0
    win_rate = (win_trades / total_trades * 100.0) if total_trades > 0 else 0.0

    return {
        'entry_th': entry_th,
        'exit_th': exit_th,
        'leverage': leverage,
        'invest_ratio': invest_ratio,
        'max_pyramid': max_pyramid,
        'rsi_long_th': rsi_long_th,
        'rsi_short_th': rsi_short_th,
        'return_pct': ret_pct,
        'mdd': mdd,
        'trades': total_trades,
        'win_rate': win_rate,
        'score': ret_pct / (abs(mdd) + 1.0)
    }

def main():
    print(">>> 4,860개 그리드 조합 백테스트 검색 중...")
    df = get_data_and_predictions()
    
    entry_ths = [0.40, 0.42, 0.45, 0.48]
    exit_ths = [0.38, 0.40, 0.42, 0.45]
    leverages = [15, 20, 25]
    invest_ratios = [0.15, 0.20, 0.25]
    max_pyramids = [1, 2, 3]
    rsi_longs = [85, 90, 95]
    rsi_shorts = [5, 10, 15]

    param_combinations = list(itertools.product(
        [df], entry_ths, exit_ths, leverages, invest_ratios, max_pyramids, rsi_longs, rsi_shorts
    ))
    
    results = []
    with ProcessPoolExecutor() as executor:
        results = list(executor.map(evaluate_params, param_combinations))
        
    df_res = pd.DataFrame(results)
    
    # 1. 안정성 최우선 (위험 대비 수익률 Score)
    df_stable = df_res[df_res['mdd'] > -20].sort_values(by='score', ascending=False)
    
    # 2. 최고 수익률 최우선
    df_high_ret = df_res.sort_values(by='return_pct', ascending=False)
    
    print("\n==========================================")
    print("🥇 [추천 1] 안정적 최고 수익률 조합 (MDD < 20% 보호)")
    print("==========================================")
    best_st = df_stable.iloc[0]
    print(f"  • 진입 임계점: {best_st['entry_th']}")
    print(f"  • 청산 임계점: {best_st['exit_th']}")
    print(f"  • 레버리지: {best_st['leverage']}x")
    print(f"  • 1회 진입 비중: {int(best_st['invest_ratio']*100)}%")
    print(f"  • 최대 물타기 횟수: {int(best_st['max_pyramid'])}회")
    print(f"  • RSI 롱 청산: {int(best_st['rsi_long_th'])}, RSI 숏 청산: {int(best_st['rsi_short_th'])}")
    print(f"  📊 총 수익률: {best_st['return_pct']:+.2f}% | MDD: {best_st['mdd']:.2f}% | 승률: {best_st['win_rate']:.2f}% ({int(best_st['trades'])}회 매매)\n")

    print("==========================================")
    print("🚀 [추천 2] 공격적 최대 수익률 조합")
    print("==========================================")
    best_hr = df_high_ret.iloc[0]
    print(f"  • 진입 임계점: {best_hr['entry_th']}")
    print(f"  • 청산 임계점: {best_hr['exit_th']}")
    print(f"  • 레버리지: {best_hr['leverage']}x")
    print(f"  • 1회 진입 비중: {int(best_hr['invest_ratio']*100)}%")
    print(f"  • 최대 물타기 횟수: {int(best_hr['max_pyramid'])}회")
    print(f"  • RSI 롱 청산: {int(best_hr['rsi_long_th'])}, RSI 숏 청산: {int(best_hr['rsi_short_th'])}")
    print(f"  📊 총 수익률: {best_hr['return_pct']:+.2f}% | MDD: {best_hr['mdd']:.2f}% | 승률: {best_hr['win_rate']:.2f}% ({int(best_hr['trades'])}회 매매)\n")

if __name__ == '__main__':
    main()
