import os
import warnings
import logging

# 1. Muzzle TensorFlow's C++ backend (Kills the oneDNN and basic info logs)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0=all, 1=info, 2=warnings, 3=errors
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Now it's safe to import tensorflow and keras
import pandas as pd
import numpy as np
import xgboost as xgb
import tensorflow as tf
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, MinMaxScaler, StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from pandas.tseries.offsets import BDay

# 2. Muzzle Python warnings and TensorFlow's internal logger (Kills the deprecation warnings)
warnings.filterwarnings('ignore')
tf.get_logger().setLevel('ERROR')
logging.getLogger('tensorflow').setLevel(logging.ERROR)

# ==========================================
# 1. PRICE DATA CLEANING
# ==========================================
def clean_market_data(data):
    data = data.drop(columns=["ticker", "Unnamed: 0"], errors="ignore")
    
    if 'Date' in data.columns:
        data['Date'] = pd.to_datetime(data['Date'].astype(str).str[:10], errors='coerce')
        data = data.dropna(subset=['Date'])
        data = data.set_index('Date')
        
    data = data.sort_index()
    
    for col in data.columns:
        if data[col].dtype == "object":
            data[col] = data[col].str.strip().str.replace('.', '', regex=False).str.replace(',', '.', regex=False)
            
    data = data.apply(pd.to_numeric, errors="coerce")
    return data

# ==========================================
# 2A. PRICE FEATURE ENGINEERING
# ==========================================
def create_features(data):
    data = data.copy()
    data['ret_1d'] = data['Close'].pct_change(1)
    data['ret_3d'] = data['Close'].pct_change(3)
    data['ret_5d'] = data['Close'].pct_change(5)

    high_low = data['High'] - data['Low']
    high_close = np.abs(data['High'] - data['Close'].shift())
    low_close = np.abs(data['Low'] - data['Close'].shift())
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    data['ATR_14'] = true_range.rolling(14).mean()

    data['std_5'] = data['ret_1d'].rolling(5).std()
    
    delta = data['Close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / (avg_loss + 1e-8)
    data['RSI_14'] = 100 - (100 / (1 + rs))

    ema_12 = data['Close'].ewm(span=12, adjust=False).mean()
    ema_26 = data['Close'].ewm(span=26, adjust=False).mean()
    macd = ema_12 - ema_26
    signal = macd.ewm(span=9, adjust=False).mean()
    data['MACD_hist'] = macd - signal

    data['vol_avg_10'] = data['Volume'].rolling(10).mean()
    data['vol_ratio'] = data['Volume'] / (data['vol_avg_10'] + 1e-8)

    return data.dropna()

# ==========================================
# 2B. BROKER SUMMARY ENGINEERING (CAUSAL & 60-DAY CONSTRAINED)
# ==========================================
def parse_value(val):
    if pd.isna(val): return 0.0
    val_str = str(val).replace('"', '').replace(',', '').strip()
    if 'T' in val_str: return float(val_str.replace('T', '')) * 1e12
    elif 'B' in val_str: return float(val_str.replace('B', '')) * 1e9
    elif 'M' in val_str: return float(val_str.replace('M', '')) * 1e6
    elif 'K' in val_str: return float(val_str.replace('K', '')) * 1e3
    else:
        try: return float(val_str)
        except: return 0.0

def create_broker_features(broker_data):
    df = broker_data.copy()
    
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce', dayfirst=True)
        df = df.dropna(subset=['Date'])
    
    buyer_col, buy_val_col = 'BY', 'B.val'
    seller_col, sell_val_col = 'SL', 'S.val'
    
    if all(col in df.columns for col in [buyer_col, buy_val_col, seller_col, sell_val_col]):
        df[buy_val_col] = df[buy_val_col].apply(parse_value)
        df[sell_val_col] = df[sell_val_col].apply(parse_value)
        
        buy_df = df[['Date', buyer_col, buy_val_col]].rename(columns={buyer_col: 'Broker', buy_val_col: 'Volume'})
        buy_df['Action'] = 'Buy'
        
        sell_df = df[['Date', seller_col, sell_val_col]].rename(columns={seller_col: 'Broker', sell_val_col: 'Volume'})
        sell_df['Action'] = 'Sell'
        
        unified = pd.concat([buy_df, sell_df])
        unified = unified[unified['Volume'] > 0].dropna(subset=['Broker'])
        
        daily_net = unified.groupby(['Date', 'Broker', 'Action'])['Volume'].sum().unstack(fill_value=0).reset_index()
        if 'Buy' not in daily_net.columns: daily_net['Buy'] = 0
        if 'Sell' not in daily_net.columns: daily_net['Sell'] = 0
        
        daily_net['Net_Vol'] = daily_net['Buy'] - daily_net['Sell']
        daily_net['Total_Vol'] = daily_net['Buy'] + daily_net['Sell']
        
        # Matrix Pivots for fast rolling calculations
        pivot_total = daily_net.pivot(index='Date', columns='Broker', values='Total_Vol').fillna(0)
        pivot_net = daily_net.pivot(index='Date', columns='Broker', values='Net_Vol').fillna(0)
        
        # -------------------------------------------------------------
        # FIX 1: The 60-Day Memory Constraint & Lookahead Prevention
        # Shift(1) ensures today's label is based ONLY on yesterday's history
        # -------------------------------------------------------------
        roll_60_total = pivot_total.rolling(window=60, min_periods=1).sum().shift(1).fillna(0)
        roll_60_net = pivot_net.rolling(window=60, min_periods=1).sum().shift(1).fillna(0)
        
        records = []
        for date in pivot_total.index:
            hist_total = roll_60_total.loc[date]
            hist_net = roll_60_net.loc[date]
            
            # Dynamic classification: Who were the big players in the last 60 days?
            whales = hist_total.nlargest(10).index.tolist()
            accumulators = hist_net.nlargest(10).index.tolist()
            distributors = hist_net.nsmallest(10).index.tolist()
            
            today_data = daily_net[daily_net['Date'] == date]
            
            total_vol = today_data['Total_Vol'].sum()
            whale_net = today_data.loc[today_data['Broker'].isin(whales), 'Net_Vol'].sum()
            acc_net = today_data.loc[today_data['Broker'].isin(accumulators), 'Net_Vol'].sum()
            dist_net = today_data.loc[today_data['Broker'].isin(distributors), 'Net_Vol'].sum()
            top10_net = today_data.loc[today_data['Broker'].isin(whales), 'Net_Vol'].sum()
            
            records.append({
                'Date': date,
                'Total_Volume': total_vol,
                'Whale_Net_Vol': whale_net,
                'Accumulator_Net_Vol': acc_net,
                'Distributor_Net_Vol': dist_net,
                'Top10_Net_Vol': top10_net
            })
            
        df_agg = pd.DataFrame(records).set_index('Date')
        
        # -------------------------------------------------------------
        # FIX 2: Drop 120D/250D Noise, Keep Short-Term Reality
        # -------------------------------------------------------------
        df_agg['Top10_Inventory'] = df_agg['Top10_Net_Vol'].cumsum()
        df_agg['Top10_Net_5D'] = df_agg['Top10_Net_Vol'].rolling(window=5, min_periods=1).sum()
        df_agg['Top10_Net_20D'] = df_agg['Top10_Net_Vol'].rolling(window=20, min_periods=1).sum()
        df_agg['Top10_Net_60D'] = df_agg['Top10_Net_Vol'].rolling(window=60, min_periods=1).sum()
        
        df_agg['Whale_Impact'] = df_agg['Whale_Net_Vol'] / (df_agg['Total_Volume'] + 1e-8)
        
        return df_agg.dropna()
        
    return df.set_index('Date').dropna() if 'Date' in df.columns else df

# ==========================================
# 3. XGBOOST TARGET GENERATOR
# ==========================================
def create_target(data, lookahead=3):
    data["future_close"] = data["Close"].shift(-lookahead)
    data["future_return"] = (data["future_close"] - data["Close"]) / data["Close"]
    data["rolling_vol"] = data["Close"].pct_change().rolling(20).std()
    data["z_score"] = data["future_return"] / (data["rolling_vol"] + 1e-8)
    data["sentiment_target"] = 1 / (1 + np.exp(-data["z_score"]))
    return data

# ==========================================
# 4. LSTM SEQUENCE GENERATOR
# ==========================================
def create_sequences(X, y=None, time_steps=30):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        Xs.append(X[i:(i + time_steps)])
        if y is not None: ys.append(y[i + time_steps])
    if y is not None: return np.array(Xs), np.array(ys)
    return np.array(Xs)

# ==========================================
# 5. MASTER FORECAST ENGINE
# ==========================================
def run_dynamic_forecast(target_csv, broker_csv, lookahead_days=3, atr_multiplier=1.5):
    price_data = clean_market_data(pd.read_csv(target_csv))
    price_data = create_features(price_data)
    broker_data = create_broker_features(pd.read_csv(broker_csv))
    
    data = pd.merge(price_data, broker_data, left_index=True, right_index=True, how='inner')
    
    if len(data) < 5:
        raise ValueError(f"Dataset too small! Only {len(data)} valid days found.")
    
    data = create_target(data, lookahead=lookahead_days)
    
    raw_cols = ['Open','High','Low','Close','Volume','Dividends','Stock Splits']
    target_cols = ['future_close', 'future_return', 'rolling_vol', 'z_score', 'sentiment_target']
    
    train_xgb_clean = data.dropna(subset=['sentiment_target'])
    X_xgb = train_xgb_clean.drop(columns=raw_cols + target_cols, errors="ignore")
    y_xgb = train_xgb_clean['sentiment_target']
    
    # -------------------------------------------------------------
    # FIX 3: Out-of-Fold Predictions (Prevent Pipeline Leakage)
    # -------------------------------------------------------------
    xgb_pipeline = Pipeline([
        ("scaler", RobustScaler()),
        ("model", xgb.XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.05, random_state=42))
    ])

    oof_preds = pd.Series(index=X_xgb.index, dtype=float)
    tscv = TimeSeriesSplit(n_splits=5)
    
    for train_idx, test_idx in tscv.split(X_xgb):
        X_tr, y_tr = X_xgb.iloc[train_idx], y_xgb.iloc[train_idx]
        X_te = X_xgb.iloc[test_idx]
        xgb_pipeline.fit(X_tr, y_tr)
        oof_preds.iloc[test_idx] = xgb_pipeline.predict(X_te)
        
    first_fold_len = len(X_xgb) - len(oof_preds.dropna())
    if first_fold_len > 0:
        X_first = X_xgb.iloc[:first_fold_len]
        xgb_pipeline.fit(X_first, y_xgb.iloc[:first_fold_len])
        oof_preds.iloc[:first_fold_len] = xgb_pipeline.predict(X_first)

    data['sentiment_likelihood'] = pd.Series(dtype=float)
    data.loc[train_xgb_clean.index, 'sentiment_likelihood'] = oof_preds
    
    # Train final XGBoost for today's forecast
    xgb_pipeline.fit(X_xgb, y_xgb)
    X_pred_all = data.drop(columns=raw_cols + target_cols + ['sentiment_likelihood'], errors="ignore")
    data.loc[data.index[-1], 'sentiment_likelihood'] = xgb_pipeline.predict(X_pred_all.iloc[[-1]])[0]

    # -------------------------------------------------------------
    # LSTM Setup (Updated Features)
    # -------------------------------------------------------------
    lstm_features = ['ret_1d', 'ret_3d', 'ATR_14', 'std_5', 'RSI_14', 
                     'MACD_hist', 'vol_ratio', 'Whale_Impact', 
                     'Whale_Net_Vol', 'Accumulator_Net_Vol', 'Distributor_Net_Vol',
                     'Top10_Inventory', 'Top10_Net_5D', 'Top10_Net_20D', 
                     'Top10_Net_60D', 'sentiment_likelihood']

    data['target_entry'] = (data['Close'].shift(-lookahead_days) - data['Close']) / data['Close']
    data['target_exit']  = (data['High'].rolling(lookahead_days).max().shift(-lookahead_days) - data['Close']) / data['Close']
    data['target_stop']  = (data['Low'].rolling(lookahead_days).min().shift(-lookahead_days) - data['Close']) / data['Close']

    train_lstm_clean = data.dropna(subset=lstm_features + ['target_entry', 'target_exit', 'target_stop'])
    pred_data = data.dropna(subset=lstm_features) 

    scaler_X = MinMaxScaler()
    scaler_y = StandardScaler()

    X_train_scaled = scaler_X.fit_transform(train_lstm_clean[lstm_features])
    y_train_scaled = scaler_y.fit_transform(train_lstm_clean[['target_entry', 'target_exit', 'target_stop']])

    available_rows = len(train_lstm_clean)
    TIME_STEPS = min(30, max(3, int(available_rows * 0.5)))

    X_seq, y_seq = create_sequences(X_train_scaled, y_train_scaled, TIME_STEPS)

    if len(X_seq) == 0:
        raise ValueError("Not enough sequences. Upload a larger Broker Summary (3+ months) or reduce forecast horizon.")

    lstm_model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(TIME_STEPS, len(lstm_features))),
        Dropout(0.2),
        LSTM(32, return_sequences=False),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(3, activation='linear')
    ])
    
    lstm_model.compile(optimizer='adam', loss='huber')
    lstm_model.fit(X_seq, y_seq, epochs=15, batch_size=32, validation_split=0.1 if len(X_seq) >= 10 else 0.0, verbose=0)

    last_sequence_raw = pred_data[lstm_features].tail(TIME_STEPS)
    last_sequence_scaled = scaler_X.transform(last_sequence_raw)
    X_final = last_sequence_scaled.reshape(1, TIME_STEPS, len(lstm_features))
    
    prediction_scaled = lstm_model.predict(X_final, verbose=0)
    predicted_returns = scaler_y.inverse_transform(prediction_scaled)[0]

    # -------------------------------------------------------------
    # FIX 4: Use LSTM's Native Target Exits & Stops
    # -------------------------------------------------------------
    last_close = pred_data['Close'].iloc[-1]
    last_sentiment = pred_data['sentiment_likelihood'].iloc[-1]
    last_atr = pred_data['ATR_14'].iloc[-1]
    forecast_date = pred_data.index[-1] + BDay(lookahead_days)

    pred_entry = last_close * (1 + predicted_returns[0])
    lstm_exit_target = last_close * (1 + predicted_returns[1])
    lstm_stop_target = last_close * (1 + predicted_returns[2])

    min_buffer = last_atr * 0.5
    pred_exit = max(lstm_exit_target, pred_entry + min_buffer)
    pred_stop = min(lstm_stop_target, pred_entry - min_buffer)

    buffer = last_atr * 0.2 
    results_df = pd.DataFrame([{
        'Forecast_Date': forecast_date.strftime('%Y-%m-%d'),
        'Sentiment_Score': round(last_sentiment, 2),
        'Target_Buy': f"Rp {int(pred_entry):,} (Range: {int(pred_entry - buffer):,} - {int(pred_entry + buffer):,})",
        'Take_Profit': f"Rp {int(pred_exit):,} (Range: {int(pred_exit - buffer):,} - {int(pred_exit + buffer):,})",
        'Stop_Loss': f"Rp {int(pred_stop):,} (Range: {int(pred_stop - buffer):,} - {int(pred_stop + buffer):,})"
    }])
    
    return results_df, last_sentiment