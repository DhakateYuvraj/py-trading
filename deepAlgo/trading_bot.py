import asyncio
import aiohttp
import os
import hashlib
import json
import requests
import logging
import signal
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, time
from scipy.stats import norm
import csv
from pathlib import Path
from dateutil.relativedelta import relativedelta
from collections import defaultdict

CONFIG = {
    "api_key": "f4360a71362621a96cc71af4b6f63f35",
    "app_key": "c185e6a0c159a7c6a59d32ba65bc45d1c940df9130f28d8d2eb55af14b1b3980",
    "user_id": "FN105570",
    "password": "Yuvi@1989",  # Replace with your plain text password
    "totp_key": "559030",
    "vendor_code": "FN105570_U",
    "imei": "1234567890",
    "base_url": "https://api.shoonya.com/NorenWClientTP/",
    "strike_interval": 50,
    "nifty_token": "26000",
    "websocket_url": "wss://api.shoonya.com/NorenWSTP/",
    "position_size": 1.0,
    "max_drawdown": 2.0,
    "max_loss_per_trade": 0.02,
    "min_holding_period": 5  # Minimum minutes between signals for same token
}

OPTIONS = [
    {"name": "Nifty", "value": "NIFTY", "token": "26000", "exch": "NSE",  "weeklyLot": 75, "monthlyLot": 25, "strike_interval": 50},
    {"name": "BankNifty", "value": "BANKNIFTY", "token": "26009", "exch": "NSE", "weeklyLot": 30, "monthlyLot": 15, "strike_interval": 100},
]

# API Endpoints
LOGIN_URL = CONFIG["base_url"] + "QuickAuth"
GET_QUOTES_URL = CONFIG["base_url"] + "GetQuotes"
TPSERIES_URL = CONFIG["base_url"] + "TPSeries"

# At the top of your code, define a global start time
ALGO_START_TIME = datetime.now()
STARTING_WARMUP_PERIOD = 3      # in min

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("trading_bot.log"), logging.StreamHandler()]
)

class RiskManager:
    def __init__(self, max_drawdown):
        self.max_drawdown = max_drawdown
        self.trade_log = []
        self.daily_pnl = 0.0
        
    def update_trade_log(self, pnl):
        self.trade_log.append({
            'timestamp': datetime.now().isoformat(),
            'pnl': pnl
        })
        self.daily_pnl += pnl

class PositionManager:
    def __init__(self):
        self.positions = defaultdict(dict)
        self.last_signal_time = defaultdict(lambda: datetime.min)
    
    def has_open_position(self, token):
        return bool(self.positions.get(token))
    
    def open_position(self, token, signal_type, price):
        self.positions[token] = {
            'direction': signal_type,
            'entry_price': price,
            'entry_time': datetime.now()
        }
    
    def close_position(self, token, exit_price):
        if token in self.positions:
            pos = self.positions.pop(token)
            pos['exit_price'] = exit_price
            pos['duration'] = datetime.now() - pos['entry_time']
            return pos
        return None

def calculate_ema(data, period):
    return data["close"].ewm(span=period, adjust=False).mean()

def calculate_atr(df, period=14):
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    df['tr'] = np.maximum.reduce([high_low, high_close, low_close])
    return df['tr'].rolling(period).mean().iloc[-1]

def calculate_vwap(df):
    tp = (df['high'] + df['low'] + df['close']) / 3
    cum_vol = df['volume'].cumsum()
    cum_tp_vol = (tp * df['volume']).cumsum()
    return cum_tp_vol / cum_vol

def calculate_rsi(df, period=14):
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def process_candle_data(raw_data):
    if not raw_data or not isinstance(raw_data, list):
        return pd.DataFrame()
    df = pd.DataFrame([{
            'time': str(candle.get('time', '')),
            'open': float(candle.get('into', 0.0)),
            'high': float(candle.get('inth', 0.0)),
            'low': float(candle.get('intl', 0.0)),
            'close': float(candle.get('intc', 0.0)),
            'volume': float(candle.get('v', 0.0))
        } for candle in raw_data if isinstance(candle, dict)])
    if df.empty:
        return df
    df['time'] = pd.to_datetime(df['time'], format='%d-%m-%Y %H:%M:%S', errors='coerce')
    df = df.dropna(subset=['time', 'close']).sort_values('time').reset_index(drop=True)
    if len(df) > 20:
        df['vwap'] = calculate_vwap(df)
        df['ema_20'] = calculate_ema(df, 20)
        df['ema_50'] = calculate_ema(df, 50)
        df['atr'] = calculate_atr(df)
        df['rsi'] = calculate_rsi(df) if len(df) >= 14 else np.nan
    return df

def check_breakout_strategy(one_min_df, three_min_df):
    if 'rsi' not in three_min_df.columns or three_min_df['rsi'].isnull().all():
        return False, None
    current_rsi = three_min_df['rsi'].iloc[-1]
    if current_rsi > 70 or current_rsi < 30:
        return False, None
    if len(one_min_df) < 15:
        return False, None
    consolidation = one_min_df.iloc[-15:-1]
    if (consolidation['high'].max() - consolidation['low'].min()) / consolidation['close'].mean() > 0.005:
        return False, None
    if one_min_df['volume'].iloc[-1] < 1.5 * consolidation['volume'].mean():
        return False, None
    if three_min_df['close'].iloc[-1] < three_min_df['vwap'].iloc[-1]:
        return False, None
    if three_min_df['rsi'].iloc[-1] > 70 or three_min_df['rsi'].iloc[-1] < 30:
        return False, None
    return True, {
        'price': three_min_df['close'].iloc[-1],
        'timestamp': three_min_df['time'].iloc[-1].isoformat()
    }
'''
def check_mean_reversion_strategy(df):
    if len(df) < 50:
        return False, None
    current = df.iloc[-1]
    previous = df.iloc[-2]
    if current['ema_20'] < current['ema_50'] and previous['ema_20'] >= previous['ema_50']:
        return True, {'type': 'BUY', 'price': current['close']}
    elif current['ema_20'] > current['ema_50'] and previous['ema_20'] <= previous['ema_50']:
        return True, {'type': 'SELL', 'price': current['close']}
    return False, None
'''

def check_mean_reversion_strategy(df_1min, df_3min, df_5min):
    # Existing EMA crossover condition (using 3-min data)
    if len(df_3min) < 50:
        return False, None
    current = df_3min.iloc[-1]
    previous = df_3min.iloc[-2]
    
    if current['ema_20'] < current['ema_50'] and previous['ema_20'] >= previous['ema_50']:
        return True, {'type': 'BUY', 'price': current['close']}
    elif current['ema_20'] > current['ema_50'] and previous['ema_20'] <= previous['ema_50']:
        return True, {'type': 'SELL', 'price': current['close']}
    
    # New Condition 3: Mean Reversion via RSI Divergence (Sell Premium)
    # Ensure enough data exists in each timeframe
    if len(df_5min) >= 14 and len(df_1min) >= 2 and len(df_3min) >= 2:
        # Calculate the 5-min RSI(14)
        rsi_5min_series = calculate_rsi(df_5min, period=14)
        rsi_5min = rsi_5min_series.iloc[-1]
        
        # Check for an extreme 5-min RSI reading
        if rsi_5min > 75 or rsi_5min < 25:
            # Calculate the 1-min RSI(14)
            rsi_1min_series = calculate_rsi(df_1min, period=14)
            rsi_1min = rsi_1min_series.iloc[-1]
            
            # Check for divergence on the 1-min price:
            # For a bearish scenario: price makes a new high but RSI does not confirm (rsi_1min is lower than expected).
            # For a bullish scenario: price makes a new low but RSI does not confirm.
            new_high = df_1min['close'].iloc[-1] >= df_1min['close'].max()
            new_low = df_1min['close'].iloc[-1] <= df_1min['close'].min()
            
            # You might want to compare the latest 1-min RSI with the previous 1-min value:
            prev_rsi_1min = rsi_1min_series.iloc[-2]
            
            # Check if there's a divergence (using a simple comparison as an example)
            bearish_divergence = (new_high and (rsi_1min < prev_rsi_1min))
            bullish_divergence = (new_low and (rsi_1min > prev_rsi_1min))
            
            # Check the 3-min volume trend (declining)
            volume_declining = df_3min['volume'].iloc[-1] < df_3min['volume'].iloc[-2]
            
            if (rsi_5min > 75 and bearish_divergence and volume_declining) or \
               (rsi_5min < 25 and bullish_divergence and volume_declining):
                # Signal to Sell OTM iron condor (for example, a SELL_PREMIUM signal)
                return True, {'type': 'SELL', 'price': current['close']}
    return False, None

    
def calculate_strike_price(spot, token):
    if "NIFTY" in token:
        interval = 50
        strike = round(spot / interval) * interval
    else:  # BANKNIFTY
        interval = 100
        strike = (int(spot) // interval) * interval
        if (spot - strike) >= interval / 2:
            strike += interval
    return int(strike)

def generate_symbol(token, expiry, strike, option_type):
    expiry_str = expiry.strftime("%d%b%y").upper()
    return f"{token} {expiry_str} {strike} {'CE' if option_type == 'CALL' else 'PE'}"

def get_expiry_date(signal_type, token):
    today = datetime.today()
    if "NIFTY" in token:
        days_to_thursday = (3 - today.weekday()) % 7
        if days_to_thursday == 0 and today.hour > 15:
            days_to_thursday = 7
        expiry_date = today + timedelta(days=days_to_thursday)
    else:  # BANKNIFTY: monthly expiry (last Thursday)
        last_day = today + relativedelta(day=31)
        while last_day.weekday() != 3:
            last_day -= timedelta(days=1)
        expiry_date = last_day
    return expiry_date

def log_signal_to_csv(token, signal_data):
    try:
        today = signal_data['timestamp'].strftime("%Y-%m-%d")
        filename = f"data/{today}/{token}_trades.csv"
        Path(f"data/{today}").mkdir(parents=True, exist_ok=True)
        file_exists = Path(filename).exists()
        with open(filename, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=signal_data.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(signal_data)
    except Exception as e:
        logging.error(f"CSV logging failed: {str(e)}")


async def execute_trade(session, signal, risk_manager, token, is_exit=False):
    try:
        spot_price = signal['price']
        if is_exit:
            option_type = 'PE' if 'SHORT' in signal['signal_type'] else 'CE'
            action = "Exit"
        else:
            option_type = 'CE' if 'BUY' in signal['signal_type'] or 'LONG' in signal['signal_type'] else 'PE'
            action = "Entry"
        strike = calculate_strike_price(spot_price, token)
        expiry_date = get_expiry_date(signal['signal_type'], token)
        symbol = generate_symbol(token, expiry_date, strike, option_type)
        
        csv_data = {
            'timestamp': signal['timestamp'],
            'symbol': symbol,
            'action': action,
            'type': option_type,
            'strike': strike,
            'expiry': expiry_date.strftime("%d-%b-%Y"),
            'spot_price': spot_price,
            'rsi': signal['rsi'],
            'vwap': signal['vwap'],
            'atr': signal['atr'],
            'ema_20': signal['ema_20'],
            'ema_50': signal['ema_50']
        }
        if is_exit:
            csv_data.update({
                'pnl': f"{signal.get('pnl', 0):.2f}%",
                'duration': f"{signal.get('duration_min', 0):.1f} min"
            })
        log_signal_to_csv(token, csv_data)
        
        print(f"\n{'-'*40}")
        print(f"Executing {action} Order:")
        print(f"Token: {token}")
        print(f"Symbol: {symbol}")
        print(f"Strike: {strike}")
        print(f"Expiry: {expiry_date.strftime('%d-%b-%Y')}")
        print(f"Spot Price: {spot_price:.2f}")
        print(f"RSI: {signal['rsi']:.2f}")
        print(f"VWAP: {signal['vwap']:.2f}")
        if is_exit:
            print(f"P&L: {csv_data['pnl']}")
            print(f"Duration: {csv_data['duration']}")
        print(f"{'-'*40}")
        
        if is_exit and 'pnl' in signal:
            risk_manager.update_trade_log(signal['pnl'])
    except Exception as e:
        logging.error(f"Trade execution failed ({token}): {str(e)}")
        raise

def read_token_from_file():
    try:
        with open("tokens.json", "r") as file:
            tokens = json.load(file)
            return tokens.get("susertoken"), tokens.get("susertokenspl"), tokens.get("date")
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logging.error(f"Error reading token file: {e}")
        return None, None, None

async def fetch_historical_data(session, token, interval, exch):
    try:
        now = datetime.now()
        start_time = now - timedelta(hours=100)
        jData = {
            "uid": CONFIG["user_id"],
            "exch": exch,
            "token": str(token),
            "st": str(int(start_time.timestamp())),
            "et": str(int(now.timestamp())),
            "intrv": str(interval),
        }
        susertoken, _, _ = read_token_from_file()
        form_data = f"jData={json.dumps(jData)}&jKey={susertoken}"
        timeout = aiohttp.ClientTimeout(total=30)
        async with session.post(TPSERIES_URL, data=form_data, timeout=timeout) as response:
            if response.status != 200:
                logging.error(f"API Error: {response.status}")
                return None
            raw_data = await response.json()
            if isinstance(raw_data, dict):
                if raw_data.get('stat') != 'Ok':
                    logging.error(f"API Error: {raw_data.get('emsg')}")
                    return None
                return raw_data.get('data', [])
            elif isinstance(raw_data, list):
                return raw_data
            else:
                logging.error(f"Unexpected response format: {type(raw_data)}")
                return None
    except Exception as e:
        logging.error(f"Data fetch error: {str(e)}")
        return None


async def trading_cycle(session, token, exch, risk_manager, position_manager):
    try:
        # Wait for at least 5 minutes from algorithm start before generating any signals
        if datetime.now() - ALGO_START_TIME < timedelta(minutes=STARTING_WARMUP_PERIOD):
            logging.info(f"{token}: Waiting for initial {STARTING_WARMUP_PERIOD}-minute period before generating signals.")
            return

        last_signal_time = position_manager.last_signal_time.get(token, datetime.min)
        if (datetime.now() - last_signal_time).total_seconds() / 60 < CONFIG["min_holding_period"]:
            logging.info(f"{token}: Cooldown period active")
            return

        one_min_data = await fetch_historical_data(session, token, "1", exch)
        three_min_data = await fetch_historical_data(session, token, "3", exch)
        if not one_min_data or not three_min_data:
            logging.warning(f"{token}: Missing data")
            return

        one_min_df = process_candle_data(one_min_data)
        three_min_df = process_candle_data(three_min_data)
        if one_min_df.empty or three_min_df.empty:
            logging.warning(f"{token}: Empty data after processing")
            return

        current_close = three_min_df['close'].iloc[-1]
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(f"\n{one_min_df['time'].iloc[-1]} {token} Prices:")
        print(f"1min Close: {one_min_df['close'].iloc[-1]:.2f}")
        print(f"3min Close: {current_close:.2f}")
        print(f"RSI: {three_min_df['rsi'].iloc[-1]:.2f}")
        print(f"VWAP: {three_min_df['vwap'].iloc[-1]:.2f}")

        breakout_signal, breakout_details = check_breakout_strategy(one_min_df, three_min_df)
        mean_rev_signal, mean_rev_details = check_mean_reversion_strategy(three_min_df)
        
        signal_type = ""
        action = None
        trade_price = current_close
        is_exit = False
        
        if position_manager.has_open_position(token):
            current_position = position_manager.positions[token]
            current_direction = current_position['direction']
            exit_condition = False
            if breakout_signal:
                new_direction = "LONG" if breakout_details['price'] > three_min_df['vwap'].iloc[-1] else "SHORT"
                exit_condition = ("LONG" in current_direction and "SHORT" in new_direction) or \
                                 ("SHORT" in current_direction and "LONG" in new_direction)
            if mean_rev_signal:
                new_direction = mean_rev_details['type'].upper()
                exit_condition = ("LONG" in current_direction and "SELL" in new_direction) or \
                                 ("SHORT" in current_direction and "BUY" in new_direction)
            if exit_condition:
                action = 'exit'
                signal_type = f"EXIT {current_direction}"
                is_exit = True
                position = position_manager.close_position(token, trade_price)
                pnl = ((trade_price - position['entry_price']) / position['entry_price']) * 100
                if "SHORT" in current_direction:
                    pnl *= -1
        else:
            if breakout_signal:
                action = 'entry'
                signal_type = "BREAKOUT LONG" if breakout_details['price'] > three_min_df['vwap'].iloc[-1] else "BREAKOUT SHORT"
                trade_price = breakout_details['price']
            elif mean_rev_signal:
                action = 'entry'
                signal_type = mean_rev_details['type']
                trade_price = mean_rev_details['price']

        if action:
            signal_data = {
                'timestamp': current_time,
                'action': action,
                'signal_type': signal_type,
                'price': trade_price,
                'rsi': three_min_df['rsi'].iloc[-1],
                'vwap': three_min_df['vwap'].iloc[-1],
                'atr': three_min_df['atr'].iloc[-1],
                'ema_20': three_min_df['ema_20'].iloc[-1],
                'ema_50': three_min_df['ema_50'].iloc[-1]
            }
            if action == 'exit':
                signal_data.update({
                    'pnl': pnl,
                    'duration_min': (datetime.now() - position['entry_time']).total_seconds() / 60
                })
            await execute_trade(session, signal_data, risk_manager, token, is_exit)
            if action == 'entry':
                position_manager.open_position(token, signal_type, trade_price)
            position_manager.last_signal_time[token] = datetime.now()
            print(f"\n{token}: {action.upper()} signal - {signal_type}")
        else:
            print(f"{token}: No valid signals detected")
    except Exception as e:
        logging.error(f"Trading cycle error ({token}): {str(e)}")

def sha256_encode(password):
    return hashlib.sha256(password.encode()).hexdigest()



def save_token_to_file(susertoken, susertokenspl):
    try:
        today_date = datetime.today().strftime("%Y-%m-%d")
        token_data = {
            "susertoken": susertoken,
            "susertokenspl": susertokenspl,
            "date": today_date,
        }
        with open("tokens.json", "w") as file:
            json.dump(token_data, file)
    except Exception as e:
        logging.error(f"Error saving token: {e}")


def login():
    logging.info("Logging in...")
    hashed_password = sha256_encode(CONFIG["password"])
    hashed_app_key = sha256_encode(CONFIG["user_id"] + "|" + CONFIG["api_key"])
    totp_key = input("Enter your TOTP: ")
    jData = {
        "apkversion": "1.0.0",
        "uid": CONFIG["user_id"],
        "pwd": hashed_password,
        "appkey": hashed_app_key,
        "factor2": totp_key,
        "imei": CONFIG["imei"],
        "source": "API",
        "vc": CONFIG["vendor_code"],
    }
    payload = f"jData={json.dumps(jData)}"
    try:
        response = requests.post(LOGIN_URL, data=payload)
        response.raise_for_status()
        data = response.json()
        if data.get("stat") == "Ok":
            susertoken = data.get("susertoken")
            susertokenspl = data.get("susertokenspl")
            if susertoken and susertokenspl:
                save_token_to_file(susertoken, susertokenspl)
                return susertoken, susertokenspl
            else:
                logging.error("Login failed: Missing tokens.")
                return None, None
        else:
            logging.error(f"Login failed: {data.get('emsg')}")
            return None, None
    except requests.exceptions.RequestException as e:
        logging.error(f"Error logging in: {e}")
        return None, None



def check_token_file_exists():
    return os.path.exists("tokens.json")

async def main():
    risk_manager = RiskManager(CONFIG["max_drawdown"])
    position_manager = PositionManager()

    if check_token_file_exists():
        today_date = datetime.today().strftime("%Y-%m-%d")
        susertoken, susertokenspl, date = read_token_from_file()
        if susertoken and date == today_date:
            logging.info(f"Using existing token: {susertoken}")
        else:
            logging.info("Token is missing or outdated. Logging in...")
            susertoken, susertokenspl = login()
    else:
        logging.info("Token file not found. Logging in...")
        susertoken, susertokenspl = login()

    if not susertoken:
        logging.error("Login failed. Exiting.")
        return

    async with aiohttp.ClientSession() as session:
        while True:
            try:
                now_time = datetime.now().time()
                if not (time(9, 25) < now_time < time(2, 58)):
                    logging.info(f"Outside trading hours, sleeping for {STARTING_WARMUP_PERIOD} minutes.")
                    await asyncio.sleep(STARTING_WARMUP_PERIOD) # 60 * 5
                else:
                    tasks = [
                        trading_cycle(session, opt['token'], opt['exch'], risk_manager, position_manager)
                        for opt in OPTIONS
                    ]
                    await asyncio.gather(*tasks)
                    await asyncio.sleep(60 - datetime.now().second)
            except KeyboardInterrupt:
                logging.info("Shutting down gracefully...")
                break
            except Exception as e:
                logging.error(f"Main loop error: {e}")
                await asyncio.sleep(60)

if __name__ == "__main__":
    # Handle graceful shutdown
    signal.signal(signal.SIGINT, lambda s, f: sys.exit(0))
    signal.signal(signal.SIGTERM, lambda s, f: sys.exit(0))
    asyncio.run(main())
