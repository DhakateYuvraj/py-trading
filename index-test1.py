import asyncio
import aiohttp
import os
import hashlib
import json
import requests
import certifi
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
    "min_holding_period" : 5  # Minimum minutes between signals for same token
}

OPTIONS = [
    {"name": "Nifty", "value": "NIFTY", "token": "26000", "exch": "NSE", 
     "weeklyLot": 75, "monthlyLot": 25, "strike_interval": 50},
    {"name": "BankNifty", "value": "BANKNIFTY", "token": "26009", "exch": "NSE",
     "weeklyLot": 30, "monthlyLot": 15, "strike_interval": 100},
]


# API Endpoints
LOGIN_URL = CONFIG["base_url"] + "QuickAuth"
GET_QUOTES_URL = CONFIG["base_url"] + "GetQuotes"
TPSERIES_URL = CONFIG["base_url"] + "TPSeries"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("trading_bot.log"),
        logging.StreamHandler()
    ]
)

# -----------------------------
# Technical Analysis Functions
# -----------------------------
class RiskManager:
    def __init__(self, max_drawdown=2.0):
        self.max_drawdown = max_drawdown
        self.trade_log = []
        self.daily_pnl = 0.0
        
    def validate_trade(self, signal_type, price):
        if len(self.trade_log) >= 5:
            loss_count = sum(1 for t in self.trade_log[-5:] if t['pnl'] < 0)
            if loss_count >= 4:
                logging.error("5-trade loss streak detected. Trading paused.")
                return False
                
        if abs(self.daily_pnl) >= self.max_drawdown:
            logging.error(f"Daily drawdown limit reached: {self.daily_pnl}%")
            return False
            
        return True

    def update_trade_log(self, pnl):
        self.trade_log.append({
            'timestamp': datetime.now().isoformat(),
            'pnl': pnl,
            'drawdown': abs(pnl)/self.max_drawdown
        })
        self.daily_pnl += pnl


class PositionManager:
    def __init__(self):
        self.positions = defaultdict(dict)
        self.last_signal_time = defaultdict(lambda: datetime.min)  # Fix here
        
    def has_open_position(self, token):
        return bool(self.positions.get(token))
    
    def get_position_direction(self, token):
        return self.positions[token].get('direction')
    
    def open_position(self, token, signal_type, price):
        self.positions[token] = {
            'direction': signal_type,
            'entry_price': price,
            'entry_time': datetime.now()
        }
        self.last_signal_time[token] = datetime.now()
        
    def close_position(self, token, exit_price):
        if token in self.positions:
            position = self.positions.pop(token)
            position['exit_price'] = exit_price
            position['duration'] = datetime.now() - position['entry_time']
            return position
        return None

def calculate_ema(data, period):
    return data["close"].ewm(span=period, adjust=False).mean()

def calculate_atr(df, period=14):
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    df['tr'] = np.maximum(high_low, high_close, low_close)
    return df['tr'].rolling(period).mean().iloc[-1]

def detect_rsi_divergence(df, period=14, lookback=5):
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    
    rs = avg_gain / avg_loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    price_high = df['high'].rolling(lookback).max().iloc[-1]
    rsi_high = df['rsi'].rolling(lookback).max().iloc[-1]
    
    return (price_high > df['high'].iloc[-lookback-1]) and (rsi_high < df['rsi'].iloc[-lookback-1])

def calculate_vwap(df):
    
    tp = (df['high'] + df['low'] + df['close']) / 3
    cum_vol = df['volume'].cumsum()
    cum_tp_vol = (tp * df['volume']).cumsum()
    return cum_tp_vol / cum_vol

def option_greek_estimator(spot, strike, iv, days_to_expiry):
    t = days_to_expiry / 365.0
    d1 = (np.log(spot/strike) + (0.04 + (iv**2)/2 * t)) / (iv * np.sqrt(t))
    delta = norm.cdf(d1)
    theta = (-(spot * norm.pdf(d1) * iv) / (2 * np.sqrt(t))) / 365
    return delta, theta

def select_strike(signal_type, spot, iv, expiry_minutes=15):
    expiry_days = expiry_minutes / (60 * 24)
    strikes = np.arange(spot * 0.98, spot * 1.02, 50)
    
    for strike in strikes:
        delta, theta = option_greek_estimator(spot, strike, iv, expiry_days)
        if signal_type == "BREAKOUT":
            if 0.25 < delta < 0.35 and theta > -0.05:
                return strike
        elif signal_type == "MEAN_REVERSION":
            if delta < 0.2 and theta > -0.02:
                return strike
    return None



def read_token_from_file():
    try:
        with open("tokens.json", "r") as file:
            tokens = json.load(file)
            return tokens.get("susertoken"), tokens.get("susertokenspl"), tokens.get("date")
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logging.error(f"Error reading token file: {e}")
        return None, None, None

# -----------------------------
# Data Processing Functions
# -----------------------------
async def fetch_historical_data(session, token, interval, exch):
    """Fetch historical data with robust type validation"""
    try:
        now = datetime.now() - timedelta(hours=6)
        start_time = now - timedelta(hours=106)
        st = int(start_time.timestamp())
        et = int(now.timestamp())
        jData = {
            "uid": CONFIG["user_id"],
            "exch": exch,
            "token": str(token),
            "st": str(st),
            "et": str(et),
            "intrv": str(interval),
        }

        #logging.info(f"Fetching data with parameters: {jData}")
        susertoken, _, _ = read_token_from_file()
        form_data = f"jData={json.dumps(jData)}&jKey={susertoken}"
        timeout = aiohttp.ClientTimeout(total=30)
        
        async with session.post(TPSERIES_URL, data=form_data, timeout=timeout) as response:
            
            # Validate response status
            if response.status != 200:
                logging.error(f"API Error 1 : {response.status}")
                return None
                
            # Parse response with type validation
            raw_data = await response.json()
            
            # Handle different API response formats
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
                
    except json.JSONDecodeError:
        logging.error("Invalid JSON response")
        return None
    except Exception as e:
        logging.error(f"Data fetch error: {str(e)}")
        return None

# -----------------------------
# Technical Indicators
# -----------------------------
def calculate_rsi(df, period=14):
    """Calculate Relative Strength Index (RSI)"""
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
    
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def process_candle_data(raw_data):
    """Process candle data with strict type enforcement"""
    try:
        if not raw_data or not isinstance(raw_data, list):
            return pd.DataFrame()
            
        # Create DataFrame with forced string conversion
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
            
        # Convert timestamp with validation
        df['time'] = pd.to_datetime(
            df['time'],
            format='%d-%m-%Y %H:%M:%S', 
            errors='coerce'
        )
        
        # Drop invalid timestamps
        df = df.dropna(subset=['time'])
        
        # Sort and calculate indicators
        df = df.sort_values('time').reset_index(drop=True)
        if len(df) > 20:
            df['vwap'] = calculate_vwap(df)
            df['ema_20'] = calculate_ema(df, 20)
            df['ema_50'] = calculate_ema(df, 50)
            df['atr'] = calculate_atr(df)
                        
            # Add RSI calculation
            if len(df) >= 14:
                df['rsi'] = calculate_rsi(df)
            else:
                df['rsi'] = np.nan

        return df
        
    except Exception as e:
        logging.error(f"Data processing error: {str(e)}")
        return pd.DataFrame()
    
# -----------------------------
# Trading Strategies
# -----------------------------
def check_breakout_strategy(one_min_df, three_min_df):
        # Add RSI validation
    if 'rsi' not in three_min_df.columns or three_min_df['rsi'].isnull().all():
        logging.warning("Missing RSI data")
        return False, None
        
    # Modified RSI check
    current_rsi = three_min_df['rsi'].iloc[-1]
    if current_rsi > 70 or current_rsi < 30:
        return False, None  # Only trade in neutral RSI ranges
    # 1-minute breakout detection
    if len(one_min_df) < 15:
        return False, None
        
    consolidation = one_min_df.iloc[-15:-1]
    if (consolidation['high'].max() - consolidation['low'].min()) / consolidation['close'].mean() > 0.005:
        return False, None
        
    if one_min_df['volume'].iloc[-1] < 1.5 * consolidation['volume'].mean():
        return False, None
        
    # 3-minute confirmation
    if three_min_df['close'].iloc[-1] < three_min_df['vwap'].iloc[-1]:
        return False, None
        
    if three_min_df['rsi'].iloc[-1] > 70 or three_min_df['rsi'].iloc[-1] < 30:
        return False, None
        
    return True, {
        'price': three_min_df['close'].iloc[-1],
        'timestamp': three_min_df['time'].iloc[-1].isoformat()
    }

def check_mean_reversion_strategy(df):
    if len(df) < 50:
        return False, None
        
    current = df.iloc[-1]
    previous = df.iloc[-2]
    
    # EMA crossover check
    if current['ema_20'] < current['ema_50'] and previous['ema_20'] >= previous['ema_50']:
        return True, {'type': 'BUY', 'price': current['close']}
    elif current['ema_20'] > current['ema_50'] and previous['ema_20'] <= previous['ema_50']:
        return True, {'type': 'SELL', 'price': current['close']}
    
    return False, None

def calculate_strike_price(spot_price, token):
    """Calculate nearest strike price based on index rules"""
    if "NIFTY" in token:
        interval = 50
        strike = round(spot_price / interval) * interval
    else:  # BANKNIFTY
        interval = 100
        strike = (int(spot_price) // interval) * interval
        if (spot_price - strike) >= interval/2:
            strike += interval
    return int(strike)

def generate_symbol(token, expiry, strike, option_type):
    """Generate option symbol in NSE format"""
    # Example: NIFTY 06MAR25 17500 CE
    expiry_str = expiry.strftime("%d%b%y").upper()
    return f"{token} {expiry_str} {strike} {'CE' if option_type == 'CALL' else 'PE'}"



# -----------------------------
# New Option Handling Functions
# -----------------------------
def get_expiry_date(option_type, token):
    """Get expiry date based on option type and token"""
    today = datetime.today()
    
    if "NIFTY" in token:
        # Weekly expiry (next Thursday)
        days_to_thursday = (3 - today.weekday()) % 7  # 0=Monday
        if days_to_thursday == 0 and today.hour > 15:  # If today is Thursday after market close
            days_to_thursday = 7
        expiry_date = today + timedelta(days=days_to_thursday)
    else:  # BANKNIFTY
        # Monthly expiry (last Thursday)
        last_day = today + relativedelta(day=31)
        while last_day.weekday() != 3:  # Thursday
            last_day -= timedelta(days=1)
        expiry_date = last_day
        
    return expiry_date

# -----------------------------
# New Helper Functions
# -----------------------------
def log_signal_to_csv(token, signal_data):
    """Updated CSV logging with option details"""

    try:
        today = datetime.now().strftime("%Y-%m-%d")
        filename = f"data/{today}/{token}.csv"
        Path(f"data/{today}").mkdir(parents=True, exist_ok=True)
        
        # Create/append to CSV file
        file_exists = Path(filename).exists()
        
        with open(filename, 'a', newline='') as csvfile:
            fieldnames = [
                'timestamp', 'signal_type', 'price', 'strike',
                'expiry', 'symbol', 'option_type', 'rsi',
                'ema_20', 'ema_50', 'vwap', 'atr'
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            if not file_exists:
                writer.writeheader()
                
            writer.writerow(signal_data)
            
    except Exception as e:
        logging.error(f"CSV logging failed: {str(e)}")


async def trading_cycle(session, token, exch, risk_manager, position_manager):
    try:
        # Check minimum holding period
        last_signal_time = position_manager.last_signal_time.get(token, datetime.min)
        time_since_last = (datetime.now() - last_signal_time).total_seconds() / 60
        
        if time_since_last < CONFIG["min_holding_period"]:
            logging.info(f"{token}: Skipping cycle - {CONFIG['min_holding_period']} min cooldown")
            return

        # Fetch and process data
        one_min_data = await fetch_historical_data(session, token, "1", exch)
        three_min_data = await fetch_historical_data(session, token, "3", exch)
        
        if not one_min_data or not three_min_data:
            logging.warning(f"{token}: Missing data for current cycle")
            return
            
        one_min_df = process_candle_data(one_min_data)
        three_min_df = process_candle_data(three_min_data)
        
        if one_min_df.empty or three_min_df.empty:
            logging.warning(f"{token}: Empty dataframes after processing")
            return

        # Get current prices
        current_close = three_min_df['close'].iloc[-1]
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        print(f"\n[{current_time}] {token} Prices:")
        print(f"1min Close: {one_min_df['close'].iloc[-1]:.2f}")
        print(f"3min Close: {current_close:.2f}")
        print(f"RSI: {three_min_df['rsi'].iloc[-1]:.2f}")
        print(f"VWAP: {three_min_df['vwap'].iloc[-1]:.2f}")

        # Check strategies
        breakout_signal, breakout_details = check_breakout_strategy(one_min_df, three_min_df)
        mean_rev_signal, mean_rev_details = check_mean_reversion_strategy(three_min_df)
        
        # Determine signal type
        signal_type = ""
        action = None
        trade_price = current_close
        is_exit = False
        
        # Check existing positions
        if position_manager.has_open_position(token):
            current_position = position_manager.positions[token]
            current_direction = current_position['direction']
            
            # Check exit conditions
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
            # Check entry conditions
            if breakout_signal:
                action = 'entry'
                signal_type = "BREAKOUT LONG" if breakout_details['price'] > three_min_df['vwap'].iloc[-1] else "BREAKOUT SHORT"
                trade_price = breakout_details['price']
            elif mean_rev_signal:
                action = 'entry'
                signal_type = mean_rev_details['type']
                trade_price = mean_rev_details['price']

        # Execute trade if needed
        if action:
            # Prepare signal data
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
            
            # Execute trade and log
            await execute_trade(session, signal_data, risk_manager, token, is_exit)
            
            # Update position manager
            if action == 'entry':
                position_manager.open_position(token, signal_type, trade_price)
            
            position_manager.last_signal_time[token] = datetime.now()
            
            print(f"\n{token}: {action.upper()} signal triggered - {signal_type}")
        else:
            print(f"{token}: No valid signals detected")

    except Exception as e:
        logging.error(f"Trading cycle error ({token}): {str(e)}")
        raise

async def execute_trade(session, signal, risk_manager, token, is_exit=False):
    try:
        # Determine option parameters
        strike_interval = 50 if "NIFTY" in token else 100
        spot_price = signal['price']
        
        # Generate trade details
        if is_exit:
            option_type = 'PE' if 'SHORT' in signal['signal_type'] else 'CE'
            action = "Exit"
        else:
            option_type = 'CE' if 'BUY' in signal['signal_type'] or 'LONG' in signal['signal_type'] else 'PE'
            action = "Entry"
        
        strike = calculate_strike_price(spot_price, token)
        expiry_date = get_expiry_date(signal['signal_type'], token)
        symbol = generate_symbol(token, expiry_date, strike, option_type)
        
        # Prepare CSV data
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

        # Log to CSV
        today = datetime.now().strftime("%Y-%m-%d")
        filename = f"data/{today}/{token}_trades.csv"
        Path(f"data/{today}").mkdir(parents=True, exist_ok=True)
        
        file_exists = Path(filename).exists()
        with open(filename, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=csv_data.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(csv_data)

        # Print trade execution details
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

        # Update risk manager
        if is_exit and 'pnl' in signal:
            risk_manager.update_trade_log(signal['pnl'])

    except Exception as e:
        logging.error(f"Trade execution failed ({token}): {str(e)}")
        raise

# -----------------------------
# Authentication & Main Loop
# -----------------------------
async def main():
    # Initialize risk management
    risk_manager = RiskManager(CONFIG["max_drawdown"])
    position_manager = PositionManager()  # Initialize position tracker
    
    # Initialize API session
    async with aiohttp.ClientSession() as session:
        while True:
            try:
                now = datetime.now().time()
                if not (time(9, 25) < now < time(23, 58)):
                    await asyncio.sleep(300)
                    continue
                    
                # Trading cycle for each instrument
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
    asyncio.run(main())