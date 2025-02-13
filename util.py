import asyncio
import aiohttp
import requests
import logging
import hashlib
import json
import pandas as pd
import os
import numpy as np
from collections import deque
from datetime import datetime, timedelta
import threading
import functools
import certifi

time_window = timedelta(minutes=30)

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
    "websocket_url":"wss://api.shoonya.com/NorenWSTP/",
}

PLACE_ORDER_URL = CONFIG["base_url"] + "PlaceOrder"
TRADE_BOOK_URL  = CONFIG["base_url"] + "TradeBook"
ORDER_BOOK_URL  = CONFIG["base_url"] + "OrderBook"
TPSERIES_URL    = CONFIG["base_url"] + "TPSeries"

# List of NSE holidays (example for 2025-2025)
NSE_HOLIDAYS = [
    "2025-02-26",  # Mahashivratri
    "2025-03-14",  # Holi
    "2025-03-31",  # Id-Ul-Fitr
    "2025-04-10",  # Shri Mahavir Jayanti
    "2025-04-14",  # Dr. Baba Saheb Ambedkar Jayanti
    "2025-04-18",  # Good Friday
    "2025-05-01",  # Maharashtra Day
    "2025-08-15",  # Independence Day
    "2025-08-27",  # Ganesh Chaturthi
    "2025-10-02",  # Mahatma Gandhi Jayanti/Dussehra
    "2025-10-21",  # Diwali Laxmi Pujan
    "2025-10-22",  # Diwali-Balipratipada
    "2025-11-05",  # Prakash Gurpurb Sri Guru Nanak Dev
    "2025-12-25",  # Christmas
    # Add more holidays as needed
]

all_data = deque()

async def shoonya_get_tp_series_async(session, token, intrv='1'):
    now = datetime.now()
    start_time = now - timedelta(hours=1)  # Fetch last 1 hour of data

    st = int(start_time.timestamp())  # Convert to epoch
    et = int(now.timestamp())  # Convert to epoch

    jData = {
        "uid": CONFIG["user_id"],
        "exch": "NFO",
        "token": str(token),
        "st": str(st),
        "et": str(et),
        "intrv": str(intrv)
    }

    susertoken, _, _ = read_token_from_file()
    form_data = "jData=" + json.dumps(jData) + "&jKey=" + susertoken
    #print("Requesting:", form_data)

    timeout = aiohttp.ClientTimeout(total=30)  # Set 15 seconds timeout

    try:
        async with session.post(TPSERIES_URL, data=form_data, timeout=timeout) as response:
            response_text = await response.text()  # Read response as text (for debugging)
            #print("Response:", response_text)
            return json.loads(response_text)  # Convert back to JSON
    except asyncio.TimeoutError:
        print("❌ Request timed out after 30 seconds")
        return None
    except Exception as e:
        print("❌ Error:", e)
        return None

async def fetch_all_data_async(data_ce, data_pe):
    """Fetch both 1-min and 5-min data for CE and PE asynchronously"""
    # print("fetch_all_data_async", datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

    async with aiohttp.ClientSession() as session:
        # Fetch 1-minute data for CE & PE
        tasks = [
            shoonya_get_tp_series_async(session, data_ce["token"], '1'),
            shoonya_get_tp_series_async(session, data_pe["token"], '1')
        ]

        # Fetch 5-minute data only when minute % 5 == 0
        if datetime.now().minute % 5 == 0:
            tasks.extend([
                shoonya_get_tp_series_async(session, data_ce["token"], '5'),
                shoonya_get_tp_series_async(session, data_pe["token"], '5')
            ])

        results = await asyncio.gather(*tasks)

    #print('results - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - ',results)

    # Store results in the respective dictionaries
    data_ce["ohlc_1m_api"] = results[0]
    data_pe["ohlc_1m_api"] = results[1]

    #print("CE 1m Data:", results[0])
    #print("PE 1m Data:", results[1])

    # If we fetched 5-minute data, store it as well
    if len(results) > 2:
        data_ce["ohlc_5m_api"] = results[2]
        data_pe["ohlc_5m_api"] = results[3]
        #print("CE 5m Data:", results[2])
        #print("PE 5m Data:", results[3])

def fetch_data(data_ce, data_pe):
    """Wrapper to run the async function inside the same event loop"""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(fetch_all_data_async(data_ce, data_pe))

    # Calculate the next execution time
    now = datetime.now()

    # Next execution at HH:MM:01
    next_run = (now + timedelta(minutes=1)).replace(second=1, microsecond=0)
    delay = (next_run - now).total_seconds()

    # Schedule the next run
    threading.Timer(delay, functools.partial(fetch_data, data_ce, data_pe)).start()
    
def shoonya_get_order_book():
    jData = {
        "uid": CONFIG["user_id"],
        "prd": "M",
    }
    susertoken, _, _  = read_token_from_file()
    form_data = "jData="+ json.dumps(jData)+"&jKey="+susertoken
    response = requests.post(ORDER_BOOK_URL, data=form_data, verify=certifi.where())
    return response.json()

def shoonya_get_trade_book():
    jData = {
        "uid": CONFIG["user_id"],
        "actid": CONFIG["user_id"],
    }
    susertoken, _, _  = read_token_from_file()
    form_data = "jData="+ json.dumps(jData)+"&jKey="+susertoken
    response = requests.post(TRADE_BOOK_URL, data=form_data, verify=certifi.where())
    return response.json()

def shoonya_place_order(tsym,prc,remarks):
    jData = {
        "uid": CONFIG["user_id"],
        "actid": CONFIG["user_id"],
        "exch": "NFO",
        "tsym": tsym,
        "qty": "75",
        "prc": prc,
        "trantype": "B",
        "ret": "DAY",
        "prctyp": "MKT",
        "bpprc": "{:.2f}".format(float(prc) * 1.1),  # ✅ Correct
        "blprc": "{:.2f}".format(float(prc) * 0.9),  # ✅ Correct
        "prd": "B",
        "remarks": remarks
    }
    susertoken, _, _  = read_token_from_file()

    form_data = "jData="+ json.dumps(jData)+"&jKey="+susertoken

    # Send the POST request
    response = requests.post(PLACE_ORDER_URL, data=form_data, verify=certifi.where())

    # Print the response
    print(response.json())

def is_holiday(date):
    """
    Check if the given date is a holiday.
    """
    date_str = date.strftime("%Y-%m-%d")
    return date_str in NSE_HOLIDAYS

def convert_datetime(obj):
    """ Convert datetime objects to strings for JSON serialization """
    if isinstance(obj, datetime):
        return obj.strftime('%Y-%m-%d %H:%M:%S')  # Convert to string format
    raise TypeError(f"Type {type(obj)} not serializable")

def get_nifty_expiry():
    """
    Get the next weekly expiry date for Nifty options.
    If expiry falls on a holiday, move to Wednesday.
    """
    today = datetime.today().date()  # Get only date part

    # Find the next Thursday
    days_to_thursday = (3 - today.weekday()) % 7  # 3 represents Thursday
    expiry_date = today + timedelta(days=days_to_thursday)

    # If today is already Thursday and not a holiday, return today
    if today.weekday() == 3 and not is_holiday(today):
        return today.strftime("%d%b%y").upper()

    # If Thursday is a holiday, move to Wednesday
    if is_holiday(expiry_date):
        expiry_date -= timedelta(days=1)

    return expiry_date

def get_expiry_date(symbol):
    """
    Calculate the expiry date for the given month and year.
    Expiry is on the last Thursday of the month.
    If today is after the expiry date, move to next month's expiry.
    """
        # Get the current year and month
    today = datetime.now()
    year = today.year
    month = today.month

    today_data = datetime.now().date()  # Current date without time

    while True:  # Loop until we find a valid expiry
        # Get the last day of the month
        if month == 12:
            next_month = datetime(year + 1, 1, 1)
        else:
            next_month = datetime(year, month + 1, 1)
        last_day = next_month - timedelta(days=1)

        # Find the last Thursday of the month
        last_thursday = last_day - timedelta(days=(last_day.weekday() - 3) % 7)

        # If expiry has already passed, move to next month
        if last_thursday.date() > today_data:
            break
        else:
            month += 1  # Move to next month
            if month > 12:
                month = 1
                year += 1

    # Handle holiday check
    while is_holiday(last_thursday):
        last_thursday -= timedelta(days=1)

    return last_thursday

def calculate_tsym(symbol, latest_price, strike_interval=50):
    """
    Calculate the trading symbol (tsym) for an option contract.
    """
    latest_price = float(latest_price)
    # Round the latest price to the nearest strike interval
    strike_price_c = round(latest_price / strike_interval) * strike_interval + strike_interval
    strike_price_p = round(latest_price / strike_interval) * strike_interval - strike_interval

    # Get the expiry date
    if symbol == "NIFTY":
        expiry_date = get_nifty_expiry()
    else:
        expiry_date = get_expiry_date(symbol)

    # Format the expiry date as DDMMMYY (e.g., 30JAN25)
    print('expiry_date',expiry_date)
    expiry_date = datetime.strptime(expiry_date, "%d%b%y")
    expiry_str = expiry_date.strftime("%d%b%y").upper()

    # Construct the tsym
    tsym_c = f"{symbol}{expiry_str}C{strike_price_c}"
    tsym_p = f"{symbol}{expiry_str}P{strike_price_p}"
    return [tsym_c,tsym_p]

# Function to check if the tokens.json file exists
def check_token_file_exists():
    return os.path.exists('tokens.json')

def sha256_encode(password):
    return hashlib.sha256(password.encode()).hexdigest()

# Function to save token to tokens.json
def save_token_to_file(susertoken, susertokenspl):
    try:
        today_date = datetime.today().strftime('%Y-%m-%d')  # Get today's date in YYYY-MM-DD format
        token_data = {
            "susertoken": susertoken,
            "susertokenspl": susertokenspl,
            "date": today_date  # Save today's date
        }
        with open('tokens.json', 'w') as file:
            json.dump(token_data, file)
    except Exception as e:
        print(f"Error saving token: {e}")

# Function to read token from tokens.json
def read_token_from_file():
    try:
        with open('tokens.json', 'r') as file:
            tokens = json.load(file)
            return tokens.get("susertoken"), tokens.get("susertokenspl"), tokens.get("date")  # Get the user token and susertokenspl from the file
    except FileNotFoundError:
        print("Token file not found.")
        return None, None, None
    except json.JSONDecodeError:
        print("Error decoding the token file.")
        return None, None, None

def is_order_already_open(trade_book, tsym_to_check):
    try:
        # Ensure trade_book is a list
        if isinstance(trade_book, str):  
            trade_book = json.loads(trade_book)  # Convert JSON string to list

        for order in trade_book:
            if isinstance(order, dict) and order.get("status") == "OPEN" and order.get("tsym") == tsym_to_check:
                return True
    except Exception as e:
        logging.error(f"Error processing trade_book: {e}")
    return False

def placeOrder(data,comment,tsym):
    print("Placing Order... 🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀",comment,data)
    date_str = datetime.now().strftime('%Y-%m-%d')  # Example: "2025-01-29"
    file_path = f'orders_{date_str}.txt'  # Example: "orders_2025-01-29.txt"
    trade_book = shoonya_get_trade_book()
    if is_order_already_open(trade_book, tsym):
        print(f"Skipping order placement as {tsym} is already OPEN in trade book.")
        return  # Exit early if an open order exists
    
    shoonya_place_order(tsym,data["lp"],comment)
    # Step 1: Read existing data (if file exists)
    if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
        with open(file_path, 'r') as file:
            try:
                orders = json.load(file)  # Load existing orders
                if not isinstance(orders, list):
                    orders = []  # Ensure it's a list
            except json.JSONDecodeError:
                orders = []  # Handle corrupted file
    else:
        orders = []  # If file doesn't exist, start with an empty list

    # Step 2: Append new data
    ft_datetime = datetime.fromtimestamp(int(data["ft"]))
    orders.append({**data, "date": ft_datetime.strftime("%Y-%m-%d %H:%M:%S"),"tsym":tsym,"comment":comment,"bpprc": "{:.2f}".format(float(data["lp"]) * 1.1),"blprc": "{:.2f}".format(float(data["lp"]) * 0.9)})

    # Step 3: Write updated list back to file
    try:
        with open(file_path, 'w') as file:
            json.dump(orders, file, default=convert_datetime, indent=4)
    except Exception as e:
        print(f"Error saving order: {e}")

def calculate_atr(highs, lows, closes, period=14):
    tr = np.maximum(highs - lows, np.maximum(abs(highs - closes.shift(1)), abs(lows - closes.shift(1))))
    return tr.rolling(window=period).mean()

def calculate_supertrend(highs, lows, closes, period=10, multiplier=3):
    atr = calculate_atr(highs, lows, closes, period)
    hl2 = (highs + lows) / 2
    upper_band = hl2 + (multiplier * atr)
    lower_band = hl2 - (multiplier * atr)

    supertrend = np.zeros_like(closes)
    trend = np.ones_like(closes)

    for i in range(1, len(closes)):
        if closes[i] > upper_band[i - 1]:
            trend[i] = 1
        elif closes[i] < lower_band[i - 1]:
            trend[i] = -1
        else:
            trend[i] = trend[i - 1]

        if trend[i] == 1:
            supertrend[i] = lower_band[i]
        else:
            supertrend[i] = upper_band[i]

    return supertrend, trend

def calculate_rsi(closes, period=14):
    delta = closes.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def update_ohlc(data, ohlc_1m, ohlc_5m):
    timestamp = int(data["ft"])
    dt = datetime.fromtimestamp(timestamp)

    minute_key = dt.strftime("%Y-%m-%d %H:%M")
    five_min_key = dt.strftime("%Y-%m-%d %H:") + str((dt.minute // 5) * 5).zfill(2)
    lp = float(data.get('lp', data.get('sp1', 0)))

    # Update 1-minute OHLC
    if minute_key not in ohlc_1m:
        ohlc_1m[minute_key] = {"open": lp, "high": lp, "low": lp, "close": lp}
    else:
        ohlc_1m[minute_key]["high"] = max(ohlc_1m[minute_key]["high"], lp)
        ohlc_1m[minute_key]["low"] = min(ohlc_1m[minute_key]["low"], lp)
        ohlc_1m[minute_key]["close"] = lp

    # Update 5-minute OHLC
    if five_min_key not in ohlc_5m:
        ohlc_5m[five_min_key] = {"open": lp, "high": lp, "low": lp, "close": lp}
    else:
        ohlc_5m[five_min_key]["high"] = max(ohlc_5m[five_min_key]["high"], lp)
        ohlc_5m[five_min_key]["low"] = min(ohlc_5m[five_min_key]["low"], lp)
        ohlc_5m[five_min_key]["close"] = lp

    # Convert OHLC data into DataFrame
    df = pd.DataFrame.from_dict(ohlc_5m, orient="index").sort_index()
    df["close"] = df["close"].astype(float)
    df["high"] = df["high"].astype(float)
    df["low"] = df["low"].astype(float)

    if len(df) >= 20:
        # Calculate RSI
        df["RSI"] = calculate_rsi(df["close"])

        # Calculate Supertrend
        df["Supertrend"], df["Trend"] = calculate_supertrend(df["high"], df["low"], df["close"])

        latest_rsi = df["RSI"].iloc[-1]
        latest_trend = df["Trend"].iloc[-1]
        latest_price = df["close"].iloc[-1]

        # Buy Signal: Price above Supertrend and RSI < 30
        if latest_trend == 1 and latest_rsi < 30 and ohlc_5m[five_min_key]["close"] > ohlc_5m[five_min_key]["open"]:
            logging.info(f"BUY Signal at {five_min_key}: RSI={latest_rsi}, Price={latest_price}")
            placeOrder(data, f"BUY Signal: Supertrend Bullish & RSI Oversold ({latest_rsi})", data["tk"])

        # Sell Signal: Price below Supertrend and RSI > 70
        elif latest_trend == -1 and latest_rsi > 70 and ohlc_5m[five_min_key]["close"] < ohlc_5m[five_min_key]["open"]:
            logging.info(f"SELL Signal at {five_min_key}: RSI={latest_rsi}, Price={latest_price}")
            placeOrder(data, f"SELL Signal: Supertrend Bearish & RSI Overbought ({latest_rsi})", data["tk"])

def updateData(data,data_option):
    stored_records = data_option['data']
    #print(data_option)
    ohlc_1m = data_option['ohlc_1m']
    ohlc_5m = data_option['ohlc_5m']

    try:
        current_volume = int(data.get('v',0))
        current_lp = float(data.get('lp', data.get('sp1', 0)))
        record_time = datetime.fromtimestamp(int(data['ft']))
        
    except (ValueError, TypeError) as e:
        logging.error(f"Invalid data format: {e}")
        return


    previous_record = stored_records[-1] if stored_records else None
    previous_volume = previous_record['v'] if previous_record else 0
    previous_lp = previous_record['lp'] if previous_record else 0

    # Use actual volume if present; otherwise, copy previous volume
    if (current_volume is None or current_volume == 0) and previous_volume > 0:
        current_volume = previous_volume  # Copy only if missing
    else:
        current_volume = float(current_volume)  # Convert valid volume

    # Use actual lp if present; otherwise, copy previous lp
    if (current_lp is None or current_lp == 0) and previous_lp > 0:
        current_lp = previous_lp  # Copy only if missing
    else:
        current_lp = float(current_lp)  # Convert valid volume

    # Calculate changes
    delta_v = current_volume - previous_volume if previous_volume is not None else 0
    delta_v_percent = (delta_v / previous_volume * 100) if previous_volume else 0

    delta_lp = current_lp - previous_lp if previous_lp else 0
    delta_lp_percent = (delta_lp / previous_lp * 100) if previous_lp else 0

    # Append new record only if it wasn't merged
    stored_records.append({
        'time': record_time, 
        'lp': current_lp, 
        'lp_change': round(delta_lp, 2), 
        'lp_change_per': round(delta_lp_percent, 2),
        'v': current_volume, 
        'v_change': delta_v, 
        'v_change_per': round(delta_v_percent, 2)
    })
    
    update_ohlc(data,ohlc_1m,ohlc_5m)

    # Save stored_records to temp.json
    #with open('feed_data.json', 'w') as file:
        #json.dump(data_option, file, indent=4, default=str)
    
    if previous_volume is None or previous_volume == 0 or  previous_lp is None or  previous_lp == 0:
        logging.info("First message received. Skipping volume and LP comparisons.")
        return
    else:
        #print(delta_v_percent, " > 0.5 |||| " ,delta_lp_percent, " > 1", record_time,"======>",data_option['name'],current_lp)
        if delta_v_percent > 0.5 and delta_v_percent < 10 and delta_lp_percent > 1 and delta_lp_percent < 20 and previous_lp > 75:
            placeOrder(data,f'condition_1 => {delta_v_percent} > 0.5 and {delta_lp_percent} > 1',data_option["name"])
        else: 
            if delta_v_percent > 0 and delta_v_percent < 10 and delta_lp_percent > 4 and delta_lp_percent < 20 and previous_lp > 75:
                placeOrder(data,f'condition_2 => {delta_v_percent} > 0 and {delta_lp_percent} > 3 ',data_option["name"])
        

    # Remove records older than the time window (15 minutes)
    current_time = datetime.now()
    #while stored_records and stored_records[0]['time'] < current_time - time_window:
        #stored_records.popleft()
        #ERROR:root:Unexpected error: 'list' object has no attribute 'popleft'
        #ERROR:root:Unexpected error: 'list' object has no attribute 'popleft'

def handleData(data, data_ce, data_pe):
    try:
        current_token = data.get('tk') 
    except (ValueError, TypeError) as e:
        logging.error(f"Invalid data format: {e}")
        return
    
    if current_token == data_ce['token']:
        updateData(data,data_ce)
    if current_token == data_pe['token']:
        updateData(data,data_pe)
