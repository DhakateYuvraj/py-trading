import asyncio
import aiohttp
import os
import hashlib
from datetime import datetime, timedelta
import json
import requests
import certifi
import logging
import signal
import sys
import pandas as pd

# -----------------------------
# Configuration and Constants
# -----------------------------
OPTIONS = [
    {"name": "Nifty", "value": "NIFTY", "token": "26000", "exch": "NSE", "weeklyLot": 75, "monthlyLot": 25, "strike_interval": 50},
    {"name": "BankNifty", "value": "BANKNIFTY", "token": "26009", "exch": "NSE", "weeklyLot": 30, "monthlyLot": 15, "strike_interval": 100},
    {"name": "CRUDEOILM19MAR25", "value": "CRUDEOILM19MAR25", "token": "CRUDEOILM19MAR25", "exch": "MCX", "weeklyLot": 30, "monthlyLot": 15, "strike_interval": 100},
]

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
}

# API URLs
LOGIN_URL = CONFIG["base_url"] + "QuickAuth"
GET_QUOTES_URL = CONFIG["base_url"] + "GetQuotes"
TPSERIES_URL = CONFIG["base_url"] + "TPSeries"

# Logging configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


# -----------------------------
# Technical Analysis Functions
# -----------------------------
def calculate_ema(data, period):
    """Calculate Exponential Moving Average (EMA) for a given period."""
    return data["close"].ewm(span=period, adjust=False).mean()


def log_signal_to_file(signal, closing_price, timestamp, ema_values):
    """Log signal data into a JSON file named with today's date."""
    filename = f"{timestamp.strftime('%Y-%m-%d')}.json"
    new_entry = {
        "signal": signal,
        "closing_price": closing_price,
        "time": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
        "ema_values": ema_values
    }
    
    if os.path.exists(filename):
        try:
            with open(filename, "r") as f:
                data = json.load(f)
        except json.JSONDecodeError:
            data = []
    else:
        data = []
    
    data.append(new_entry)
    
    with open(filename, "w") as f:
        json.dump(data, f, indent=4)


def check_ema_conditions(data):
    # Calculate EMAs for the provided data
    data["ema_20"] = calculate_ema(data, 20)
    data["ema_50"] = calculate_ema(data, 50)
    data["ema_100"] = calculate_ema(data, 100)
    data["ema_200"] = calculate_ema(data, 200)

    current = data.iloc[-1]
    previous = data.iloc[-2]

    buy_condition = (
        current["ema_20"] < current["ema_50"] < current["ema_100"] < current["ema_200"] and
        not (
            (previous["ema_20"] < previous["ema_50"] < previous["ema_100"] < previous["ema_200"]) or
            (previous["ema_20"] > previous["ema_50"] < previous["ema_100"] < previous["ema_200"])
        )
    )

    sell_condition = (
        current["ema_20"] > current["ema_50"] > current["ema_100"] > current["ema_200"] and
        not (
            (previous["ema_20"] > previous["ema_50"] > previous["ema_100"] > previous["ema_200"]) or
            (previous["ema_20"] < previous["ema_50"] > previous["ema_100"] > previous["ema_200"])
        )
    )

    if buy_condition:
        return "SELL"
    elif sell_condition:
        return "BUY"
    else:
        return "NO SIGNAL"


# -----------------------------
# Breakout Condition Functions
# -----------------------------
def check_breakout_condition_1min(df):
    """
    Check the 1‑min chart for breakout:
      - Pre-breakout consolidation: 10 candles within a 0.5% band and gradually declining volume.
      - Trigger candle: Close at least 0.2% above previous candle's high and volume surge of at least 150%.
    Returns (True, details) if condition met; else (False, None).
    """
    df = df.sort_values("time").reset_index(drop=True)
    if len(df) < 11:
        return False, None

    consolidation = df.iloc[-11:-1].copy()  # Last 10 candles for consolidation
    breakout_candle = df.iloc[-1]

    consolidation["mid"] = (consolidation["high"] + consolidation["low"]) / 2
    avg_mid = consolidation["mid"].mean()
    max_high = consolidation["high"].max()
    min_low = consolidation["low"].min()
    if (max_high - min_low) > (0.005 * avg_mid):
        return False, None

    volumes = consolidation["volume"].tolist()
    if not all(earlier >= later for earlier, later in zip(volumes, volumes[1:])):
        return False, None

    previous_candle = consolidation.iloc[-1]
    if breakout_candle["close"] < previous_candle["high"] * 1.002:
        return False, None

    avg_volume = consolidation["volume"].mean()
    if breakout_candle["volume"] < 1.5 * avg_volume:
        return False, None

    return True, {"breakout_low": breakout_candle["low"], "time": breakout_candle["time"]}


def check_breakout_confirmation_3min(df):
    """
    Check the 3‑min chart for breakout confirmation:
      - Last two candles must show a higher high and higher low.
      - The last candle's volume must be at least 1.5× the average volume of the previous 10 candles.
    Returns True if confirmed; otherwise, False.
    """
    df = df.sort_values("time").reset_index(drop=True)
    if len(df) < 11:
        return False

    last_two = df.iloc[-2:]
    if not (last_two.iloc[1]["high"] > last_two.iloc[0]["high"] and last_two.iloc[1]["low"] > last_two.iloc[0]["low"]):
        return False

    volume_avg = df.iloc[-11:-1]["volume"].mean()
    if last_two.iloc[1]["volume"] < 1.5 * volume_avg:
        return False

    return True


def check_breakout_conditions(one_min_df, three_min_df):
    """
    Check if the breakout condition on the 1‑min chart and its confirmation on the 3‑min chart are met.
    Returns (True, details) if both conditions are satisfied.
    """
    cond1, breakout_details = check_breakout_condition_1min(one_min_df)
    cond2 = check_breakout_confirmation_3min(three_min_df)
    if cond1 and cond2:
        return True, breakout_details
    return False, None


# -----------------------------
# Token and Login Functions
# -----------------------------
def check_token_file_exists():
    return os.path.exists("tokens.json")


def sha256_encode(password):
    return hashlib.sha256(password.encode()).hexdigest()


def read_token_from_file():
    try:
        with open("tokens.json", "r") as file:
            tokens = json.load(file)
            return tokens.get("susertoken"), tokens.get("susertokenspl"), tokens.get("date")
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logging.error(f"Error reading token file: {e}")
        return None, None, None


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


# -----------------------------
# Data Fetching Functions
# -----------------------------
def get_market_data(token, exch, susertoken):
    if not susertoken:
        logging.error("Token is missing. Please login again.")
        return None
    jData = {
        "uid": CONFIG["user_id"],
        "exch": exch,
        "token": token,
    }
    data = f"jData={json.dumps(jData)}&jKey={susertoken}"
    try:
        response = requests.post(GET_QUOTES_URL, data=data, verify=certifi.where())
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        logging.error(f"Error fetching market data: {e}")
        return None


async def shoonya_get_tp_series_async(session, token, intrv="1", exch=""):
    now = datetime.now()
    # Fetch last 100 hours of data (adjust if needed)
    start_time = now - timedelta(hours=100)
    st = int(start_time.timestamp())
    et = int(now.timestamp())
    jData = {
        "uid": CONFIG["user_id"],
        "exch": exch,
        "token": str(token),
        "st": str(st),
        "et": str(et),
        "intrv": str(intrv),
    }
    logging.info(f"Fetching data with parameters: {jData}")
    susertoken, _, _ = read_token_from_file()
    form_data = f"jData={json.dumps(jData)}&jKey={susertoken}"
    timeout = aiohttp.ClientTimeout(total=30)
    try:
        async with session.post(TPSERIES_URL, data=form_data, timeout=timeout) as response:
            response_text = await response.text()
            return json.loads(response_text)
    except asyncio.TimeoutError:
        logging.error("Request timed out after 30 seconds")
        return None
    except Exception as e:
        logging.error(f"Error fetching time series data: {e}")
        return None


async def fetch_all_data_async(token_to_fetch, exch):
    async with aiohttp.ClientSession() as session:
        # Fetch both 1‑min and 3‑min data concurrently
        results = await asyncio.gather(
            shoonya_get_tp_series_async(session, token_to_fetch, "1", exch),
            shoonya_get_tp_series_async(session, token_to_fetch, "3", exch)
        )
    try:
        one_min_raw = results[0]
        three_min_raw = results[1]

        # Process 1‑min data
        if isinstance(one_min_raw, list) and len(one_min_raw) > 0:
            one_min_df = pd.DataFrame(one_min_raw)
            one_min_df["time"] = pd.to_datetime(one_min_df["time"], format="%d-%m-%Y %H:%M:%S", errors="coerce")
            # Rename keys to standard OHLCV names
            one_min_df.rename(columns={
                "into": "open",
                "inth": "high",
                "intl": "low",
                "intc": "close",
                "v": "volume"
            }, inplace=True)
            one_min_df["close"] = pd.to_numeric(one_min_df["close"], errors="coerce")
            one_min_df["high"] = pd.to_numeric(one_min_df["high"], errors="coerce")
            one_min_df["low"] = pd.to_numeric(one_min_df["low"], errors="coerce")
            one_min_df["volume"] = pd.to_numeric(one_min_df["volume"], errors="coerce")
            one_min_df.sort_values("time", inplace=True)
        else:
            logging.info("1‑min fetched data is empty or not in expected format.")
            one_min_df = None

        # Process 3‑min data
        if isinstance(three_min_raw, list) and len(three_min_raw) > 0:
            three_min_df = pd.DataFrame(three_min_raw)
            three_min_df["time"] = pd.to_datetime(three_min_df["time"], format="%d-%m-%Y %H:%M:%S", errors="coerce")
            three_min_df.rename(columns={
                "into": "open",
                "inth": "high",
                "intl": "low",
                "intc": "close",
                "v": "volume"
            }, inplace=True)
            three_min_df["close"] = pd.to_numeric(three_min_df["close"], errors="coerce")
            three_min_df["high"] = pd.to_numeric(three_min_df["high"], errors="coerce")
            three_min_df["low"] = pd.to_numeric(three_min_df["low"], errors="coerce")
            three_min_df["volume"] = pd.to_numeric(three_min_df["volume"], errors="coerce")
            three_min_df.sort_values("time", inplace=True)
        else:
            logging.info("3‑min fetched data is empty or not in expected format.")
            three_min_df = None

        # Existing EMA-based condition on 3‑min data
        if three_min_df is not None and len(three_min_df) >= 2:
            signal = check_ema_conditions(three_min_df)
            current = three_min_df.iloc[-1]
            last_close = current["close"]
            ema_values = {
                "ema_20": current["ema_20"],
                "ema_50": current["ema_50"],
                "ema_100": current["ema_100"],
                "ema_200": current["ema_200"],
            }
            logging.info(f"EMA Signal: {signal}; Closing Price: {last_close}")
            if signal != "NO SIGNAL":
                log_signal_to_file(signal, last_close, current["time"], ema_values)
        else:
            logging.info("Not enough 3‑min data to calculate EMA signal.")

        # New Breakout Condition using both 1‑min and 3‑min data
        if one_min_df is not None and three_min_df is not None:
            breakout_triggered, breakout_details = check_breakout_conditions(one_min_df, three_min_df)
            if breakout_triggered:
                breakout_stop_loss = breakout_details["breakout_low"]
                breakout_time = breakout_details["time"]
                logging.info(f"Breakout Signal: BREAKOUT LONG; Stop Loss: {breakout_stop_loss}; Time: {breakout_time}")
                log_signal_to_file("BREAKOUT LONG", breakout_stop_loss, breakout_time, {"breakout_details": breakout_details})

        # Save raw data to JSON files
        today = datetime.today().strftime("%Y-%m-%d")
        folder_path = os.path.join("data", today)
        os.makedirs(folder_path, exist_ok=True)
        file_path_1min = os.path.join(folder_path, f"{token_to_fetch}_1min.json")
        with open(file_path_1min, "w") as f:
            json.dump(one_min_raw, f, indent=4)
        file_path_3min = os.path.join(folder_path, f"{token_to_fetch}_3min.json")
        with open(file_path_3min, "w") as f:
            json.dump(three_min_raw, f, indent=4)
    except Exception as e:
        logging.error(f"Error processing data: {e}")


# -----------------------------
# Asynchronous Scheduler
# -----------------------------
async def scheduler(token_to_fetch, exch):
    """Asynchronous scheduler that fetches data every minute at HH:MM:01."""
    while True:
        await fetch_all_data_async(token_to_fetch, exch)
        now = datetime.now()
        next_run = (now + timedelta(minutes=1)).replace(second=1, microsecond=0)
        delay = (next_run - now).total_seconds()
        logging.info(f"Sleeping for {delay:.2f} seconds until next fetch.")
        await asyncio.sleep(delay)


# -----------------------------
# Graceful Shutdown Handling
# -----------------------------
def shutdown(signum, frame):
    logging.info("Shutting down gracefully...")
    sys.exit(0)


# -----------------------------
# Main Execution
# -----------------------------
def main():
    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    print("Select an option:")
    for i, option in enumerate(OPTIONS, start=1):
        print(f"{i}. {option['name']}")

    choice = input("Enter your choice (1, 2, ...): ")
    if not choice.isdigit() or not (1 <= int(choice) <= len(OPTIONS)):
        logging.error("Invalid input. Exiting.")
        return
    selected_option = OPTIONS[int(choice) - 1]
    logging.info(f"You selected: {selected_option['name']} ({selected_option['token']} - {selected_option['exch']})")

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

    market_data = get_market_data(selected_option["token"], "NSE", susertoken)
    if market_data:
        fetch_token = selected_option["token"]
        exch = selected_option["exch"]
        logging.info(f"Fetching data for token: {fetch_token} on exchange: {exch}")
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.create_task(scheduler(fetch_token, exch))
            loop.run_forever()
        finally:
            loop.close()
    else:
        logging.error("Failed to fetch market data. Exiting.")


if __name__ == "__main__":
    main()
