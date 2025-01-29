import requests
import logging
import hashlib
import json
import os
import websocket
import time
from collections import deque
from datetime import datetime, timedelta, timezone


latest_records = deque()
# Set the time window for the last 30 minutes
time_window = timedelta(minutes=30)

# List of NSE holidays (example for 2025-2025)
NSE_HOLIDAYS = [
    "2025-01-26",  # Republic Day
    "2025-03-08",  # Mahashivratri
    "2025-03-25",  # Holi
    "2025-04-11",  # Id-Ul-Fitr
    "2025-05-01",  # Maharashtra Day
    "2025-06-17",  # Bakri Id
    "2025-07-17",  # Muharram
    "2025-08-15",  # Independence Day
    "2025-10-02",  # Gandhi Jayanti
    "2025-11-01",  # Diwali Laxmi Pujan
    "2025-12-25",  # Christmas
    "2025-01-26",  # Republic Day
    # Add more holidays as needed
]

def is_holiday(date):
    """
    Check if the given date is a holiday.
    """
    date_str = date.strftime("%Y-%m-%d")
    return date_str in NSE_HOLIDAYS

def get_expiry_date(year, month):
    """
    Calculate the expiry date for the given month and year.
    Expiry is on the last Thursday, or the previous day if Thursday is a holiday.
    """
    # Get the last day of the month
    if month == 12:
        next_month = datetime(year + 1, 1, 1)
    else:
        next_month = datetime(year, month + 1, 1)
    last_day = next_month - timedelta(days=1)

    # Start with the last Thursday
    last_thursday = last_day - timedelta(days=(last_day.weekday() - 3) % 7)

    # Check if Thursday is a holiday
    if is_holiday(last_thursday):
        # Move to Wednesday
        last_thursday -= timedelta(days=1)
        # Check if Wednesday is also a holiday
        if is_holiday(last_thursday):
            # Move to Tuesday
            last_thursday -= timedelta(days=1)
            # Check if Tuesday is also a holiday
            if is_holiday(last_thursday):
                # Move to Monday
                last_thursday -= timedelta(days=1)

    return last_thursday

def calculate_tsym(symbol, latest_price, option_type="C", strike_interval=50):
    """
    Calculate the trading symbol (tsym) for an option contract.
    """
    latest_price = float(latest_price)
    # Round the latest price to the nearest strike interval
    strike_price = round(latest_price / strike_interval) * strike_interval + strike_interval
    strike_price_c = round(latest_price / strike_interval) * strike_interval + strike_interval
    strike_price_p = round(latest_price / strike_interval) * strike_interval - strike_interval

    # Get the current year and month
    today = datetime.now()
    year = today.year
    month = today.month

    # Get the expiry date
    expiry_date = get_expiry_date(year, month)

    # Format the expiry date as DDMMMYY (e.g., 30JAN25)
    expiry_str = expiry_date.strftime("%d%b%y").upper()

    # Construct the tsym
    tsym = f"{symbol}{expiry_str}{option_type}{strike_price}"
    tsym_c = f"{symbol}{expiry_str}{option_type}{strike_price_c}"
    tsym_p = f"{symbol}{expiry_str}{option_type}{strike_price_p}"
    return [tsym_c,tsym_p]

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

# Shoonya API URL
LOGIN_URL = CONFIG["base_url"] + "QuickAuth"
GET_QUOTES_URL = CONFIG["base_url"] + "GetQuotes"
SEARCH_SCRIP_URL = CONFIG["base_url"] + "SearchScrip"

# Function to check if the tokens.json file exists
def check_token_file_exists():
    return os.path.exists('tokens.json')


def sha256_encode(password):
    return hashlib.sha256(password.encode()).hexdigest()

def on_message(ws, message):
    try:
        # Parse the incoming message
        data = json.loads(message)
        logging.info(f"WebSocket Data: {data}")
        
        # Get the current volume from the message (assuming the message has a 'v' field)
        current_volume = data.get('v')
        current_lp = data.get('lp')  # Assuming 'lp' is the last price field

            #if 'v' in data or 'lp' in data:

        #print(data)
        # Check if the message is an acknowledgment of authentication
        if data.get("t") == "ck" and data.get("s") == "OK":
            logging.info("Authentication successful.")
            
            # Send subscription message
            subscribe_message = {
                "uid": "FN105570",  # Replace with dynamic UID if needed
                "actid": "FN105570",  # Replace with dynamic account ID if needed
                "source": "WEB",
                "susertoken": susertoken,  # Assume susertoken is available globally
                "t": "t",
                "k": ws.tokens,
            }
            ws.send(json.dumps(subscribe_message))
            logging.info(f"Subscribed to tokens: {ws.tokens}")
            
        if 'v' in data or 'lp' in data:
            if current_volume is None or current_lp is None:
                logging.warning("Volume or LP is missing in the message. Skipping.")
                return
            
            # Extract previous_volume from the latest record (last record in latest_records)
            if latest_records:
                previous_volume = latest_records[-1].get('v')
                previous_lp= latest_records[-1].get('lp')
            else:
                previous_volume = None
                previous_lp = None
            
            # If previous_volume is None, it's the first record, so skip comparison
            if previous_volume is None:
                logging.info("First message received. Skipping volume change comparison.")
                latest_records.append(data)
                return
            
            # Calculate volume change percentage if previous_volume is available
            volume_change_percentage = (current_volume - previous_volume) / previous_volume * 100
            
            # Condition 1: Volume change percentage > 30%
            if volume_change_percentage > 30:
                logging.info(f"Volume change > 30%. Current volume: {current_volume}, Previous volume: {previous_volume}")
                # Call placeOrder or any other action here
            
            # Condition 2: Current volume > 30% of the average volume of the last 15 minutes
            if len(latest_records) > 15:
                avg_volume = sum(record.get('v', 0) for record in latest_records) / len(latest_records)
                if current_volume > 0.3 * avg_volume:
                    logging.info(f"Current volume is greater than 30% of average volume in last 15 minutes.")
                    # Call placeOrder or any other action here
            
            # Condition 3: Change in LP is greater than 4%
            if previous_lp is not None and abs((current_lp - previous_lp) / previous_lp * 100) > 4:
                logging.info(f"Change in LP is greater than 4%. Current LP: {current_lp}, Previous LP: {previous_lp}")
                # Call placeOrder or any other action here
            
            # Append the current data to the latest_records
            latest_records.append(data)
            
            # Remove records older than the time window (15 minutes)
            current_time = datetime.now()
            while latest_records and latest_records[0]['ft'] < current_time - time_window:
                latest_records.popleft()

    except json.JSONDecodeError as e:
        logging.error(f"Failed to parse WebSocket message: {e}")


def on_error(ws, error):
    """
    Handle WebSocket errors.
    """
    print('on_error')
    logging.error(f"WebSocket Error: {error}")

def on_close(ws, close_status_code, close_msg):
    """
    Handle WebSocket connection close.
    """
    print('on_close')
    logging.warning(f"WebSocket Connection Closed: {close_msg}")
    logging.info("Reconnecting...")
    #start_websocket(ws.tokens)


def on_open(ws):
    print("WebSocket connection opened.")
    logging.info("WebSocket Connection Established")
    if ws.tokens:
        if not susertoken:
            logging.error("No valid token found. Cannot subscribe.")
            ws.close()
            return

        # Authentication payload
        auth_payload = {
            "t": "c",  # Authentication message type
            "uid": "FN105570",  # Replace with actual UID
            "actid": "FN105570",  # Replace with actual account ID
            "susertoken": susertoken,  # Use token attached to WebSocket object
        }
        
        # Send authentication payload
        ws.send(json.dumps(auth_payload))
        logging.info(f"Subscribed to tokens: {ws.tokens}")

def start_websocket(tokens):
    websocket_url = CONFIG["websocket_url"]
    ws = websocket.WebSocketApp(
        websocket_url,
        on_message=on_message,
        on_error=on_error,
        on_close=on_close,
    )
    ws.on_open = on_open
    ws.tokens = tokens  # Attach tokens to WebSocket object
    ws.run_forever()


def get_token_for_tsym(tsym, susertoken, exchange="NFO"):
    if not susertoken:
        logging.error("Token missing. Please login.")
        return None

    jData = {
        "uid": CONFIG["user_id"],
        "stext": tsym,  # Trading symbol to search
        "exch": exchange,  # Exchange (NFO for options)
    }
    form_data = "jData="+ json.dumps(jData)+"&jKey="+susertoken

    try:
        response = requests.post(SEARCH_SCRIP_URL, data=form_data)
        response.raise_for_status()
        data = response.json()
        if data.get("stat") == "Ok":
            # Find the exact match for the tsym
            for item in data["values"]:
                if item["tsym"] == tsym:
                    return item["token"]  # Return the token for the tsym
            logging.error(f"No exact match found for {tsym}.")
            return None
        else:
            logging.error(f"Failed to fetch token for {tsym}: {data.get('emsg')}")
            return None
    except requests.RequestException as e:
        logging.error(f"Error fetching token for {tsym}: {e}")
        return None


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

# Login to Shoonya API
def login():
    print("Logging in...")
    # Ask for TOTP input
    hashed_password = sha256_encode(CONFIG["password"])
    hashed_app_key = sha256_encode(CONFIG["user_id"] + "|" + CONFIG["api_key"])
    totp_key = input("Enter your TOTP: ")
    jData = {
        "apkversion": "1.0.0",
        "uid": CONFIG["user_id"],
        "pwd": hashed_password,
        "appkey": hashed_app_key,
        "factor2": totp_key,#CONFIG["totp_key"],
        "imei": CONFIG["imei"],
        "source": "API",
        "vc": CONFIG["vendor_code"],
    }
    payload = f'jData={json.dumps(jData)}' # Convert dictionary to raw text format

    try:
        response = requests.post(LOGIN_URL, data=payload, timeout=10)
        response.raise_for_status()  # Raise error for bad status codes
        data = response.json()
        if data.get("stat") == "Ok":
            susertoken = data.get("susertoken")
            susertokenspl = data.get("susertokenspl")
            if susertoken and susertokenspl:
                save_token_to_file(susertoken, susertokenspl)
                return susertoken, susertokenspl
            else:
                print("Login failed: Missing tokens.")
                return None, None
        else:
            print(f"Login failed: {data.get('emsg')}")
            return None, None
    except requests.exceptions.RequestException as e:
        print(f"Error logging in: {e}")
        return None, None

# Function to get market data
def get_market_data(token,exch,susertoken):
    if not susertoken:
        print("Token is missing. Please login again.")
        return None

    # jData payload and jKey (API key)
    jData = {
        "uid": CONFIG["user_id"],  # Replace with your user ID
        "exch":exch,  # Specify the exchange (e.g., NSE for Nifty)
        "token":token  # Replace with the desired token (e.g., Nifty Token)
    }

    # Construct the data payload
    data =  'jData=' + json.dumps(jData) + '&jKey=' + susertoken

    # Sending the POST request
    try:
        response = requests.post(GET_QUOTES_URL, data=data)
        response.raise_for_status()  # Raise error for bad status codes
        return response.json()  # Return the response JSON if successful
    except requests.exceptions.RequestException as e:
        print(f"Error fetching market data: {e}")
        return None

# Main function
if __name__ == "__main__":

    if check_token_file_exists():
        today_date = datetime.today().strftime('%Y-%m-%d')  # Get today's date in YYYY-MM-DD format
        susertoken, susertokenspl, date = read_token_from_file()
        if susertoken and date == today_date:
            print(f"Using existing token: {susertoken}")


            tsym_ce = "NIFTY30JAN25C22600"  # Call Option
            tsym_pe = "NIFTY30JAN25P22600"  # Put Option

            # Fetch tokens for CE and PE
            token_ce = get_token_for_tsym(tsym_ce,susertoken)
            token_pe = get_token_for_tsym(tsym_pe,susertoken)
            print(token_ce,token_pe)

            if not token_ce or not token_pe:
                logging.error("Failed to fetch tokens for CE and PE.")
            else:
                logging.info(f"Token for {tsym_ce}: {token_ce}")
                logging.info(f"Token for {tsym_pe}: {token_pe}")

                # Start WebSocket connection
                #tokens_to_subscribe = f"NFO|{token_ce}#NFO|{token_pe}"
                tokens_to_subscribe = "MCX|438576"
                print('tokens_to_subscribe',tokens_to_subscribe)
                start_websocket(tokens_to_subscribe)
                #print(token_ce,token_pe)



        else:
            print("Token is missing. Please login again.")
            susertoken, susertokenspl = login()
    else:
        print("Token file not found. Logging in...")
        susertoken, susertokenspl = login()

    if susertoken:
        market_data = get_market_data("26000","NSE",susertoken)
        if market_data:
            print(market_data['tsym'] + ' => ' + market_data['lp'])  # Pretty-print the response
            tsym = calculate_tsym("NIFTY", market_data['lp'], option_type="C")
            print(tsym)
        else:
            print("Failed to retrieve market data.")
    else:
        print("Login failed. Exiting.")
