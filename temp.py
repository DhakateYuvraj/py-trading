import asyncio
import requests
import logging
import json
import pandas as pd
import os
import websocket
from collections import deque
from datetime import datetime, timedelta, timezone
from util import calculate_tsym, convert_datetime, check_token_file_exists, sha256_encode, handleData, read_token_from_file, save_token_to_file, fetch_data, CONFIG

latest_records = deque()
data_ce={}
data_pe={}

# Set the time window for the last 30 minutes
time_window = timedelta(minutes=30)


# Shoonya API URL
LOGIN_URL = CONFIG["base_url"] + "QuickAuth"
GET_QUOTES_URL = CONFIG["base_url"] + "GetQuotes"
SEARCH_SCRIP_URL = CONFIG["base_url"] + "SearchScrip"


options = [
    {"name": "Nifty","value": "NIFTY", "token": "26000", "exch": "NSE","weeklyLot":75,"monthlyLot":25,"strike_interval":50},
    {"name": "BankNifty","value": "BANKNIFTY", "token": "26009", "exch": "NSE","weeklyLot":30,"monthlyLot":15,"strike_interval":100},
]


def display_latest_records():
    """ Display latest_records in a table format """
    if not latest_records:
        print("\nNo records to display.\n")
        return
    
    os.system('cls' if os.name == 'nt' else 'clear')  # Clear the console for a fresh display
    
    df = pd.DataFrame(latest_records)
    df['time'] = df['time'].dt.strftime('%H:%M:%S')  # Format time for better readability
    
    print(df.to_string(index=False, justify='center'))


def on_message(ws, message):
    try:
        # Parse the incoming message
        data = json.loads(message)
        logging.info(f"WebSocket Data: {data}")

        # Handle authentication acknowledgment
        if data.get("t") == "ck" and data.get("s") == "OK":
            logging.info("Authentication successful.")
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
            return
        
        #condition_1(data)
        handleData(data, data_ce, data_pe)
        #handleData(data, {"name":"443491","token":"443491","data":[], "ohlc_1m" : {},"ohlc_5m" : {},}, data_pe)
     
        #print(data)
        #display_latest_records()  # Refresh table after updating records

    except json.JSONDecodeError as e:
        logging.error(f"Failed to parse WebSocket message: {e}")
    except Exception as e:
        logging.error(f"Unexpected error: {e}")

def on_error(ws, error):
    print('on_error')

def on_close(ws, close_status_code, close_msg):
    print('on_close')
    try:
        with open('temp_CE.txt', 'w') as file:
            json.dump(data_ce, file, default=convert_datetime, indent=4)
        with open('temp_PE.txt', 'w') as file:
            json.dump(data_pe, file, default=convert_datetime, indent=4)
    except Exception as e:
        print(f"Error saving token: {e}")

def on_open(ws):
    print("WebSocket connection opened.")
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

async def startApp(data_ce, data_pe, token_ce, token_pe):
    await asyncio.gather(
        fetch_data(data_ce, data_pe),
        start_websocket(f'NFO|{token_ce}#NFO|{token_pe}')
    )


# Main function
if __name__ == "__main__":
    # Display options
    print("Select an option:")
    for i, option in enumerate(options, start=1):
        print(f"{i}. {option['name']}")

    # Take user input
    choice = input("Enter your choice (1 or 2): ")

    # Validate input
    if choice.isdigit():
        choice = int(choice)
        if 1 <= choice <= len(options):
            selected_option = options[choice - 1]
            print(f"You selected: {selected_option['name']} ({selected_option['token']} - {selected_option['exch']})")
        else:
            print("Invalid choice. Please select a valid option.")
    else:
        print("Invalid input. Please enter a number.")

    if check_token_file_exists():
        today_date = datetime.today().strftime('%Y-%m-%d')  # Get today's date in YYYY-MM-DD format
        susertoken, susertokenspl, date = read_token_from_file()
        if susertoken and date == today_date:
            print(f"Using existing token: {susertoken}")
        else:
            print("Token is missing. Please login again.")
            susertoken, susertokenspl = login()
    else:
        print("Token file not found. Logging in...")
        susertoken, susertokenspl = login()

    if susertoken:
        market_data = get_market_data(selected_option["token"],"NSE",susertoken)
        if market_data:
            market_data_lp = str(market_data.get('lp', market_data.get('sp1', 0)))
            print(market_data['tsym'] + ' => ' + market_data_lp)
            tsym = calculate_tsym(selected_option["value"], market_data_lp,selected_option["strike_interval"])
            token_ce = get_token_for_tsym(tsym[0],susertoken)
            token_pe = get_token_for_tsym(tsym[1],susertoken)
            data_ce = {
                "name":tsym[0],
                "token":token_ce,
                "ohlc_1m" : {},
                "ohlc_5m" : {},
                "ohlc_1m_api" : {},
                "ohlc_5m_api" : {},
                "data":[]
            }
            data_pe = {
                "name":tsym[1],
                "token":token_pe,
                "ohlc_1m" : {},
                "ohlc_5m" : {},
                "ohlc_1m_api" : {},
                "ohlc_5m_api" : {},
                "data":[]
            }
            print(tsym,token_ce,token_pe)
            #asyncio.run(startApp(data_ce, data_pe, token_ce, token_pe))
            fetch_data(data_ce,data_pe)
            print(f'NFO|{token_ce}#NFO|{token_pe}')
            #MCX|443491
            #start_websocket("MCX|443491")
            start_websocket(f'NFO|{token_ce}#NFO|{token_pe}')
        else:
            print("Failed to retrieve market data.")
    else:
        print("Login failed. Exiting.")
