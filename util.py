import requests
import logging
import hashlib
import json
import pandas as pd
import os
import websocket
import time
from collections import deque
from datetime import datetime, timedelta, timezone
from datetime import datetime, timedelta

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
TRADE_BOOK_URL =  CONFIG["base_url"] + "TradeBook"
ORDER_BOOK_URL =  CONFIG["base_url"] + "OrderBook"



def shoonya_get_order_book():
    jData = {
        "uid": CONFIG["user_id"],
        "prd": "M",
    }
    susertoken, susertokenspl, date  = read_token_from_file()
    form_data = "jData="+ json.dumps(jData)+"&jKey="+susertoken
    response = requests.post(ORDER_BOOK_URL, data=form_data)
    return response.json()


def shoonya_get_trade_book():
    jData = {
        "uid": CONFIG["user_id"],
        "actid": CONFIG["user_id"],
    }
    susertoken, susertokenspl, date  = read_token_from_file()
    form_data = "jData="+ json.dumps(jData)+"&jKey="+susertoken
    response = requests.post(TRADE_BOOK_URL, data=form_data)
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
    susertoken, susertokenspl, date  = read_token_from_file()

    form_data = "jData="+ json.dumps(jData)+"&jKey="+susertoken

    # Send the POST request
    response = requests.post(PLACE_ORDER_URL, data=form_data)

    # Print the response
    print(response.json())

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


def calculate_tsym(symbol, latest_price, strike_interval=50):
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
    tsym = f"{symbol}{expiry_str}{strike_price}"
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
    
def condition_1(data):
    print(data)
    current_lp = float(data.get('lp', 0))
    record_time = datetime.fromtimestamp(int(data['ft']))
    if all_data:
        first_record_time = all_data[0]['record_time']  # Directly access without converting
        if record_time.replace(second=0, microsecond=0) == first_record_time.replace(second=0, microsecond=0):
            if current_lp > 0:
                all_data[0]['c'] = current_lp
                if all_data[0]['o'] == 0:
                    all_data[0]['o'] = current_lp
                if all_data[0]['l'] == 0:
                    all_data[0]['l'] = current_lp
                if all_data[0]['h'] == 0:
                    all_data[0]['h'] = current_lp
                if current_lp > all_data[0]['h']:
                    all_data[0]['h'] = current_lp
                if current_lp < all_data[0]['l']:
                    all_data[0]['l'] = current_lp
            print("The times match up to the minute!",record_time,first_record_time, data.get('lp', 0))
        else:
            print("The times do not match.",record_time,first_record_time)
            all_data.insert(0, {"current_lp":current_lp,
                                "record_time":record_time.replace(second=0, microsecond=0),
                                "o" : current_lp,
                                "c" : current_lp,
                                "h" : current_lp,
                                "l" : current_lp,
                                }) 
            print("=========>",all_data)
    else:
        all_data.insert(0, {"current_lp":current_lp,
                    "record_time":record_time.replace(second=0, microsecond=0),
                    "o" : current_lp,
                    "c" : current_lp,
                    "h" : current_lp,
                    "l" : current_lp,
                    }) 
        print("=========>",all_data)

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
    
    #shoonya_place_order(tsym,data["lp"],comment)
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
    orders.append({**data, "date": ft_datetime.strftime("%Y-%m-%d"),"tsym":tsym,"comment":comment})

    # Step 3: Write updated list back to file
    try:
        with open(file_path, 'w') as file:
            json.dump(orders, file, default=convert_datetime, indent=4)
    except Exception as e:
        print(f"Error saving order: {e}")

def updateData(data,data_option):
    stored_records = data_option['data']
    try:
        current_volume = int(data.get('v',0))
        current_lp = float(data.get('lp', 0))
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
    
    if previous_volume is None or previous_volume == 0 or  previous_lp is None or  previous_lp == 0:
        logging.info("First message received. Skipping volume and LP comparisons.")
        return
    else:
        if delta_v_percent > 0.5 and delta_v_percent < 10 and delta_lp_percent > 1 and delta_lp_percent < 20 and previous_lp > 0:
            placeOrder(data,f'condition_1 => {delta_v_percent} > 0.5 and {delta_lp_percent} > 1',data_option["name"])
        else: 
            if delta_v_percent > 0 and delta_v_percent < 10 and delta_lp_percent > 3 and delta_lp_percent < 20 and previous_lp > 75:
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
