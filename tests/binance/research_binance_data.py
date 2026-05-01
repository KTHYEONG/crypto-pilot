import requests

def check_monthly_vision():
    symbol = "BTCUSDT"
    month = "2024-01"
    paths = [
        f"https://data.binance.vision/data/futures/um/monthly/klines/{symbol}/1h/{symbol}-1h-{month}.zip",
        f"https://data.binance.vision/data/futures/um/monthly/metrics/{symbol}/{symbol}-metrics-{month}.zip",
        f"https://data.binance.vision/data/futures/um/monthly/allForceOrders/{symbol}/{symbol}-allForceOrders-{month}.zip"
    ]
    for url in paths:
        try:
            resp = requests.head(url, timeout=5)
            print(f"Monthly {url.split('/')[-3]}: {resp.status_code}")
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    check_monthly_vision()
