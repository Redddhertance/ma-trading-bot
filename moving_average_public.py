import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import ta
from statistics import stdev
import numpy as np
import alpaca_trade_api as tradeapi
import os
import time
import datetime as dt
import requests
import smtplib
from email.message import EmailMessage
from newsapi.newsapi_client import NewsApiClient
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Load Alpaca API credentials
ALPACA_API_KEY = 'YOUR_KEY_HERE'
ALPACA_SECRET_KEY = 'YOUR_KEY_HERE'
BASE_URL = 'YOUR_URL_HERE'
NEWS_API_KEY = os.getenv("YOUR_NEWS_KEY_HERE")

api = tradeapi.REST(ALPACA_API_KEY, ALPACA_SECRET_KEY, BASE_URL, api_version='v2')


def backtest_strategy(ticker, use_sl_tp=True):
    weak_tickers = ["AMZN", "NFLX", "NVDA"]
    data = yf.download(ticker, start="2022-01-01", end="2023-12-31")
    data['RSI'] = ta.momentum.RSIIndicator(close=data['Close'].squeeze(), window=5).rsi()
    data['200SMA'] = data['Close'].rolling(window=200).mean()

    data['Signal'] = 0

    
    initial_cash = 1000
    cash = initial_cash
    shares = 0
    short_shares = 0
    portfolio_values = []

    entry_price = None
    short_entry_price = None
    stop_loss = None
    take_profit = None
    last_buy_index = -1

    atr = ta.volatility.AverageTrueRange(
        high=data['High'].squeeze(),
        low=data['Low'].squeeze(),
        close=data['Close'].squeeze(),
        window=14
    ).average_true_range()

    atr_threshold = atr.mean() * 0.9 

    total_trades = 0
    winning_trades = 0
    profits = []
    returns = []
    trade_log = []

    for i in range(1, len(data)):
        today = data.iloc[i:i+1].copy().squeeze()
        today['Date'] = data.index[i]
        price = today['Close'] if 'Close' in today else None

        # Buy when RSI < 40
        if (
            float(today['RSI'].iloc[0]) < 40 and
            shares == 0 and
            float(atr.iloc[i]) < atr_threshold and
            float(today['Close'].iloc[0]) > float(data['200SMA'].iloc[i])
        ):
            shares = int(cash // price.iloc[0])
            if shares > 0:
                cash -= shares * price.iloc[0]
                print(f"BUY: {shares} shares at ${price.iloc[0]:.2f} on {today['Date'].iloc[0].date()}")
                entry_price = price.iloc[0]
                last_buy_index = i
                if use_sl_tp:
                    atr_value = atr.iloc[i]
                    multiplier = 2.0 if ticker in weak_tickers else (1.0 + 0.25 * (40 - today['RSI'].iloc[0]) / 40)
                    stop_loss = entry_price - atr_value * multiplier
                    take_profit = entry_price + atr_value * multiplier

                if shares > 0:
                    execute_trade(ticker, 'buy', shares)

        # Short when RSI > 65
        # elif (
        #     float(today['RSI'].iloc[0]) > 65 and
        #     short_shares == 0 and shares == 0 and
        #     float(atr.iloc[i]) < atr_threshold and
        #     float(today['Close'].iloc[0]) < float(data['200SMA'].iloc[i])
        # ):
        #     short_price = price.iloc[0]
        #     short_shares = int(cash // short_price)
        #     if short_shares > 0:
        #         cash += short_shares * short_price  # receive cash for selling borrowed shares
        #         print(f"SHORT: {short_shares} shares at ${short_price:.2f} on {today['Date'].iloc[0].date()}")
        #         short_entry_price = short_price
        #         last_buy_index = i
        #         if use_sl_tp:
        #             atr_value = atr.iloc[i]
        #             stop_loss = short_entry_price + atr_value * (1.0 + 0.5 * (today['RSI'].iloc[0] - 70) / 30)
        #             take_profit = short_entry_price - atr_value * (1.0 + 0.5 * (today['RSI'].iloc[0] - 70) / 30)

        # Sell condition with optimized exit rule
        elif shares > 0 and (
            (today['RSI'].iloc[0] > 70 and today['Close'].iloc[0] < data['Close'].iloc[i - 1].item())
            or (i - last_buy_index > 10)
        ):
            cash += shares * price.iloc[0]
            print(f"SELL: {shares} shares at ${price.iloc[0]:.2f} on {today['Date'].iloc[0].date()}")
            total_trades += 1
            if price.iloc[0] > entry_price:
                winning_trades += 1
            pnl = price.iloc[0] - entry_price
            returns.append(pnl / entry_price)
            trade_log.append({
                'Date': today['Date'],
                'Type': 'SELL',
                'Entry Price': entry_price,
                'Exit Price': price.iloc[0],
                'Profit': pnl
            })
            profits.append(pnl)
            shares = 0
            entry_price = None
            stop_loss = None
            take_profit = None

            execute_trade(ticker, 'sell', shares)

        # Short exit condition
        # elif short_shares > 0 and (
        #     (today['RSI'].iloc[0] < 30 and today['Close'].iloc[0] > data['Close'].iloc[i - 1])
        #     or (i - last_buy_index > 10)
        # ):
        #     cover_price = price.iloc[0]
        #     cash -= short_shares * cover_price  # buy back the shares
        #     print(f"COVER: {short_shares} shares at ${cover_price:.2f} on {today['Date'].iloc[0].date()}")
        #     total_trades += 1
        #     pnl = short_entry_price - cover_price
        #     if pnl > 0:
        #         winning_trades += 1
        #     returns.append(pnl / short_entry_price)
        #     trade_log.append({
        #         'Date': today['Date'],
        #         'Type': 'COVER',
        #         'Entry Price': short_entry_price,
        #         'Exit Price': cover_price,
        #         'Profit': pnl
        #     })
        #     profits.append(pnl)
        #     short_shares = 0
        #     short_entry_price = None
        #     stop_loss = None
        #     take_profit = None

        # Exit on SL/TP for long trades
        elif shares > 0 and use_sl_tp and (today['Low'].iloc[0] <= stop_loss or today['High'].iloc[0] >= take_profit):
            exit_price = stop_loss if today['Low'].iloc[0] <= stop_loss else take_profit
            cash += shares * exit_price
            print(f"EXIT (SL/TP): {shares} shares at ${exit_price:.2f} on {today['Date'].iloc[0].date()}")
            total_trades += 1
            if exit_price > entry_price:
                winning_trades += 1
            pnl = exit_price - entry_price
            returns.append(pnl / entry_price)
            trade_log.append({
                'Date': today['Date'],
                'Type': 'EXIT',
                'Entry Price': entry_price,
                'Exit Price': exit_price,
                'Profit': pnl
            })
            profits.append(pnl)
            shares = 0
            entry_price = None
            stop_loss = None
            take_profit = None

            execute_trade(ticker, 'sell', shares)

        # Exit on SL/TP for short trades
        # elif short_shares > 0 and use_sl_tp and (today['High'].iloc[0] >= stop_loss or today['Low'].iloc[0] <= take_profit):
        #     exit_price = stop_loss if today['High'].iloc[0] >= stop_loss else take_profit
        #     cash -= short_shares * exit_price
        #     print(f"EXIT SHORT (SL/TP): {short_shares} shares at ${exit_price:.2f} on {today['Date'].iloc[0].date()}")
        #     total_trades += 1
        #     pnl = short_entry_price - exit_price
        #     if pnl > 0:
        #         winning_trades += 1
        #     returns.append(pnl / short_entry_price)
        #     trade_log.append({
        #         'Date': today['Date'],
        #         'Type': 'EXIT SHORT',
        #         'Entry Price': short_entry_price,
        #         'Exit Price': exit_price,
        #         'Profit': pnl
        #     })
        #     profits.append(pnl)
        #     short_shares = 0
        #     short_entry_price = None
        #     stop_loss = None
        #     take_profit = None

        
        portfolio_value = cash + (shares * price.iloc[0]) - (short_shares * price.iloc[0])
        portfolio_values.append(portfolio_value)

   
    final_price = data.iloc[-1]['Close']
    final_value = cash + (shares * final_price) - (short_shares * final_price)

    print(f"\nStarting Balance: ${initial_cash}")
    print(f"Final Portfolio Value: ${round(final_value, 2)}")
    print(f"Net Profit: ${round(final_value - initial_cash, 2)}")

    if len(returns) > 1:
        sharpe = round((np.mean(returns) / (stdev(returns) if stdev(returns) != 0 else 1e-6)) * (252 ** 0.5), 2)
        print(f"Sharpe Ratio: {sharpe}")
    else:
        print("Sharpe Ratio: Not enough data points")

    if len(portfolio_values) > 1:
        max_dd = round(max([max(portfolio_values[:i+1]) - pv for i, pv in enumerate(portfolio_values)]), 2)
        print(f"Max Drawdown: ${max_dd}")
    else:
        print("Max Drawdown: Not enough data points")

    if total_trades > 0:
        print(f"Total Trades: {total_trades}")
        print(f"Winning Trades: {winning_trades}")
        print(f"Win Rate: {round(winning_trades / total_trades * 100, 2)}%")
        print(f"Average Profit per Trade: ${round(sum(profits) / len(profits), 2)}")

    pd.DataFrame(trade_log).to_csv(f"{ticker}_trade_log.csv", index=False)

def execute_trade(symbol, side, qty):
    try:
        api.submit_order(
            symbol=symbol,
            qty=qty,
            side=side,
            type='market',
            time_in_force='gtc'
        )
        print(f"{side.upper()} order placed for {qty} shares of {symbol}")
        
        now = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        trade_log_entry = {
            'Timestamp': now,
            'Symbol': symbol,
            'Side': side,
            'Quantity': int(qty),
            'Price': 'Market'
        }
        log_file = "paper_trade_log.csv"
        if not os.path.exists(log_file):
            pd.DataFrame.from_records([trade_log_entry]).to_csv(log_file, index=False)
        else:
            pd.DataFrame.from_records([trade_log_entry]).to_csv(log_file, mode='a', index=False, header=False)

        send_notification(f"{side.upper()} order placed: {qty} shares of {symbol}")
    except Exception as e:
        print(f"Error placing order: {e}")

def send_notification(message):
    try:
        sender_email = os.getenv("EMAIL_HERE")
        sender_password = os.getenv("EMAIL_HERE")
        recipient_email = os.getenv("EMAIL_HERE")

        if not sender_email or not sender_password or not recipient_email:
            print(f"Notification: {message} (email credentials not set)")
            return

        msg = EmailMessage()
        msg.set_content(message)
        msg['Subject'] = 'Trading Bot Alert'
        msg['From'] = sender_email
        msg['To'] = recipient_email

        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
            smtp.login(sender_email, sender_password)
            smtp.send_message(msg)
        print(f"Email sent: {message}")
    except Exception as e:
        print(f"Failed to send email notification: {e}")

def get_news_sentiment(symbol):
    try:
        newsapi = NewsApiClient(api_key=NEWS_API_KEY)
        analyzer = SentimentIntensityAnalyzer()
        query = f"{symbol} stock"
        news = newsapi.get_everything(q=query, language='en', sort_by='publishedAt', page_size=5)
        headlines = [article['title'] for article in news['articles']]

        if not headlines:
            return 0  

        scores = [analyzer.polarity_scores(headline)['compound'] for headline in headlines]
        avg_sentiment = sum(scores) / len(scores)
        return avg_sentiment
    except Exception as e:
        print(f"Error fetching news sentiment for {symbol}: {e}")
        return 0

def run_paper_trading(symbols, use_sl_tp=True):
    print("Starting paper trading session...")
    active_trades = {}
    while True:
        now = dt.datetime.now()
        if (now.hour > 14 or (now.hour == 14 and now.minute >= 30)) and now.hour < 20:  # US market hours (Eastern Time)
            try:
                account = api.get_account()
                equity = float(account.equity)
                print(f"Account Equity: ${equity:.2f}")
            except Exception as e:
                print(f"Error fetching account equity: {e}")
                import traceback; traceback.print_exc()

            for symbol in symbols:
                try:
                    live_data = yf.download(tickers=symbol, period="1d", interval="1m", auto_adjust=True, progress=False)
                    
                    required_cols = ['Close', 'High', 'Low']
                    if not isinstance(live_data, pd.DataFrame) or live_data.empty:
                        print(f"Invalid or empty DataFrame for {symbol}. Skipping.")
                        continue

                    if not all(col in live_data.columns for col in required_cols):
                        print(f"Missing required columns for {symbol}. Columns present: {live_data.columns.tolist()}")
                        continue

                    skip_symbol = False
                    for col in required_cols:
                        if live_data[col].ndim == 0:
                            print(f"{symbol}: Column '{col}' has unexpected dimensions. Skipping.")
                            skip_symbol = True
                            break
                        if live_data[col].dropna().empty:
                            print(f"{symbol}: Column '{col}' is all NaN. Skipping.")
                            skip_symbol = True
                            break
                    if skip_symbol:
                        continue
                    
                    if isinstance(live_data, pd.Series):
                        live_data = live_data.to_frame().T
                    
                    
                    if not isinstance(live_data.index, pd.DatetimeIndex):
                        try:
                            live_data.index = pd.to_datetime(live_data.index)
                        except Exception as e:
                            print(f"Error parsing datetime index for {symbol}: {e}")
                            continue
                    
                    
                    if live_data.empty or live_data.index.isnull().any():
                        print(f"{symbol} returned DataFrame with invalid index. Skipping.")
                        continue
                    
                    
                    if 'Datetime' in live_data.columns:
                        live_data.set_index('Datetime', inplace=True)
                    elif 'Date' in live_data.columns:
                        live_data.set_index('Date', inplace=True)

                    if isinstance(live_data, pd.Series):
                        live_data = live_data.to_frame().T

                    if not isinstance(live_data.index, pd.DatetimeIndex):
                        try:
                            live_data.index = pd.to_datetime(live_data.index)
                        except Exception as e:
                            print(f"Error parsing datetime index for {symbol}: {e}")
                            continue

                    if live_data.empty:
                        print(f"{symbol} returned empty DataFrame after cleanup. Skipping.")
                        continue
                    if len(live_data) > 0:
                        close_prices = live_data['Close']
                        if isinstance(close_prices, pd.DataFrame):
                            close_prices = close_prices.squeeze()
                        
                        if isinstance(close_prices, pd.Series) and not close_prices.empty:
                            rsi_series = ta.momentum.RSIIndicator(close=close_prices, window=5).rsi()
                            if isinstance(rsi_series, pd.Series) and not rsi_series.empty and rsi_series.index.equals(live_data.index):
                                live_data['RSI'] = rsi_series
                            else:
                                print(f"{symbol}: RSI series invalid or mismatched. Skipping.")
                                continue
                        else:
                            print(f"{symbol}: Invalid 'Close' prices for RSI. Skipping.")
                            continue

                        live_data['200SMA'] = live_data['Close'].rolling(window=200).mean()
                        
                        high = live_data['High']
                        low = live_data['Low']
                        close = live_data['Close']
                        tr1 = high - low
                        tr2 = (high - close.shift(1)).abs()
                        tr3 = (low - close.shift(1)).abs()
                        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
                        
                        live_data['ATR'] = true_range.rolling(window=14).mean()

                        if live_data is None or live_data.empty:
                            print(f"No data returned for {symbol}. Skipping.")
                            continue

                        
                        last_row = live_data.iloc[-1]
                        rsi = last_row['RSI'].item()
                        price = last_row['Close'].item()
                        sma_200 = last_row['200SMA'].item()
                        atr_value = last_row['ATR'].item()
 
                       
                        if np.isnan([rsi, sma_200, atr_value]).any():
                            with open("trade_rejections.log", "a") as log:
                                log.write(f"{dt.datetime.now()} - {symbol} - Skipped due to NaN in RSI/SMA/ATR\n")
                            continue

                        atr_threshold = live_data['ATR'].mean() * 0.9
                        weak_tickers = ["AMZN", "NFLX", "NVDA"]
                        multiplier = 2.0 if symbol in weak_tickers else (1.0 + 0.25 * (40 - rsi) / 40)

                        if rsi < 40 and price > sma_200 and atr_value < atr_threshold:
                            sentiment_score = get_news_sentiment(symbol)
                            if sentiment_score < -0.2:
                                print(f"Negative sentiment for {symbol} ({sentiment_score:.2f}). Skipping trade.")
                                with open("trade_rejections.log", "a") as log:
                                    log.write(f"{dt.datetime.now()} - {symbol} - Skipped due to negative sentiment: {sentiment_score:.2f}\n")
                                continue

                            try:
                                account = api.get_account()
                                cash = float(account.cash)
                                qty = int(cash // price)
                            except Exception as e:
                                print(f"Error fetching cash balance: {e}")
                                qty = 0

                            if qty > 0:
                                entry_price = price
                                stop_loss = entry_price - atr_value * multiplier
                                take_profit = entry_price + atr_value * multiplier
                                trailing_stop_price = entry_price - atr_value * multiplier
                                active_trades[symbol] = {
                                    "entry": entry_price,
                                    "qty": qty,
                                    "trailing_stop": trailing_stop_price
                                }
                                send_notification(f"BUY {qty} of {symbol} at {entry_price:.2f} | SL: {stop_loss:.2f}, TP: {take_profit:.2f}, Trailing Stop: {trailing_stop_price:.2f}")
                                execute_trade(symbol, 'buy', qty)
                        else:
                            sma_check = bool(price > sma_200)
                            atr_check = bool(atr_value < atr_threshold)
                            with open("trade_rejections.log", "a") as log:
                                log.write(f"{dt.datetime.now()} - {symbol} - Trade not placed | RSI: {rsi:.2f}, SMA check: {sma_check}, ATR check: {atr_check}\n")

                        if float(rsi) > 70 and symbol in active_trades:
                            trade = active_trades[symbol]
                            send_notification(f"SELL signal (RSI > 70): Selling {trade['qty']} shares of {symbol} at {price:.2f}")
                            execute_trade(symbol, 'sell', trade["qty"])
                            del active_trades[symbol]
                        elif symbol in active_trades:
                            trade = active_trades[symbol]
                            new_stop = price - atr_value * multiplier
                            if new_stop > trade["trailing_stop"]:
                                trade["trailing_stop"] = new_stop
 
                            if float(price) < float(trade["trailing_stop"]):
                                send_notification(f"TRAILING STOP HIT: Selling {trade['qty']} shares of {symbol} at {price:.2f}")
                                execute_trade(symbol, 'sell', trade["qty"])
                                del active_trades[symbol]

                except Exception as e:
                    print(f"Error fetching data for {symbol}: {e}")
                    import traceback; traceback.print_exc()
                    try:
                        with open("trade_rejections.log", "a") as log:
                            log.write(f"{dt.datetime.now()} - {symbol} - Exception occurred: {str(e)}\n")
                    except Exception as log_error:
                        print(f"Failed to write to log for {symbol}: {log_error}")
            time.sleep(60)
        else:
            print("Market is closed. Sleeping for 5 minutes.")
            time.sleep(300)

tickers = ["AAPL", "MSFT", "TSLA", "NVDA", "GOOG", "AMZN", "META", "NFLX"]

if __name__ == "__main__":
    run_paper_trading(tickers, use_sl_tp=True)