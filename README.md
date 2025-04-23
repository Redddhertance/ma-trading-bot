# ma-trading-bot

Trading bot based on RSI moving average

Performs automated technical trading following real-time and historical data, incorporates moving average (RSI), volatility filtering (ATR) and long-term trend confirmation (200 day SMA) alongside sentiment analysis to create buy and sell signals which are then passed through the API to execute trades. I used Alpaca's paper trading features for my API hence the imported Alpaca library.

I conducted multiple back tests across multiple one-year periods, ranging from 2020-2024, to test and evaluate strategy performance in different market conditions in an effort to create a consistently profitable bot without overengineering it.

The backtesting engine is still included in the bot, though simplified to not include the graphs and projections it initially had (will upload at a later date)

Makes entries based on RSI < 40  and RSI > 70 for exits and then filters them based on the ATR, 200-day SMA and VADER sentiment analysis using news-api.

The bot includes trailing stop logic and shorting which has been commented out for simplicity and I have not refined the shorting strategy to make it consistently profitable as of yet.

Automatic trade logging and email notifications are also included (I set it to use an email I don't use to send messages to my main email through Google's API)

Compatible with .env for API security (alternative to hard-coding credentials for security reasons)

In my most recent update, I kept encountering issues with scalar values and empty/malformed dataframes during development, so I updated my rsi, atr_value, price and sma_200 variables to items and implemented a lot of safety checks to help me identify errors through exception output messages and the imported traceback library.
