import requests
from datetime import datetime
from statsmodels.tsa.statespace.sarimax import SARIMAX
from flask import Flask, render_template, request
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import pandas as pd

app = Flask(__name__)


def plot_graph(price_data, prediction, symbol):
    # Plot time series of prices and prediction
    fig, ax = plt.subplots()
    ax.plot(price_data['close'], label="Price")
    ax.plot(price_data['ma'], label="Moving Average")
    ax.plot(prediction, label="Prediction")
    ax.text(prediction.index[0], prediction[0], f'Predicted price:\n${prediction[-1]:0.2f}')
    plt.title(f'Predictions for: {symbol}')
    plt.xticks(rotation=30)
    plt.grid()
    ax.legend()

    # Convert plot to image
    image = BytesIO()
    plt.savefig(image, format="png")
    image.seek(0)

    # Encode image as base64 string
    image_base64 = base64.b64encode(image.read()).decode("utf-8")
    return image_base64


def load_prices(symbol, days=None):
    print('Getting data...')
    if days is None:
        days = 30
    else:
        days = int(days)
        if days > 30:
            days = 30
            print('Cant retreive data with more than 30 days old')

    # Retrieve the historical data for the cryptocurrency
    response = requests.get(
        f'https://api.coingecko.com/api/v3/coins/{symbol}/market_chart?vs_currency=usd&days={days}&interval=daily')
    data = response.json()

    # Convert the data to a pandas dataframe
    prices = pd.DataFrame(data['prices'], columns=['timestamp', 'close'])

    # Filter data to current date.
    # Filter data to current date.
    prices['timestamp'] = pd.to_datetime(prices['timestamp'], unit='ms')
    prices.sort_values(by='timestamp', ascending=True, inplace=True)

    # Filter the data to include only the data before the specified date
    prices = prices.loc[prices.timestamp <= datetime.today().replace(
        hour=0, minute=0, second=0, microsecond=0)]

    prices.set_index('timestamp', inplace=True)
    featured_prices = create_features(prices)
    print('Data getted')
    return featured_prices


def create_features(df):
    # Calculate the moving average
    df['ma'] = df['close'].rolling(window=14).mean()

    # Calculate the moving standard deviation
    df['std'] = df['close'].rolling(window=14).std()

    # Calculate the relative strength index (RSI)
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df['rsi'] = 100 - (100 / (1 + rs))

    # Calculate the rate of change (ROC)
    n = 14
    M = df['close'].diff(n)
    N = df['close'].shift(n)
    df['roc'] = M / N

    # Calculate the momentum
    df['momentum'] = df['close'] - df['close'].shift(14)
    df.fillna(method='bfill', inplace=True)
    df.fillna(method='ffill', inplace=True)

    return df


def sarima_predict(df):
    # Train the SARIMA model
    p, d, q = (0, 0, 0)
    P, D, Q, s = (0, 0, 1, 7)
    # model = SARIMAX(prices['close'], order=(p, d, q), seasonal_order=(P, D, Q, s))
    model = SARIMAX(endog=df['close'], exog=df[['momentum', 'roc', 'rsi', 'ma']],
                    order=(p, d, q), seasonal_order=(P, D, Q, s))
    model_fit = model.fit(disp=False)

    # Make a prediction for the next time step
    next_day = df.index[-1] + pd.Timedelta(days=1)

    prediction = model_fit.predict(
        start=df.index[-5], end=next_day, dynamic=False, exog=model_fit.model.exog[-1:])
    return prediction


@app.route("/predict", methods=["GET"])
def predict():
    symbol = request.args.get("symbol")

    price_data = load_prices(symbol)
    prediction = sarima_predict(price_data)
    image_base64 = plot_graph(price_data, prediction, symbol)
    return render_template("prediction_result.html", image_base64=image_base64)


@app.route('/')
def index():
    symbols = ['bitcoin', 'ethereum', 'kleros']
    return render_template('index.html', symbols=symbols)


if __name__ == "__main__":
    app.run(debug=True)
