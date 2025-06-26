import os
import pandas as pd
import numpy as np
import tensorflow as tf
import yfinance as yf
from firebase_functions import https_fn
import firebase_admin
from firebase_admin import credentials, storage
from tensorflow.keras.models import load_model
import json
from firebase_admin import firestore
from datetime import datetime, timedelta, time
import pytz

print("Cloud Function TensorFlow version:", tf.__version__)

# Firebase Admin SDK Initialization
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SERVICE_ACCOUNT_PATH = os.path.join(BASE_DIR, "proiectcorect-b09a1-firebase-adminsdk-9vrwx-b34483d1a9.json")

if not firebase_admin._apps:
    cred = credentials.Certificate(SERVICE_ACCOUNT_PATH)
    firebase_admin.initialize_app(cred, {"storageBucket": "proiectcorect-b09a1.firebasestorage.app"})

def update_csv_in_firebase(csv_path: str, firebase_blob_path: str, ticker: str = "^GSPC", period: str = "6mo", interval: str = "1d"):
    """
    Descarcă datele actualizate de la Yahoo Finance, salvează local și urcă în Firebase.
    Salvează doar ultimele 30 de zile în Firestore.
    """
    print(f" Descarc cele mai noi date pentru {ticker}...")
    df = yf.download(ticker, period=period, interval=interval)
    df.to_csv(csv_path)
    print(f" CSV salvat local: {csv_path}")

    today = datetime.now().date()
    cutoff_date = today - timedelta(days=30)
    db = firestore.client()

    ny_tz = pytz.timezone("America/New_York")
    ro_tz = pytz.timezone("Europe/Bucharest")
    now_ny = datetime.now(ny_tz)
    now_ro = datetime.now(ro_tz)
    closing_time_ny = time(16, 0)

    for index, row in df.iterrows():
        row_date = pd.to_datetime(index).date()

        if row_date < cutoff_date:
            continue

        if row_date > today:
            continue

        if row_date == today:
            if now_ny.time() < closing_time_ny:
                print(f" {row_date} este azi, dar piața NY nu s-a închis încă → ignorăm")
                continue
            else:
                print(f" {row_date} este azi, dar piața NY s-a închis → salvăm")

        closing_price = float(row["Close"])
        closing_date_str = row_date.strftime("%d.%m.%Y")
        doc_ref = db.collection("real_prices").document(closing_date_str)
        doc_ref.set({"closingPrice": closing_price})
        print(f" closingPrice salvat în Firestore: {closing_date_str} → {closing_price}")

    bucket = storage.bucket()
    blob = bucket.blob(firebase_blob_path)
    blob.upload_from_filename(csv_path)
    print(f" CSV actualizat și urcat pe Firebase Storage: {firebase_blob_path}")




@https_fn.on_request()
def get_stock_prediction(req: https_fn.Request) -> https_fn.Response:
    try:
        print(" Function Execution Started")

        bucket = storage.bucket()
        model_blob = bucket.blob("model_sp500.h5")

        model_path = os.path.join(BASE_DIR, "model_sp500.h5")
        csv_path = os.path.join(BASE_DIR, "SP500.csv")

        #  Actualizez fișierul CSV și îl urc în Firebase Storage
        update_csv_in_firebase(csv_path=csv_path, firebase_blob_path="SP500.csv")

        #  CSV-ul local e actualizat – îl folosesc mai departe
        df = pd.read_csv(csv_path)
        df = df.reset_index()  # mută indexul datetime în coloane
        if 'Close' not in df.columns:
            print(" CSV missing 'Close' column")
            return https_fn.Response("CSV file does not contain 'Close' column.", status=500)

        #  elimină rândurile cu valori non-numerice
        df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
        df = df.dropna(subset=['Close'])
        print(" CSV file successfully loaded")

        #  Pregătire date
        close_prices = df['Close'].values.reshape(-1, 1)
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(close_prices)
        look_back = 60
        X_input = scaled_data[-look_back:].reshape(1, look_back, 1)

        #  Model
        if not os.path.exists(model_path):
            if not model_blob.exists():
                print(" Model file missing in Firebase Storage")
                return https_fn.Response("Model does not exist in Firebase Storage.", status=500)
            model_blob.download_to_filename(model_path)
            print(" Model file downloaded")

        model = load_model(model_path)
        print(" Model loaded successfully!")

        #  Predictie
        prediction = model.predict(X_input)
        predicted_price = scaler.inverse_transform(prediction)[0, 0]
	#  Adaug mesajul cu data actuală (în interiorul funcției)
        eastern = pytz.timezone("Europe/Bucharest")
        current_date = datetime.now(eastern).strftime("%d-%m-%Y")
        response_text = f"Prețul prezis pentru data de {current_date} este: {predicted_price}"

        return https_fn.Response(json.dumps({
            "predicted_price": float(predicted_price),
            "message": response_text
        }), mimetype="application/json")

    except Exception as e:
        print(f" Error occurred: {str(e)}")
        return https_fn.Response(f"Error: {str(e)}", status=500)

if __name__ == "__main__":
    from firebase_functions.https_fn import Request

    mock_request = Request({"method": "GET"})
    response = get_stock_prediction(mock_request)
    print(response.data)
