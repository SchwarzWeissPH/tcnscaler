# rescaler_flask.py
from flask import Flask, jsonify
import numpy as np
import pymysql
import pickle
import threading
import time
from sklearn.preprocessing import StandardScaler
from datetime import datetime

app = Flask(__name__)

SCALER_PATH = 'weather_scaler_standard.pkl'

DB_CONFIG = {
    'host': '118.139.162.228',
    'user': 'drei',
    'password': 'madalilangto',
    'database': 'signup'
}

def connect_to_database():
    try:
        return pymysql.connect(**DB_CONFIG, connect_timeout=60, read_timeout=60)
    except Exception as e:
        print(f"[ERROR] DB connection failed: {e}")
        return None

def load_scaler():
    with open(SCALER_PATH, 'rb') as f:
        return pickle.load(f)

def fetch_and_rescale_and_save():
    try:
        conn = connect_to_database()
        if not conn:
            print("[ERROR] DB connection failed.")
            return
        cursor = conn.cursor()

        # Fetch the latest scaled prediction
        cursor.execute("""
            SELECT predicted_value_1, predicted_value_2, predicted_value_3
            FROM signup_predictions
            ORDER BY timestamp DESC
            LIMIT 1
        """)
        row = cursor.fetchone()

        if not row:
            print("[ERROR] No predictions found.")
            return

        # Rescale
        scaler: StandardScaler = load_scaler()
        scaled = np.array([[float(row[0]), float(row[1]), float(row[2])]])
        rescaled = scaler.inverse_transform(scaled)[0]

        temperature = round(float(rescaled[0]), 2)
        humidity = round(float(rescaled[1]), 2)
        pressure = round(float(rescaled[2]), 2)

        # Save to prediction table
        cursor.execute("""
            INSERT INTO prediction (tcn_temperature, tcn_humidity, tcn_pressure, timestamp)
            VALUES (%s, %s, %s, %s)
        """, (
            temperature,
            humidity,
            pressure,
            datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        ))
        conn.commit()
        print(f"[INFO] Saved new prediction: {temperature}°C, {humidity}%, {pressure}hPa")

    except Exception as e:
        print(f"[ERROR] Background job error: {e}")

    finally:
        if 'cursor' in locals(): cursor.close()
        if 'conn' in locals(): conn.close()

def background_loop():
    while True:
        fetch_and_rescale_and_save()
        time.sleep(60)  # wait 60 seconds before running again

@app.route('/rescaled', methods=['GET'])
def manual_rescale():
    try:
        conn = connect_to_database()
        if not conn:
            return jsonify({'error': 'DB connection failed'}), 500
        cursor = conn.cursor()

        cursor.execute("""
            SELECT predicted_value_1, predicted_value_2, predicted_value_3
            FROM signup_predictions
            ORDER BY timestamp DESC
            LIMIT 1
        """)
        row = cursor.fetchone()

        if not row:
            return jsonify({'error': 'No predictions found'}), 404

        scaler: StandardScaler = load_scaler()
        scaled = np.array([[float(row[0]), float(row[1]), float(row[2])]])
        rescaled = scaler.inverse_transform(scaled)[0]

        temperature = round(float(rescaled[0]), 2)
        humidity = round(float(rescaled[1]), 2)
        pressure = round(float(rescaled[2]), 2)

        # --- ADD: Save to database manually too ---
        cursor.execute("""
            INSERT INTO prediction (tcn_temperature, tcn_humidity, tcn_pressure, timestamp)
            VALUES (%s, %s, %s, %s)
        """, (
            temperature,
            humidity,
            pressure,
            datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        ))
        conn.commit()

        return jsonify({
            'temperature': temperature,
            'humidity': humidity,
            'pressure': pressure,
            'message': 'Manual rescale successful and saved to database'
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

    finally:
        if 'cursor' in locals(): cursor.close()
        if 'conn' in locals(): conn.close()



# Start background loop even when deployed via Gunicorn
threading.Thread(target=background_loop, daemon=True).start()
