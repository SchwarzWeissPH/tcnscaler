# rescaler_flask.py
from flask import Flask, jsonify
import numpy as np
import pymysql
import pickle, gzip
from keras.models import load_model
from keras.utils import custom_object_scope
from tcn import TCN
from datetime import datetime

app = Flask(__name__)

MODEL_PATH = 'tcnflask.keras'
SCALER_PATH = 'tcnflask.pkl'  # Add this if you're rescaling

DB_CONFIG = {
    'host': '118.139.162.228',
    'user': 'drei',
    'password': 'madalilangto',
    'database': 'signup'
}

def connect_to_database():
    try:
        conn = pymysql.connect(**DB_CONFIG, connect_timeout=60, read_timeout=60)
        return conn
    except Exception as e:
        print(f"[ERROR] Database connection failed: {e}")
        return None

def load_scaler():
    with gzip.open(SCALER_PATH, 'rb') as f:
        return pickle.load(f)

@app.route('/rescaled', methods=['GET'])
def rescale_latest_prediction():
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

        scaler = load_scaler()
        scaled = np.array(row).reshape(1, -1)
        rescaled = scaler.inverse_transform(scaled)[0]

        return jsonify({
            'temperature': float(rescaled[0]),
            'humidity': float(rescaled[1]),
            'pressure': float(rescaled[2])
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

    finally:
        if 'cursor' in locals(): cursor.close()
        if 'conn' in locals(): conn.close()

# --- MAIN ---
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
