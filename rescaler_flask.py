# rescaler_flask.py
from flask import Flask, jsonify
import numpy as np
import pymysql
import pickle
from sklearn.preprocessing import StandardScaler
from datetime import datetime

app = Flask(__name__)

SCALER_PATH = 'tcnflask.pkl'  # Make sure this is a StandardScaler, not just a numpy array

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
    try:
        with open(SCALER_PATH, 'rb') as f:
            scaler = pickle.load(f)
            if not hasattr(scaler, 'inverse_transform'):
                raise TypeError("Loaded object is not a valid scaler with 'inverse_transform'")
            return scaler
    except Exception as e:
        raise RuntimeError(f"Failed to load scaler: {e}")

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
        scaled = np.array([[float(row[0]), float(row[1]), float(row[2])]])
        rescaled = scaler.inverse_transform(scaled)[0]

        return jsonify({
            'temperature': round(float(rescaled[0]), 2),
            'humidity': round(float(rescaled[1]), 2),
            'pressure': round(float(rescaled[2]), 2)
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

    finally:
        if 'cursor' in locals(): cursor.close()
        if 'conn' in locals(): conn.close()

# --- MAIN ---
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
