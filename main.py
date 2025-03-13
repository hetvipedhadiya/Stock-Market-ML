from flask import Flask, request, render_template, jsonify
import numpy as np
import pandas as pd
import joblib

app = Flask(__name__)

# Load the trained model
model = joblib.load('stock_model.pkl')

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        try:
            data = [
                float(request.form['open']),
                float(request.form['high']),
                float(request.form['low']),
                float(request.form['close']),
                float(request.form['adj_close']),
                int(request.form['year']),
                int(request.form['month']),
                int(request.form['day']),
                int(request.form['day_of_week']),
                int(request.form['is_weekend']),
                int(request.form['quarter'])
            ]

            input_data = pd.DataFrame([data], columns=[
                'Open', 'High', 'Low', 'Close', 'Adj Close',
                'Year', 'Month', 'Day', 'Day_of_Week', 'Is_Weekend', 'Quarter'
            ])

            prediction = model.predict(input_data)[0]
            return jsonify({'prediction': prediction})

        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    # Render HTML for GET request
    return render_template("index.html")

if __name__ == '__main__':
    app.run(debug=True)
