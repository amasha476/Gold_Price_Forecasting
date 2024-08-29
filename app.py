from flask import Flask, render_template, request, make_response
import tensorflow as tf
import numpy as np
from sklearn.preprocessing import StandardScaler
import pandas as pd
import matplotlib.pyplot as plt
import io
import base64
import matplotlib
matplotlib.use('Agg')


app = Flask(__name__)

def get_forecast_values(start_date, end_date):
    # Initialize the StandardScaler
    scaler = StandardScaler()
    
    # Load the model
    model = tf.keras.models.load_model('model_LSTM_WB_DS.h5')

    
    # Load the test data
    dataset_test = np.load('dataset_test.npy')
    
    # Scaling the testing dataset
    scaler.fit(dataset_test)
    
    # Load the last sequence of the test set
    last_sequence = np.load('last_sequence.npy')
    
    # Calculate the number of days for forecasting
    number_of_days = (pd.to_datetime(end_date) - pd.to_datetime(start_date)).days
    
    # Initialize a list to store the forecasts
    forecasts = []

    # Forecast for the next number_of_days
    for _ in range(number_of_days):
        # Predict the next value
        next_value = model.predict(last_sequence)

        # Append the forecast to the list
        forecasts.append(next_value[0, 0])

        # Update the sequence: remove the first value and append the new prediction
        last_sequence = np.append(last_sequence[:, 1:, :], np.reshape(next_value, (1, 1, 1)), axis=1)

    # Inverse transform the forecasts to get the actual values
    forecasts = scaler.inverse_transform(np.array(forecasts).reshape(-1, 1))
    
    # Generate the date range for the actual test set
    dates_actual = pd.date_range(start='2021-11-14', end='2023-09-29', freq='D')
    
    # Generate the date range for the forecast period
    dates_forecast = pd.date_range('2023-09-30', periods=number_of_days + 1, freq='D')

    # Load the actual test values
    actual_test_values = np.load('Orginal_test_data.npy')
    
    # Combine the last point of the actual test values with the forecast
    forecast_values = np.concatenate(([actual_test_values[-1]], forecasts))
    
    # Plotting
    plt.figure(figsize=(12, 6))
    plt.plot(dates_actual, actual_test_values, label='Actual Gold Price Values', color='blue')
    plt.plot(dates_forecast, forecast_values, label='Forecast', color='orange')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title('Gold Price Forecasts')
    plt.legend()
    plt.grid(True)

    # Save the plot to a BytesIO object and encode it to base64
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plt.close()
    
    # Convert image to base64 string
    img_base64 = base64.b64encode(img.getvalue()).decode('utf-8')

    # saving the forecasted results to a dataframe
    forecasted_data = pd.DataFrame({
        'Date': dates_forecast,
        'Forecasted Daily Gold Price (US $ per troy ounce)': forecast_values.flatten()})
    
    
    return img_base64, forecasted_data

# @app.route('/HistoricalTrend')
# def historical():
#     return render_template('HistoricalTrends.html')

@app.route('/', methods=['GET', 'POST'])
def index():
    img_base64 = None
    table_html = None
    if request.method == 'POST':
        start_date = request.form['start_date']
        end_date = request.form['end_date']
        img_base64, forecasted_data = get_forecast_values(start_date, end_date)
        table_html = forecasted_data.to_html(index=False)
    return render_template('index.html', img_base64=img_base64, table_html= table_html)


@app.route('/export', methods=['POST'])
def export():
    # Retrieve the forecasted data from the form submission
    start_date = request.form['start_date']
    end_date = request.form['end_date']
    _, forecasted_data = get_forecast_values(start_date, end_date)

    # Convert the DataFrame to CSV format
    csv_data = forecasted_data.to_csv(index=False)

    # Create a response with the CSV data
    response = make_response(csv_data)
    response.headers['Content-Disposition'] = 'attachment; filename=forecasted_data.csv'
    response.headers['Content-Type'] = 'text/csv'
    
    return response
if __name__ == '__main__':
    app.run(debug=True)

