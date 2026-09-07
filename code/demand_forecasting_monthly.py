# ---------------------------
# Core Libraries
# ---------------------------
import os
import sys
import gc
import math
import warnings
from collections import Counter, defaultdict
import logging
import tqdm

# ---------------------------
# Data Manipulation and Analysis
# ---------------------------
import pandas as pd
import numpy as np

# ---------------------------
# Database Management
# ---------------------------
from pymongo import MongoClient, UpdateOne

# ---------------------------
# Machine Learning and Clustering
# ---------------------------
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, ParameterGrid
from sklearn.metrics import silhouette_score, mean_absolute_error

# ---------------------------
# Time Series Analysis and Forecasting
# ---------------------------
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet

# ---------------------------
# Deep Learning Models
# ---------------------------
import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout

# ---------------------------
# Parallel and Concurrent Processing
# ---------------------------
from joblib import Parallel, delayed
from concurrent.futures import ThreadPoolExecutor, as_completed

# ---------------------------
# Market Basket Analysis
# ---------------------------
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

# ---------------------------
# Combinatorics
# ---------------------------
import itertools

# ---------------------------
# User Interface
# ---------------------------
import gradio as gr

warnings.filterwarnings("ignore")

warnings.filterwarnings("ignore", message="Non-invertible starting MA parameters found.")
warnings.filterwarnings("ignore", message="Non-stationary starting autoregressive parameters found.")


# Suppress TensorFlow warnings and info logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # '3' means to suppress all logs (error only)
tf.get_logger().setLevel('ERROR')
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.WARNING)  # Adjust to ERROR or INFO as needed

# Connect to MongoDB
client = MongoClient("mongodb://localhost:27017/")
db = client["final_project"]

# Initialize MongoDB collections
sales_collection = db["sales_history"]
prediction_collection = db["prediction_history"]
MBA_collection = db["MBA_history"]
model_collection = db["model_history"]



class Clustering:
    def __init__(self, unique_products, max_k=None):
        self.unique_products = unique_products
        self.max_k = max_k or len(unique_products) - 1
        self.tfidf_vectorizer = TfidfVectorizer()
        self.tfidf_matrix = None
        self.optimal_clusters = None
        self.kmeans = None
        self.labels = None
        self.clusters = {}
        self.cluster_names = {}
        self.combined_clusters = defaultdict(list)
    
    def vectorize(self):
        """Convert strings to numerical vectors using TF-IDF."""
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.unique_products)
    
    def find_optimal_clusters(self):
        """Determine the optimal number of clusters using the Elbow Method and Silhouette Score."""
        inertias = []
        silhouette_scores = []
        k_values = range(2, self.max_k + 1)
        
        for k in k_values:
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(self.tfidf_matrix)
            inertias.append(kmeans.inertia_)
            silhouette_scores.append(silhouette_score(self.tfidf_matrix, kmeans.labels_))
        
        # Select the optimal number of clusters based on the highest silhouette score
        self.optimal_clusters = k_values[silhouette_scores.index(max(silhouette_scores))]
    
    def fit_kmeans(self):
        """Run KMeans clustering with the optimal number of clusters."""
        self.kmeans = KMeans(n_clusters=self.optimal_clusters, random_state=42)
        self.kmeans.fit(self.tfidf_matrix)
        self.labels = self.kmeans.labels_
    
    def assign_clusters(self):
        """Assign products to clusters and determine cluster names."""
        self.clusters = {i: [] for i in range(self.optimal_clusters)}
        for i, label in enumerate(self.labels):
            self.clusters[label].append(self.unique_products[i])
        
        # Assign names to clusters based on the most common item in each cluster
        for cluster, contents in self.clusters.items():
            most_common_item = Counter(contents).most_common(1)[0][0]
            self.cluster_names[cluster] = most_common_item.split()[0]  # Use the first word as the cluster name
    
    def combine_clusters(self):
        """Combine clusters based on their assigned names."""
        for idx, items in self.clusters.items():
            name = self.cluster_names.get(idx, None)
            if name:
                self.combined_clusters[name].extend(items)
    
    def get_combined_clusters(self):
        """Get combined clusters as a dictionary."""
        combined_result = [(name, items) for name, items in self.combined_clusters.items()]
        return dict(combined_result)
    
    def run(self):
        """Execute the full clustering process."""
        self.vectorize()
        self.find_optimal_clusters()
        self.fit_kmeans()
        self.assign_clusters()
        self.combine_clusters()
        return self.get_combined_clusters()

class SARIMAX_pred:
    def __init__(self, product_dfs, period=12):
        self.product_dfs = product_dfs
        self.period = period
        self.results_df = None
        self.SARIMAX_results = []
    
    def detect_seasonality(self, data, column):
        try:
            decomposition = seasonal_decompose(data[column], model='multiplicative', period=self.period)
            seasonal_component = decomposition.seasonal
        except:
            return "Data Tidak Musiman"
        
        ljung_box_result = acorr_ljungbox(data[column], lags=[self.period], return_df=True)
        p_value = ljung_box_result['lb_pvalue'].values[0]
        
        return "Data Musiman" if p_value < 0.05 else "Data Tidak Musiman"
    
    def evaluate_seasonality(self):
        results = []
        for data in self.product_dfs:
            item_name = data['Nama Item'].iloc[0]
            seasonality_result = self.detect_seasonality(data, 'Jumlah Item')
            results.append({'Nama Item': item_name, 'Seasonality Result': seasonality_result})
        self.results_df = pd.DataFrame(results)
    
    def fit_sarimax(self, train_data, exog_train, param, seasonal_param=None):
        try:
            model = SARIMAX(train_data, order=param, seasonal_order=seasonal_param, exog=exog_train)
            result = model.fit(disp=False, maxiter=200)
            predictions = result.predict(start=0, end=len(train_data)-1, exog=exog_train)
            mae = mean_absolute_error(train_data, predictions)
            bias = np.mean(predictions - train_data)
            error = mae + abs(bias)
            return result, param, seasonal_param, error
        except Exception:
            return None, param, seasonal_param, float('inf')
    
    def sarimax_grid_search_parallel(self, train_data, exog_train, param_grid, seasonal_grid=None, n_jobs=-1):
        param_combinations = list(itertools.product(param_grid, seasonal_grid)) if seasonal_grid else [(param, None) for param in param_grid]
        
        results = Parallel(n_jobs=n_jobs)(
            delayed(self.fit_sarimax)(train_data, exog_train, param, seasonal_param)
            for param, seasonal_param in param_combinations
        )
        
        best_model, best_param, best_seasonal_param, best_error = min(results, key=lambda x: x[3])
        return best_model, (best_param, best_seasonal_param), best_error
    
    def process_data_item(self, data):
        target = data['Jumlah Item']
        exog = data[['exog1', 'exog2']]
        nama_item = data["Nama Item"].iloc[0]
        
        seasonality_row = self.results_df[self.results_df['Nama Item'] == nama_item]
        if not seasonality_row.empty:
            is_seasonal = seasonality_row['Seasonality Result'].values[0]
        else:
            return None
        
        train_size = int(len(target) * 0.8)
        train_target, test_target = target[:train_size], target[train_size:]
        train_exog, test_exog = exog[:train_size], exog[train_size:]
        
        p = d = q = range(0, 3)  # Limiting to range(0, 2) for speed
        seasonal_grid = list(itertools.product(range(0, 2), range(0, 2), range(0, 2), [12])) if is_seasonal == "Data Musiman" else None
        param_grid = list(itertools.product(p, d, q))
        
        best_model, best_params, best_error_train = self.sarimax_grid_search_parallel(train_target, train_exog, param_grid, seasonal_grid, n_jobs=-1)
        
        forecast = best_model.predict(start=test_target.index[0], end=test_target.index[-1], exog=test_exog)
        mae_sarimax = mean_absolute_error(test_target, forecast)
        bias_sarimax = np.mean(forecast - test_target)
        error = mae_sarimax + abs(bias_sarimax)

        order, seasonal_order = best_params

        if exog['exog1'].eq(0).all() and exog['exog2'].eq(0).all():
            best_model = SARIMAX(target, order=order, seasonal_order=seasonal_order).fit(disp=False, maxiter=200)
        else:
            best_model = SARIMAX(target, order=order, seasonal_order=seasonal_order, exog=exog).fit(disp=False, maxiter=200)
        
        return {
            'Nama Item': nama_item, 
            'ERROR (SARIMAX)': error, 
            'Model (SARIMAX)': best_model}

    def run(self):
        self.evaluate_seasonality()
        
        # Use Parallel processing for data items
        results = Parallel(n_jobs=-1)(
            delayed(self.process_data_item)(data) for data in tqdm.tqdm(self.product_dfs, desc="Processing")
        )
        
        # Filter out None results
        self.SARIMAX_results = pd.DataFrame([result for result in results if result])
        
        return self.SARIMAX_results

class Prophet_pred:
    def __init__(self, product_dfs, param_grid=None):
        self.product_dfs = product_dfs
        self.param_grid = param_grid or {
            'changepoint_prior_scale': [0.001, 0.01, 0.1, 0.5],
            'seasonality_prior_scale': [0.01, 0.1, 1.0, 10.0],
            'seasonality_mode': ['additive', 'multiplicative']
        }
        self.Prophet_results = []

    def fit_prophet_with_params(self, train_data, exog_columns, params):
        model = Prophet(**params)
        for exog_col in exog_columns:
            model.add_regressor(exog_col)
        model.fit(train_data)
        future = train_data.drop(columns=['y'])
        forecast = model.predict(future)
        forecast['yhat'] = forecast['yhat'].apply(lambda x: max(0, x))  
        y_pred = forecast['yhat']
        mae = mean_absolute_error(train_data['y'], y_pred)
        bias = np.mean(y_pred - train_data['y'])
        score = mae + abs(bias)
        return model, score

    def grid_search_prophet_sequential(self, train_data, exog_columns):
        results = []  # To store results (model, params, score)

        # Iterate through each parameter combination in the parameter grid
        for params in ParameterGrid(self.param_grid):
            # Train Prophet with the given params
            model, score = self.fit_prophet_with_params(train_data, exog_columns, params)
            results.append((model, params, score))

        # Find the best result (lowest score)
        best_model, best_params, best_score = min(results, key=lambda x: x[2])

        return best_model, best_params, best_score


    
    def process_data_item(self, data, index, total_items):
        target = data['Jumlah Item']
        exog = data[['exog1', 'exog2']]
        train_size = int(len(target) * 0.8)
        train_target, test_target = target[:train_size], target[train_size:]
        train_exog, test_exog = exog[:train_size], exog[train_size:]
        train_data = train_target.reset_index(drop=False)
        train_data.columns = ['ds', 'y']
        train_data['ds'] = pd.to_datetime(train_data['ds']).dt.to_period('M').dt.to_timestamp()
        train_data['floor'] = 0  
        train_data = pd.merge(train_data, train_exog, left_on='ds', right_index=True)
        
        # Instantiate Prophet model within the function
        best_model, best_params, best_score = self.grid_search_prophet_sequential(train_data, exog.columns)
        
        test_data = test_target.reset_index(drop=False)
        test_data.columns = ['ds', 'y']
        test_data['ds'] = pd.to_datetime(test_data['ds']).dt.to_period('M').dt.to_timestamp()
        test_data = pd.merge(test_data, test_exog, left_on='ds', right_index=True)
        future_test = test_data.drop(columns=['y'])
        test_forecast = best_model.predict(future_test)
        test_forecast['yhat'] = test_forecast['yhat'].apply(lambda x: max(0, x))
        test_y_pred = test_forecast['yhat']
        test_mae = mean_absolute_error(test_data['y'], test_y_pred)
        test_bias = np.mean(test_y_pred - test_data['y'])
        test_error = test_mae + abs(test_bias)
        
        target = target.reset_index(drop=False)
        target.columns = ['ds', 'y']
        target['ds'] = pd.to_datetime(target['ds']).dt.to_period('M').dt.to_timestamp()
        target['floor'] = 0
        target = pd.merge(target, exog, left_on='ds', right_index=True)
        
        # Retrain best model with full dataset
        final_model = Prophet(**best_params)
        for exog_col in exog.columns:
            final_model.add_regressor(exog_col)
        final_model.fit(target)
        
        # Display progress
        progress = (index + 1) / total_items * 100
        sys.stdout.write(f"\rProcessing: {progress:.2f}% complete")
        sys.stdout.flush()
        
        return {
            'Nama Item': data['Nama Item'].iloc[0],
            'ERROR (Prophet)': test_error,
            'Model (Prophet)': final_model
        }

    def run(self):
        total_items = len(self.product_dfs)
        
        results = Parallel(n_jobs=-1, backend="loky")(
            delayed(self.process_data_item)(data, i, total_items) for i, data in enumerate(self.product_dfs)
        )
        
        print("\nProcessing complete.")
        self.Prophet_results = pd.DataFrame(results)
        return self.Prophet_results

class LSTM_pred:
    def __init__(self, product_dfs, time_steps=12):
        self.product_dfs = product_dfs
        self.time_steps = time_steps
        self.units_options = [50, 100]
        self.dropout_options = [0.2, 0.3]
        self.batch_size_options = [16, 32]
        self.epochs_options = [10, 20]
        self.param_combinations = list(itertools.product(self.units_options, self.dropout_options, self.batch_size_options, self.epochs_options))
        self.LSTM_results = []

    def create_sequences(self, data):
        # Ensure X and y have consistent lengths
        X = np.array([data[i:i + self.time_steps, 1:] for i in range(len(data) - self.time_steps)])
        y = data[self.time_steps:, 0]
        return X, y

    def create_tf_dataset(self, X, y, batch_size):
        # Use TensorFlow's data pipeline for efficient batching and prefetching
        dataset = tf.data.Dataset.from_tensor_slices((X, y))
        dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        return dataset

    def build_lstm_model(self, input_shape, units=50, dropout_rate=0.2):
        # Define the LSTM model
        model = Sequential([
            LSTM(units=units, input_shape=input_shape),
            Dropout(dropout_rate),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        return model

    def evaluate_model(self, params, X_train, y_train, X_test, y_test, input_shape):
        units, dropout_rate, batch_size, epochs = params
        model = self.build_lstm_model(input_shape, units=units, dropout_rate=dropout_rate)
        
        train_dataset = self.create_tf_dataset(X_train, y_train, batch_size)
        test_dataset = self.create_tf_dataset(X_test, y_test, batch_size)

        model.fit(train_dataset, epochs=epochs, verbose=0, validation_data=test_dataset)
        val_loss = model.evaluate(test_dataset, verbose=0)
        return val_loss, model

    def process_single_df(self, df):
        # Scale the data
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(df[['Jumlah Item', 'exog1', 'exog2']])
        scaled_df = pd.DataFrame(scaled_data, columns=['Jumlah Item', 'exog1', 'exog2'], index=df.index)

        # Create sequences
        X, y = self.create_sequences(scaled_df.values)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        input_shape = (X_train.shape[1], X_train.shape[2])

        # Evaluate all parameter combinations
        results = [self.evaluate_model(params, X_train, y_train, X_test, y_test, input_shape) for params in self.param_combinations]

        # Find the best model
        best_loss, best_model = min(results, key=lambda x: x[0])

        # Predict and calculate metrics
        y_pred = best_model.predict(X_test)
        y_test_rescaled = scaler.inverse_transform(np.concatenate((y_test.reshape(-1, 1), X_test[:, -1, :]), axis=1))[:, 0]
        y_pred_rescaled = scaler.inverse_transform(np.concatenate((y_pred, X_test[:, -1, :]), axis=1))[:, 0]
        mae = mean_absolute_error(y_test_rescaled, y_pred_rescaled)
        bias = np.mean(y_pred_rescaled - y_test_rescaled)
        error = mae + abs(bias)

        return {'Nama Item': df['Nama Item'].iloc[0], 'ERROR (LSTM)': error, 'Model (LSTM)': best_model}

    def run(self):
        # Parallelize the processing of each DataFrame
        results = Parallel(n_jobs=-1, backend="loky")(
            delayed(self.process_single_df)(df) for df in self.product_dfs
        )
        self.LSTM_results = pd.DataFrame(results)
        return self.LSTM_results
      
class NaiveForecast:
    def __init__(self, dataframes, split_ratio=0.8):
        """
        Initialize the NaiveForecast class.
        
        Parameters:
        - dataframes: List of DataFrames to apply naive forecasting on.
        - split_ratio: Ratio to split the data into training and testing sets (default is 0.8).
        """
        self.dataframes = dataframes
        self.split_ratio = split_ratio

    def naive_forecast_and_err(self, df):
        """
        Apply naive forecasting and calculate MAPE for a single DataFrame.
        
        Parameters:
        - df: A DataFrame with columns 'Nama Item' and 'Jumlah Item'.
        
        Returns:
        - A DataFrame with 'Nama Item' and calculated 'ERROR (Naive)'.
        """
        # Sort by YearMonth to ensure the order is correct
        df = df.sort_values(by=['Nama Item']).sort_index()
        
        # Split the data into training and test sets
        split_index = int(len(df) * self.split_ratio)
        test = df.iloc[split_index:]
        
        # Use the last training value to predict for each Nama Item in the test set
        test['Naive Prediction'] = test.groupby('Nama Item')['Jumlah Item'].shift(1)
        
        # Fill NaN in predictions with the last valid training value or 0 if unavailable
        test['Naive Prediction'] = test.groupby('Nama Item')['Naive Prediction'].fillna(method='ffill').fillna(0)
        
        # Calculate MAE and BIAS for each 'Nama Item'
        item_metrics = test.groupby('Nama Item').apply(
            lambda x: pd.Series({
                # 'MAE (Naive)': mean_absolute_error(x['Jumlah Item'], x['Naive Prediction']),
                # 'BIAS (Naive)': np.mean(x['Naive Prediction'] - x['Jumlah Item']),
                'ERROR (Naive)': mean_absolute_error(x['Jumlah Item'], x['Naive Prediction']) + abs(np.mean(x['Naive Prediction'] - x['Jumlah Item']))
            })
        ).reset_index()
        
        item_metrics.columns = ['Nama Item', 'ERROR (Naive)']
        
        return item_metrics

    def run(self):
        """
        Apply naive forecasting to all DataFrames in the list.
        
        Returns:
        - A concatenated DataFrame of results for each DataFrame in the list.
        """
        results = pd.concat([self.naive_forecast_and_err(df) for df in self.dataframes], ignore_index=True)
        return results

class ForecastGenerator:
    def __init__(self, combined_results, product_dfs, future_steps=12):
        self.combined_results = combined_results
        self.product_dfs = product_dfs
        self.future_steps = future_steps
        self.predictions = []

    def get_product_df(self, nama_item):
        """Retrieve the DataFrame corresponding to the given 'Nama Item'."""
        return next(d for d in self.product_dfs if d['Nama Item'].iloc[0] == nama_item)

    def sarimax_forecast(self, row):
        df = self.get_product_df(row['Nama Item'])
        start_date = df.index[-1] + pd.DateOffset(months=1)
        date_range = pd.date_range(start=start_date, periods=self.future_steps, freq='MS')
        df_reset = df.reset_index()
        
        if row['exog']:
            # Fit ARIMA models to exogenous variables
            exog_models = {}
            for col in ['exog1', 'exog2']:
                model = ARIMA(df_reset[col], order=(1,1,1))  # Use a simple ARIMA(1,1,1) model
                model_fit = model.fit()
                exog_models[col] = model_fit
            
            # Forecast exogenous variables
            future_exog = {}
            for col, model in exog_models.items():
                forecast = model.forecast(steps=self.future_steps)
                future_exog[col] = forecast
            
            # Create a DataFrame with the forecasted exogenous variables
            future_exog_df = pd.DataFrame(future_exog)
            
            # Prepare future exogenous values
            future_exog = future_exog_df
            
            future_predicted = row['Model (SARIMAX)'].predict(start=len(df_reset), 
                                                            end=len(df_reset) + self.future_steps - 1, 
                                                            exog=future_exog)
        else:
            future_predicted = row['Model (SARIMAX)'].predict(start=len(df_reset), 
                                                            end=len(df_reset) + self.future_steps - 1)
        temp = pd.DataFrame({'Nama Item': row['Nama Item'], 'Jumlah Item': future_predicted, 'Model': 'SARIMAX'}, index=date_range)
        temp['Jumlah Item'] = temp['Jumlah Item'].apply(lambda x: max(0, x))
        return temp[['Nama Item', 'Jumlah Item', 'Model']]

    def prophet_forecast(self, row):
        model = row['Model (Prophet)']
        df = self.get_product_df(row['Nama Item']).reset_index()
        if not row['exog']:
            df['exog1'], df['exog2'] = 0, 0
        df.rename(columns={'index': 'ds', 'Jumlah Item': 'y'}, inplace=True)
        
        # Fit ARIMA models to exogenous variables
        exog_models = {}
        for col in ['exog1', 'exog2']:
            model_arima = ARIMA(df[col], order=(1,1,1))  # Use a simple ARIMA(1,1,1) model
            model_fit = model_arima.fit()
            exog_models[col] = model_fit
        
        # Forecast exogenous variables
        future_exog = {}
        for col, model in exog_models.items():
            forecast = model.forecast(steps=self.future_steps)
            future_exog[col] = forecast
        
        # Create a DataFrame with the forecasted exogenous variables
        future_exog_df = pd.DataFrame(future_exog)
        
        future = model.make_future_dataframe(periods=self.future_steps, freq='MS')
        future['exog1'], future['exog2'] = future_exog_df['exog1'], future_exog_df['exog2']
        
        forecast = model.predict(future)[['ds', 'yhat']]
        forecast['yhat'] = forecast['yhat'].apply(lambda x: max(0, x))
        forecast['Nama Item'] = row['Nama Item']
        forecast['Model'] = 'Prophet'
        forecast['Jumlah Item'] = forecast['yhat'].apply(lambda x: max(0, x))
        
        return forecast.tail(self.future_steps).set_index('ds')[['Nama Item', 'Jumlah Item', 'Model']]

    def lstm_forecast(self, row):
        model = row['Model (LSTM)']
        df = self.get_product_df(row['Nama Item'])
        if not row['exog']:
            df['exog1'], df['exog2'] = 0, 0
        
        # Scale the data
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(df[['Jumlah Item', 'exog1', 'exog2']])
        last_sequence = scaled_data[-12:]
        future_preds = []
        
        for step in range(self.future_steps):
            pred = model.predict(last_sequence[np.newaxis, :, 1:])
            future_preds.append(pred[0, 0])
            new_sequence = np.append(last_sequence[1:], [[pred[0, 0], last_sequence[-1, 1], last_sequence[-1, 2]]], axis=0)
            last_sequence = new_sequence
        
        y_pred_rescaled = scaler.inverse_transform(np.concatenate(
            (np.array(future_preds).reshape(-1, 1), last_sequence[-self.future_steps:, 1:]), axis=1))[:, 0]
        
        temp = pd.DataFrame({
            'Nama Item': row['Nama Item'],
            'Jumlah Item': y_pred_rescaled,
            'Model': 'LSTM'
        }, index=pd.date_range(start=df.index[-1] + pd.DateOffset(months=1), periods=self.future_steps, freq='MS'))
        
        return temp[['Nama Item', 'Jumlah Item', 'Model']]

    def naive_forecast(self, row):
        """Handle Naive model where predictions are not possible."""
        print(row['Nama Item'], 'Prediction Failed')
        return pd.DataFrame()

    def generate_forecasts(self):
        """Loop through combined results and generate predictions based on the best model."""
        for index, row in self.combined_results.iterrows():
            if row['Best Model'] == 'SARIMAX':
                temp = self.sarimax_forecast(row)
                temp['Average Model Error'] = row['ERROR (SARIMAX)']
            elif row['Best Model'] == 'Prophet':
                temp = self.prophet_forecast(row)
                temp['Average Model Error'] = row['ERROR (Prophet)']
            elif row['Best Model'] == 'LSTM':
                temp = self.lstm_forecast(row)
                temp['Average Model Error'] = row['ERROR (LSTM)']
            elif row['Best Model'] == 'Naive':
                temp = self.naive_forecast(row)
                gr.Warning(f"Unable to predict {row['Nama Item']}")
            else:
                temp = pd.DataFrame()
            
            self.predictions.append(temp)

        # Concatenate all predictions
        predictions_df = pd.concat(self.predictions)

        # Add today's date as 'Tanggal Prediksi'
        predictions_df['Tanggal Prediksi'] = pd.Timestamp.now()

        # Reset the index, rename it to 'Bulan'
        predictions_df = predictions_df.reset_index().rename(columns={'index': 'Bulan'})
        predictions_df = predictions_df[["Tanggal Prediksi", "Nama Item", "Bulan", "Jumlah Item", "Model", "Average Model Error"]]
        predictions_df['Jumlah Item'] = predictions_df['Jumlah Item'].apply(lambda x: max(x, 0))
        return predictions_df

class AssociationRuleMining:
    def __init__(self, df, min_support=0.01, min_threshold=1, score_threshold=0.6, weights=None):
        self.df = df
        self.min_support = min_support
        self.min_threshold = min_threshold
        self.score_threshold = score_threshold
        self.weights = weights or {'support': 0.2, 'confidence': 0.3, 'lift': 0.5}
        self.rules = None
        self.simplified_rules_with_score = None

    def preprocess_data(self):
        # Step 1: Ensure 'Tanggal Penjualan' is in datetime format and create 'YearMonth' column
        transactions = self.df[['Tanggal Penjualan', 'Nama Pelanggan', 'Nama Item']].copy()
        transactions['Tanggal Penjualan'] = pd.to_datetime(transactions['Tanggal Penjualan'], errors='coerce')
        transactions['YearMonth'] = transactions['Tanggal Penjualan'].dt.to_period('M')
        
        # Step 2: Group by customer and month, then list items
        monthly_transactions = transactions.groupby(['Nama Pelanggan', 'YearMonth'])['Nama Item'].apply(list).reset_index()
        return monthly_transactions

    def encode_transactions(self, monthly_transactions):
        # Step 3: One-hot encode the transaction data
        te = TransactionEncoder()
        te_ary = te.fit(monthly_transactions['Nama Item']).transform(monthly_transactions['Nama Item'])
        return pd.DataFrame(te_ary, columns=te.columns_)

    def apply_apriori(self, transaction_df):
        # Step 4: Apply Apriori algorithm
        frequent_itemsets = apriori(transaction_df, min_support=self.min_support, use_colnames=True)
        return frequent_itemsets

    def generate_rules(self, frequent_itemsets):
        # Step 5: Generate association rules
        rules = association_rules(frequent_itemsets, metric="lift", min_threshold=self.min_threshold)
        
        # Normalize metrics for scoring
        rules['Support_normalized'] = rules['support'] / rules['support'].max()
        rules['Confidence_normalized'] = rules['confidence'] / rules['confidence'].max()
        rules['Lift_normalized'] = rules['lift'] / rules['lift'].max()

        # Calculate the weighted score for each rule
        rules['Score'] = (
            self.weights['support'] * rules['Support_normalized'] +
            self.weights['confidence'] * rules['Confidence_normalized'] +
            self.weights['lift'] * rules['Lift_normalized']
        )

        # Sort by score and filter by the score threshold
        rules = rules.sort_values(by='Score', ascending=False)
        self.rules = rules[rules['Score'] >= self.score_threshold]
        return self.rules

    def get_simplified_rules(self):
        # Select a simplified version of the sorted DataFrame with scores
        self.simplified_rules_with_score = self.rules[['antecedents', 'consequents', 'support', 'confidence', 'lift', 'Score']].round(3)
        return self.simplified_rules_with_score.reset_index(drop=True)

    def run_analysis(self):
        monthly_transactions = self.preprocess_data()
        transaction_df = self.encode_transactions(monthly_transactions)
        frequent_itemsets = self.apply_apriori(transaction_df)
        self.generate_rules(frequent_itemsets)
        return self.get_simplified_rules()

def upload_sales_data(data_file, progress = gr.Progress()):
    if not data_file:
        raise gr.Error("No File Inputted")

    # Read and process file
    df = pd.read_excel(data_file)
    required_columns = ['Tanggal Penjualan', 'Nama Pelanggan', 'Nama Item', 'Jumlah Item', 'Harga Satuan', 'Diskon']

    # Check if all required columns are in the DataFrame
    if not all(column in df.columns for column in required_columns):
        missing_columns = [column for column in required_columns if column not in df.columns]
        raise gr.Error(f"The DataFrame is missing the following columns: {missing_columns}")

    df.rename(columns={'Harga Satuan': 'exog1'}, inplace=True)
    df.rename(columns={'Diskon': 'exog2'}, inplace=True)
    progress(0.2, desc='preparing data')  # Update progress
    # Perform data cleaning and conversions with progress updates
    if df['Jumlah Item'].dtype != 'float':
        df['Jumlah Item'] = df['Jumlah Item'].str.replace('(', '-').str.replace(')', '').str.replace(',', '')
        df['Jumlah Item'] = df['Jumlah Item'].astype(float)
    

    if df['Grand Total'].dtype != 'float':
        df['Grand Total'] = df['Grand Total'].str.replace('(', '-').str.replace(')', '').str.replace('Rp', '').str.replace(',', '').astype(float)
    if df['exog1'].dtype != 'float':
        df['exog1'] = df['exog1'].str.replace(",", "").astype(float)
    if df['exog2'].dtype != 'float':
        df['exog2'] = df['exog2'].str.replace(",", "").astype(float)
    progress(0.3, desc='preparing data')  # Update progress

    # Convert date format
    month_mapping = {
        ' Jan ': '/01/', ' Feb ': '/02/', ' Mar ': '/03/', ' Apr ': '/04/',
        ' May ': '/05/', ' Jun ': '/06/', ' Jul ': '/07/', ' Aug ': '/08/',
        ' Sep ': '/09/', ' Oct ': '/10/', ' Nov ': '/11/', ' Dec ': '/12/'
    }
    for month, replacement in month_mapping.items():
        df['Tanggal Penjualan'] = df['Tanggal Penjualan'].str.replace(month, replacement)
    df['Tanggal Penjualan'] = pd.to_datetime(df['Tanggal Penjualan'], format='%d/%m/%Y')


    # Store the data in MongoDB
    records = df.to_dict(orient='records')
    upsert_data(records, progress)
    # sales_collection.insert_many(records)
    
    # Retrieve and return the updated sales history
    history, clusters = get_sales_history()
    opt = ['All'] + list(clusters.keys())
    progress(1)
    return history, gr.Dropdown(choices=opt, value=opt[0]), None, gr.update(visible=True), gr.update(visible=True)

def upsert_data(data, progress):
    # Initialize a set to keep track of existing keys to minimize MongoDB reads
    existing_keys = set()
    duplicate_count = 0
    total_records = len(data)
    
    # Step 1: Retrieve existing keys in a single batch operation
    existing_docs = sales_collection.find({}, {"Nomor Penjualan": 1, "Kode Item": 1})
    for doc in existing_docs:
        key = (doc["Nomor Penjualan"], doc["Kode Item"])
        existing_keys.add(key)
    
    # Step 2: Create batch updates for upserts
    batch_updates = []
    for record in data:
        nomor_penjualan = record.get("Nomor Penjualan")
        kode_item = record.get("Kode Item")
        key = (nomor_penjualan, kode_item)
        
        if key in existing_keys:
            duplicate_count += 1
        else:
            existing_keys.add(key)  # Add to avoid future duplicates
        
        # Prepare the update operation
        batch_updates.append(
            UpdateOne(
                {"Nomor Penjualan": nomor_penjualan, "Kode Item": kode_item},
                {"$set": record},
                upsert=True
            )
        )
    
    # Step 3: Execute updates in parallel
    def execute_batch(start, end):
        sales_collection.bulk_write(batch_updates[start:end])
    
    # Divide into smaller batches for parallel execution
    batch_size = 100  # Adjust batch size based on your dataset and MongoDB configuration
    with ThreadPoolExecutor() as executor:
        futures = []
        for i in range(0, len(batch_updates), batch_size):
            start = i
            end = min(i + batch_size, len(batch_updates))
            futures.append(executor.submit(execute_batch, start, end))
        
        # Update progress bar as batches complete
        for i, future in enumerate(as_completed(futures)):
            future.result()  # Wait for batch to complete
            progress(0.3 + (0.5 * (i + 1) / len(futures)), desc="storing data..")
    
    # Provide feedback on duplicates
    gr.Warning(f"Total duplicates found and updated: {duplicate_count}")
# Function to retrieve sales history from MongoDB
def get_sales_history():
    # Fetch all records from the collection
    global sales_history
    global clusters
    sales_history = pd.DataFrame(list(sales_collection.find()))
    if not sales_history.empty:
        sales_history = sales_history.drop(columns=['_id'])  # Drop MongoDB's default ID column for cleaner display
        clusters = filtering(sales_history)
        return sales_history.iloc[::-1].reset_index(drop=True), clusters
    else:
        return pd.DataFrame(), {}

def get_MBA_history():
    MBA_history = pd.DataFrame(list(MBA_collection.find()))
    if not MBA_history.empty:
        MBA_history = MBA_history.drop(columns=['_id'])
        MBA_history = MBA_history.iloc[::-1].reset_index(drop=True)
        MBA_history.to_excel('MBA_History.xlsx', index=False)
        return MBA_history
    else:
        pd.DataFrame().to_excel('MBA_History.xlsx', index=False)
        return pd.DataFrame()
    
def get_prediction_history():
    pred_history = pd.DataFrame(list(prediction_collection.find()))
    if not pred_history.empty:
        pred_history = pred_history.drop(columns=['_id'])
        pred_history = pred_history.iloc[::-1].reset_index(drop=True)
        pred_history.to_excel('Prediction_History.xlsx', index=False)
        return pred_history
    else:
        pd.DataFrame().to_excel('Prediction_History.xlsx', index=False)
        return pd.DataFrame()
    
def get_model_history():
    model_history = pd.DataFrame(list(model_collection.find()))
    if not model_history.empty:
        model_history = model_history.drop(columns=['_id']).iloc[::-1].reset_index(drop=True)
        model_history.to_excel('Model_History.xlsx', index=False)
        return model_history
    else:
        return pd.DataFrame()
    
def filtering(sales_history):
    # Filter data for 2023 and 2024
    df_2023 = sales_history[sales_history['Tanggal Penjualan'].dt.year == 2023]
    df_2024 = sales_history[sales_history['Tanggal Penjualan'].dt.year == 2024]

    # Find common ProductIDs
    common_products = pd.merge(df_2023, df_2024, on='Nama Item', suffixes=('_2023', '_2024'))

    #get unique products in common_products
    unique_products = common_products['Nama Item'].unique()
    sales_history = sales_history[sales_history['Nama Item'].isin(unique_products)]
    combined_dict = Clustering(unique_products).run()
    return combined_dict

def get_best_model_error(row):
    if row['Best Model'] == 'Naive':
        return row['ERROR (Naive)']
    elif row['Best Model'] == 'SARIMAX':
        return row['ERROR (SARIMAX)']
    elif row['Best Model'] == 'Prophet':
        return row['ERROR (Prophet)']
    elif row['Best Model'] == 'LSTM':
        return row['ERROR (LSTM)']
    else:
        return None  # If no match, return None
     
def forecasting(choice, exog_file, batch_size=5, progress=gr.Progress()):

    # Initial data preparation
    products_df = sales_history[['Tanggal Penjualan', 'Nama Item', 'Jumlah Item', 'exog1', 'exog2']]
    # Filter data for 2023 and 2024
    df_2023 = sales_history[sales_history['Tanggal Penjualan'].dt.year == 2023]
    df_2024 = sales_history[sales_history['Tanggal Penjualan'].dt.year == 2024]
    # Find common ProductIDs
    common_products = pd.merge(df_2023, df_2024, on='Nama Item', suffixes=('_2023', '_2024'))
    #get unique products in common_products
    unique_products = common_products['Nama Item'].unique()
    products_df = products_df[products_df['Nama Item'].isin(unique_products)]
    if choice != 'All':
        products_df = products_df[products_df['Nama Item'].isin(clusters[choice])]
    products_df['YearMonth'] = products_df['Tanggal Penjualan'].dt.to_period('M')
    products_df['YearMonth'] = products_df['YearMonth'].dt.to_timestamp()
    products_df.set_index('YearMonth', inplace=True)
    products_df = products_df.fillna(0)
    progress(0.1, desc='preparing data')  # Update progress

    products_df = products_df.groupby(['YearMonth', 'Nama Item']).apply(
        lambda group: pd.Series({
            'Jumlah Item': group['Jumlah Item'].sum(),
            'exog1': (group['exog1'] * group['Jumlah Item']).sum() / group['Jumlah Item'].sum() if group['Jumlah Item'].sum() != 0 else 0,
            'exog2': (group['exog2'] * group['Jumlah Item']).sum() / group['Jumlah Item'].sum() if group['Jumlah Item'].sum() != 0 else 0
        })
    ).reset_index()
    progress(0.2, desc='preparing data')  # Update progress

    product_dfs = []
    for product in products_df['Nama Item'].unique():
        temp = products_df[products_df['Nama Item'] == product]
        temp.set_index('YearMonth', inplace=True)
        all_months = pd.date_range(start=temp.index.min(), end=products_df['YearMonth'].max(), freq='MS')
        temp = temp.reindex(all_months)
        temp['Nama Item'] = temp['Nama Item'].fillna(product)
        temp['exog1'] = temp['exog1'].ffill()
        temp['exog2'] = temp['exog2'].ffill()
        temp['Jumlah Item'] = temp['Jumlah Item'].fillna(0)
        product_dfs.append(temp)
    product_dfs = [df for df in product_dfs if len(df) >= 24]
    progress(0.3, desc='preparing exogenous variable')  # Update progress

    if exog_file:
        exog_df = pd.read_excel(exog_file)
        exog_df.set_index(exog_df.columns[0], inplace=True)
        exog_df.index = pd.to_datetime(exog_df.index)
        if len(exog_df.columns) > 0:
            exog_df.rename(columns={exog_df.columns[0]: 'exog1'}, inplace=True)
            if len(exog_df.columns) > 1:
                exog_df.rename(columns={exog_df.columns[1]: 'exog2'}, inplace=True)

            # Ensure the columns exist and contain numeric data
            if 'exog1' not in exog_df.columns:
                exog_df['exog1'] = 0  # Add the column with default value 0
            else:
                exog_df['exog1'] = pd.to_numeric(exog_df['exog1'], errors='coerce').fillna(0)

            if 'exog2' not in exog_df.columns:
                exog_df['exog2'] = 0  # Add the column with default value 0
            else:
                exog_df['exog2'] = pd.to_numeric(exog_df['exog2'], errors='coerce').fillna(0)


            merged_dfs = [
                df.drop(columns=['exog1', 'exog2'], errors='ignore')
                .merge(exog_df[['exog1', 'exog2']], left_index=True, right_index=True, how='left')
                .fillna({'exog1': 0, 'exog2': 0})
                for df in product_dfs
            ]
            product_dfs = merged_dfs
            # Placeholder for correlation sums and count
            correlation_sum_exog1 = 0
            correlation_sum_exog2 = 0
            count = 0

            # Loop through each DataFrame in product_dfs
            for df in product_dfs:
                # Check if the required columns exist in the current DataFrame
                if {'Jumlah Item', 'exog1', 'exog2'}.issubset(df.columns):
                    correlation1 = abs(df['Jumlah Item'].corr(df['exog1']))
                    correlation2 = abs(df['Jumlah Item'].corr(df['exog2']))
                    
                    # Accumulate the correlations
                    correlation_sum_exog1 += correlation1 if pd.notna(correlation1) else 0
                    correlation_sum_exog2 += correlation2 if pd.notna(correlation2) else 0
                    count += 1

            # Calculate average |correlation| for each exogenous variable
            average_correlation_exog1 = correlation_sum_exog1 / count if count > 0 else 0
            if average_correlation_exog1 < 0.4 and average_correlation_exog1 != 0:
                gr.Warning("Little or no correlation found between 'Jumlah Item' and 'exog1' the prediction may not be accurate")
            else:
                gr.Info(f"Average correlation between 'Jumlah Item' and 'exog1': {average_correlation_exog1:.2f}")

            average_correlation_exog2 = correlation_sum_exog2 / count if count > 0 else 0
            if average_correlation_exog2 < 0.4 and average_correlation_exog2 != 0:
                gr.Warning("Little or no correlation found between 'Jumlah Item' and 'exog2' the prediction may not be accurate")
            else:
                gr.Info(f"Average correlation between 'Jumlah Item' and 'exog2': {average_correlation_exog2:.2f}")
        else:
            gr.Warning("No exogenous variable data found in the uploaded file.")
    


    # Modify exogenous variables
    product_dfs_no_exog = [df.copy().assign(exog1=0, exog2=0) for df in product_dfs]

    progress(0.4, desc='testing SARIMAX with exog')  # Update progress
    SARIMAX_results = SARIMAX_pred(product_dfs).run()
    progress(0.45, desc='testing SARIMAX Forecast without exog')  # Update progress
    SARIMAX_results2 = SARIMAX_pred(product_dfs_no_exog).run()
    progress(0.5, desc='testing Prophet with exog')  # Update progress
    Prophet_results = Prophet_pred(product_dfs).run()
    progress(0.55, desc='testing Prophet without exog')  # Update progress
    Prophet_results2 = Prophet_pred(product_dfs_no_exog).run()
    progress(0.6, desc='testing LSTM with exog')  # Update progress
    LSTM_results = LSTM_pred(product_dfs).run()
    progress(0.65, desc='testing LSTM without exog')  # Update progress
    LSTM_results2 = LSTM_pred(product_dfs_no_exog).run()
    progress(0.7, desc='testing Naive Forecast with exog')  # Update progress
    benchmark_results = NaiveForecast(product_dfs).run()
    progress(0.75, desc='testing Naive Forecast without exog')  # Update progress
    benchmark_results2 = NaiveForecast(product_dfs_no_exog).run()

    # Combine results for with exogenous variables
    combined_results = benchmark_results.merge(SARIMAX_results, on='Nama Item', how='outer')
    combined_results = combined_results.merge(Prophet_results, on='Nama Item', how='outer')
    combined_results = combined_results.merge(LSTM_results, on='Nama Item', how='outer')
    combined_results['Best Model'] = combined_results.apply(find_best_model, axis=1)
    combined_results['Best Model Error'] = combined_results.apply(get_best_model_error, axis=1)
    combined_results['exog'] = True

    # Combine results for no exogenous variables
    combined_results2 = benchmark_results2.merge(SARIMAX_results2, on='Nama Item', how='outer')
    combined_results2 = combined_results2.merge(Prophet_results2, on='Nama Item', how='outer')
    combined_results2 = combined_results2.merge(LSTM_results2, on='Nama Item', how='outer')
    combined_results2['Best Model'] = combined_results2.apply(find_best_model, axis=1)
    combined_results2['Best Model Error'] = combined_results2.apply(get_best_model_error, axis=1)
    combined_results2['exog'] = False

    # Select the best model between the two scenarios
    best = []
    for i in range(len(combined_results)):
        if combined_results.loc[i, 'Best Model Error'] < combined_results2.loc[i, 'Best Model Error']:
            best.append(combined_results.loc[i])
        else:
            best.append(combined_results2.loc[i])

    # Combine all batch results
    best = pd.DataFrame(best)
    progress(0.8, desc='Generating forecasts')  # Update progress

    # Timestamp and formatting
    predictions_df = ForecastGenerator(best, product_dfs).generate_forecasts()
    best['Tanggal Prediksi'] = predictions_df['Tanggal Prediksi'].iloc[0]
    cols = best.columns.tolist()
    cols.insert(0, cols.pop(cols.index('Tanggal Prediksi')))
    best = best[cols]

    progress(0.9, desc='Storing results')  # Update progress

    # Store batch results in MongoDB
    prediction_records = predictions_df.to_dict(orient='records')
    prediction_collection.insert_many(prediction_records)
    best.drop(columns=['Model (SARIMAX)', 'Model (Prophet)', 'Model (LSTM)'], inplace=True)
    model_records = best.to_dict(orient='records')
    model_collection.insert_many(model_records)

    # Clear memory after processing
    del SARIMAX_results, Prophet_results, LSTM_results, benchmark_results
    del combined_results, combined_results2, best, predictions_df
    gc.collect()  # Force garbage collection to free up memory

    progress(1.0, desc='Processing complete')  # Mark progress as complete

    # except Exception as e:
    #     print(f"An error occurred while processing: {e}")

    progress(1.0)  # Complete
    return get_prediction_history(), None, get_model_history()
# Function to find the model with the least error for each row
def find_best_model(row):
    errors = {
        'Naive': row['ERROR (Naive)'],
        'SARIMAX': row['ERROR (SARIMAX)'],
        'Prophet': row['ERROR (Prophet)'],
        'LSTM': row['ERROR (LSTM)']
    }
    best_model = min(errors, key=errors.get)
    return best_model
# Function to filter predictions based on selected date and item
def filter_predictions(date, item, data, model):
    df = data
    if df.empty:
        return df
    if date != 'All':
        df = df[df['Tanggal Prediksi'].astype(str) == date]
        model = model[model['Tanggal Prediksi'].astype(str)== date]
    if item != 'All':
        df = df[df['Nama Item'] == item]
        model = model[model['Nama Item'] == item]
    plot = df.sort_values('Tanggal Prediksi').drop_duplicates(subset=['Bulan', 'Nama Item'], keep='last')
    df.to_excel('Prediction_History.xlsx', index=False)
    model.to_excel('Model_History.xlsx', index=False)
    return df, plot, model.style.apply(highlight_rows, axis=1), gr.DownloadButton(label="Download Prediction History", value="Prediction_History.xlsx"), gr.DownloadButton(label="Download Model History", value="Model_History.xlsx")

def highlight_rows(row):
    if row['Best Model'] == 'Naive':
        return ['background-color: red'] * len(row)
    else:
        return [''] * len(row)
# Function to update dropdown choices based on the current DataFrame
def update_prediction_components(data):
    if data.empty:
        date_choices = ['All']
        item_choices = ['All']
        plot = pd.DataFrame()
    else:
        date_choices = ['All'] + data['Tanggal Prediksi'].astype(str).unique().tolist()
        item_choices = ['All'] + data['Nama Item'].unique().tolist()
        plot = data.sort_values('Tanggal Prediksi').drop_duplicates(subset=['Bulan', 'Nama Item'], keep='last')
    return gr.Dropdown(choices=date_choices), gr.Dropdown(choices=item_choices), gr.DataFrame(value=data), gr.LinePlot(value = plot), gr.DownloadButton(label="Download Prediction History", value="Prediction_History.xlsx"), gr.DownloadButton(label="Download Model History", value="Model_History.xlsx")

def update_model_components(data):
    data = data.style.apply(highlight_rows, axis=1)
    return gr.Dataframe(value=data)

def MBA_analysis(data):
    association_mining = AssociationRuleMining(data)
    results = association_mining.run_analysis()
    results['Tanggal Analisis'] = pd.Timestamp.now()

    # Reorder columns to place 'Tanggal Analisis' in the second column
    cols = results.columns.tolist()
    cols.insert(0, cols.pop(cols.index('Tanggal Analisis')))
    results = results[cols]

    records = results.to_dict(orient='records')
    records = [convert_frozensets(record) for record in records]
    MBA_collection.insert_many(records)
    return get_MBA_history()
# Function to convert frozenset to list
def convert_frozensets(record):
    for key, value in record.items():
        if isinstance(value, frozenset):
            record[key] = list(value)  # Convert frozenset to list
    return record

def shutdown():
    demo.close()


with gr.Blocks(title='Demand Forecasting CV Kita Jaya') as demo:
    # Load sales history and prediction history
    history, opt = get_sales_history()
    pred_history = get_prediction_history()
    pred_history_state = gr.State(pred_history)
    model_history = get_model_history()
    model_history_state = gr.State(model_history)
    model_history = model_history.style.apply(highlight_rows, axis=1)
    MBA_history = get_MBA_history()
    opt = ['All'] + list(clusters.keys()) if opt else []
    
    shutdown_button = gr.Button("Shutdown Server", variant="primary")
    # Sales Tab
    with gr.Tab("Data Penjualan") as sales_tab:
        gr.Markdown("# Data Penjualan")
        # File upload section
        with gr.Row():
            progress_bar = gr.Markdown(value="")  
        with gr.Row():
            file_input = gr.UploadButton(label="Upload", file_types=[".xls", ".xlsx"])
        with gr.Row():
            gr.Markdown("Upload file data penjualan dalam format Excel (.xls).")
        # Display sales history table
        with gr.Row():
            sales_history_output = gr.Dataframe(value=history if not history.empty else None, label="Histori Penjualan", interactive=False)
    
    # Forecast Tab
    with gr.Tab("Forecast", visible=bool(opt)) as forecast_tab:
        gr.Markdown("# Forecast")
        with gr.Row():
            # Dropdown for categories if available
            categories = gr.Dropdown(choices=opt, value=opt[0] if opt else None, label="Categories")
            # File upload for exogenous data
            exog_file = gr.File(label="Data Eksogen", file_types=[".xls", ".xlsx"], interactive=True)
        # Forecast button and prediction output
        forecast = gr.Button("Forecast")
        # Dropdowns for filtering
        with gr.Row():
            date_choices = (['All'] + pred_history['Tanggal Prediksi'].astype(str).unique().tolist()) if not pred_history.empty else None
            item_choices = (['All'] + pred_history['Nama Item'].unique().tolist()) if not pred_history.empty else None
            
            date_dropdown = gr.Dropdown(choices=date_choices, value='All', label="Tanggal Prediksi")
            item_dropdown = gr.Dropdown(choices=item_choices, value='All', label="Nama Item")
        
        # Dataframe to display filtered predictions
        with gr.Tab("Table"):
            download_pred = gr.DownloadButton(label="Download Prediction History", value="Prediction_History.xlsx")
            prediction_output = gr.Dataframe(value=pred_history, interactive=False)
        with gr.Tab("Plot"):
            prediction_plot = gr.LinePlot(value = pred_history.sort_values('Tanggal Prediksi').drop_duplicates(subset=['Bulan', 'Nama Item'], keep='last') if not pred_history.empty else pd.DataFrame(), x = "Bulan", y = "Jumlah Item", color = "Nama Item")
        with gr.Tab("Model Report"):
            download_report = gr.DownloadButton(label="Download Model History", value="Model_History.xlsx")
            model_report = gr.DataFrame(value=model_history, interactive=False)
        

    with gr.Tab("MBA Analysis", visible=bool(opt)) as MBA_tab:
        gr.Markdown("# MBA")
        with gr.Row():
            MBA_button = gr.Button("Generate Analysis")
            download_MBA = gr.DownloadButton(label="Download MBA History", value="MBA_History.xlsx")
        MBA_result = gr.DataFrame(value = MBA_history, label="Hasil MBA", interactive=False)
    
    MBA_button.click(
        fn=MBA_analysis,
        inputs=sales_history_output, 
        outputs=MBA_result
    )
    # Forecast button action
    forecast.click(
        fn=forecasting, 
        inputs=[categories, exog_file], 
        outputs=[pred_history_state, exog_file, model_history_state]
    )
    
    # Action for file upload
    file_input.upload(
        fn=upload_sales_data, 
        inputs=file_input, 
        outputs=[sales_history_output, categories, file_input, forecast_tab, MBA_tab]
    )

    shutdown_button.click(fn=shutdown)

    # Update prediction_output when dropdown values change
    date_dropdown.change(
        fn=filter_predictions,
        inputs=[date_dropdown, item_dropdown, pred_history_state, model_history_state],
        outputs=[prediction_output,prediction_plot, model_report, download_pred, download_report]
    )
    item_dropdown.change(
        fn=filter_predictions,
        inputs=[date_dropdown, item_dropdown, pred_history_state, model_history_state],
        outputs=[prediction_output, prediction_plot, model_report, download_pred, download_report]
    )
    # Update dropdown choices when prediction_output changes
    pred_history_state.change(
        fn=update_prediction_components,
        inputs=pred_history_state,
        outputs=[date_dropdown, item_dropdown, prediction_output, prediction_plot, download_pred, download_report]
    )
    model_history_state.change(
        fn=update_model_components,
        inputs=model_history_state,
        outputs=model_report
    )

demo.launch()