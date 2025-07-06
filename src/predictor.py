import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import joblib
import os

# Updated import path: from src.module_name import ClassName
from src.main import WalmartSalesForecaster

class WalmartPredictor:
    def __init__(self):
        """Initialize the predictor with trained models"""
        # Ensure data_path is correctly set for the forecaster, pointing to the data/ directory
        self.forecaster = WalmartSalesForecaster(data_path='data/Walmart.csv')
        self.models_trained = False
        self.model_name = 'RandomForest' # Default model to use for prediction

    def setup_models(self):
        """Setup and train models if not already done"""
        print("🚀 Setting up Walmart Sales Predictor...")

        # Load and prepare data using the forecaster's methods
        if not self.forecaster.load_data():
            print("Failed to load data for forecaster.")
            return False

        self.forecaster.preprocess_data()
        if self.forecaster.processed_data is None:
            print("Failed to preprocess data for forecaster. Exiting model setup.")
            return False

        self.forecaster.train_models()
        if not self.forecaster.models:
            print("No models were trained. Exiting model setup.")
            return False

        self.models_trained = True
        print("✅ Models ready for predictions!")
        return True

    def predict_single_store(self, store_id, date_str, holiday_flag=0,
                           temperature=60.0, fuel_price=3.5, cpi=180.0, unemployment=8.0,
                           dept_id=1, type_id='A'):
        """
        Predict sales for a single store on a specific date
        Parameters:
        - store_id: Store number (1-45)
        - date_str: Date in YYYY-MM-DD format
        - holiday_flag: 0 or 1 (False/True)
        - temperature: Average temperature for the week
        - fuel_price: Average fuel price for the week
        - cpi: Consumer Price Index for the week
        - unemployment: Unemployment rate for the week
        - dept_id: Department ID (default to 1)
        - type_id: Store Type ('A', 'B', 'C') (default to 'A')
        """
        if not self.models_trained:
            print("Models are not set up. Please run setup_models first.")
            return None

        # Use the forecaster's method to prepare the input data
        input_data = self.forecaster.prepare_single_prediction_data(
            store_id=store_id,
            date_str=date_str,
            holiday_flag=holiday_flag,
            temperature=temperature,
            fuel_price=fuel_price,
            cpi=cpi,
            unemployment=unemployment,
            dept_id=dept_id,
            type_id=type_id
        )

        if input_data is None:
            print("Failed to prepare prediction data.")
            return None

        model = self.forecaster.get_model(self.model_name)
        if model:
            predicted_sales = model.predict(input_data)[0]
            return predicted_sales
        else:
            print(f"Error: Model '{self.model_name}' not found.")
            return None

    def predict_multiple_stores(self, store_ids, date_str, holiday_flag=0,
                                temperature=60.0, fuel_price=3.5, cpi=180.0, unemployment=8.0,
                                dept_id=1, type_id='A'):
        """
        Predict sales for multiple stores on a specific date.
        """
        if not self.models_trained:
            print("Models are not set up. Please run setup_models first.")
            return None

        results = {}
        for store_id in store_ids:
            predicted_sales = self.predict_single_store(
                store_id, date_str, holiday_flag, temperature, fuel_price,
                cpi, unemployment, dept_id, type_id
            )
            if predicted_sales is not None:
                results[f'Store {store_id}'] = predicted_sales
            else:
                print(f"Skipping prediction for Store {store_id} due to data preparation issues.")
        return results

    def predict_time_series_for_store(self, store_id, start_date_str, num_weeks,
                                      temperature=60.0, fuel_price=3.5, cpi=180.0, unemployment=8.0,
                                      dept_id=1, type_id='A'):
        """
        Predict sales for a single store over a specified number of weeks.
        """
        if not self.models_trained:
            print("Models are not set up. Please run setup_models first.")
            return None, None

        predictions = []
        dates = []
        try:
            current_date = datetime.strptime(start_date_str, '%Y-%m-%d')
        except ValueError:
            print(f"Error: Invalid start date format '{start_date_str}'. Please use YYYY-MM-DD.")
            return None, None

        model = self.forecaster.get_model(self.model_name)

        if not model:
            print(f"Error: Model '{self.model_name}' not found.")
            return None, None

        for i in range(num_weeks):
            date_to_predict = current_date + timedelta(weeks=i)
            date_str_to_predict = date_to_predict.strftime('%Y-%m-%d')

            # The prepare_single_prediction_data will handle holiday derivation
            holiday_flag_for_date = 0 # Let the function determine based on date

            input_data = self.forecaster.prepare_single_prediction_data(
                store_id=store_id,
                date_str=date_str_to_predict,
                holiday_flag=holiday_flag_for_date,
                temperature=temperature,
                fuel_price=fuel_price,
                cpi=cpi,
                unemployment=unemployment,
                dept_id=dept_id,
                type_id=type_id
            )

            if input_data is None:
                print(f"Failed to prepare prediction data for date {date_str_to_predict}. Skipping.")
                continue

            predicted_sales = model.predict(input_data)[0]
            predictions.append(predicted_sales)
            dates.append(date_to_predict)

        return dates, predictions

    def _single_store_interactive(self):
        """Interactive prediction for a single store and date."""
        print("\n--- Predict for a Single Store and Date ---")
        try:
            store_id = int(input("Enter Store ID (1-45): "))
            date_str = input("Enter Date (YYYY-MM-DD): ")
            holiday_flag = int(input("Is it a holiday? (0 for No, 1 for Yes, or leave blank to auto-detect): ") or "0")
            temperature = float(input("Enter Temperature (e.g., 60.0, default 60.0): ") or "60.0")
            fuel_price = float(input("Enter Fuel Price (e.g., 3.5, default 3.5): ") or "3.5")
            cpi = float(input("Enter CPI (e.g., 180.0, default 180.0): ") or "180.0")
            unemployment = float(input("Enter Unemployment Rate (e.g., 8.0, default 8.0): ") or "8.0")
            dept_id = int(input("Enter Department ID (e.g., 1, default 1): ") or "1")
            type_id = input("Enter Store Type (A, B, or C, default A): ") or "A"

            predicted_sales = self.predict_single_store(
                store_id, date_str, holiday_flag, temperature, fuel_price, cpi, unemployment, dept_id, type_id
            )
            if predicted_sales is not None:
                print(f"\nPredicted Weekly Sales for Store {store_id} on {date_str}: ${predicted_sales:,.2f}")
            else:
                print("Prediction failed.")
        except ValueError:
            print("Invalid input. Please enter numerical values where expected.")
        except Exception as e:
            print(f"An error occurred: {e}")

    def _multiple_stores_interactive(self):
        """Interactive prediction for multiple stores on a date."""
        print("\n--- Predict for Multiple Stores on a Date ---")
        try:
            store_ids_input = input("Enter Store IDs (comma-separated, e.g., 1,5,10): ")
            store_ids = [int(s.strip()) for s in store_ids_input.split(',')]
            date_str = input("Enter Date (YYYY-MM-DD): ")
            holiday_flag = int(input("Is it a holiday? (0 for No, 1 for Yes, or leave blank to auto-detect): ") or "0")
            temperature = float(input("Enter Temperature (e.g., 60.0, default 60.0): ") or "60.0")
            fuel_price = float(input("Enter Fuel Price (e.g., 3.5, default 3.5): ") or "3.5")
            cpi = float(input("Enter CPI (e.g., 180.0, default 180.0): ") or "180.0")
            unemployment = float(input("Enter Unemployment Rate (e.g., 8.0, default 8.0): ") or "8.0")
            dept_id = int(input("Enter Department ID (e.g., 1, default 1): ") or "1")
            type_id = input("Enter Store Type (A, B, or C, default A): ") or "A"

            results = self.predict_multiple_stores(
                store_ids, date_str, holiday_flag, temperature, fuel_price, cpi, unemployment, dept_id, type_id
            )
            if results:
                print(f"\nPredicted Weekly Sales for {date_str}:")
                for store, sales in results.items():
                    print(f"  {store}: ${sales:,.2f}")
            else:
                print("Prediction failed for multiple stores.")
        except ValueError:
            print("Invalid input. Please ensure store IDs are numbers and comma-separated.")
        except Exception as e:
            print(f"An error occurred: {e}")


    def _time_series_interactive(self):
        """Interactive time series prediction for a store."""
        print("\n--- Predict Time Series for a Store ---")
        try:
            store_id = int(input("Enter Store ID (1-45): "))
            start_date_str = input("Enter Start Date (YYYY-MM-DD): ")
            num_weeks = int(input("Enter number of weeks to predict: "))
            temperature = float(input("Enter Temperature (e.g., 60.0, default 60.0): ") or "60.0")
            fuel_price = float(input("Enter Fuel Price (e.g., 3.5, default 3.5): ") or "3.5")
            cpi = float(input("Enter CPI (e.g., 180.0, default 180.0): ") or "180.0")
            unemployment = float(input("Enter Unemployment Rate (e.g., 8.0, default 8.0): ") or "8.0")
            dept_id = int(input("Enter Department ID (e.g., 1, default 1): ") or "1")
            type_id = input("Enter Store Type (A, B, or C, default A): ") or "A"

            dates, predictions = self.predict_time_series_for_store(
                store_id, start_date_str, num_weeks, temperature, fuel_price, cpi, unemployment, dept_id, type_id
            )

            if dates and predictions:
                print(f"\nPredicted Weekly Sales for Store {store_id} starting {start_date_str}:")
                for date, sales in zip(dates, predictions):
                    print(f"  {date.strftime('%Y-%m-%d')}: ${sales:,.2f}")

                # Plotting the time series prediction
                plt.figure(figsize=(12, 6))
                plt.plot(dates, predictions, marker='o', linestyle='-')
                plt.title(f'Predicted Weekly Sales for Store {store_id}')
                plt.xlabel('Date')
                plt.ylabel('Predicted Weekly Sales')
                plt.grid(True)
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.show()
            else:
                print("Time series prediction failed.")
        except ValueError:
            print("Invalid input. Please enter valid numbers/dates.")
        except Exception as e:
            print(f"An error occurred: {e}")

    def _quick_prediction(self):
        """Quick prediction for Store 1 and Today's date."""
        print("\n--- Quick Prediction (Store 1, Today) ---")
        today_str = datetime.now().strftime('%Y-%m-%d')
        # Assuming typical values for other parameters for a quick prediction
        # These could be averaged from the dataset or set as sensible defaults
        predicted_sales = self.predict_single_store(
            store_id=1,
            date_str=today_str,
            holiday_flag=0, # Let the function determine based on date
            temperature=60.0,
            fuel_price=3.0,
            cpi=210.0,
            unemployment=7.0,
            dept_id=1,
            type_id='A'
        )
        if predicted_sales is not None:
            print(f"\nQuick Prediction for Store 1 on {today_str}: ${predicted_sales:,.2f}")
        else:
            print("Quick prediction failed.")

    def run_predictor_interface(self):
        """Main function to run the prediction interface"""
        print("🏪 WALMART SALES FORECASTING SYSTEM")
        print("=" * 50)
        print("This system uses trained Random Forest model with 95.88% accuracy!")

        # Setup models
        if not self.models_trained:
            if not self.setup_models():
                print("Exiting predictor due to model setup failure.")
                return

        while True:
            print("\nChoose a prediction option:")
            print("1. Predict for a Single Store and Date")
            print("2. Predict for Multiple Stores on a Date")
            print("3. Predict Time Series for a Store")
            print("4. Quick Prediction (Store 1, Today)")
            print("5. Exit")

            choice = input("Enter your choice (1-5): ")

            if choice == '1':
                self._single_store_interactive()
            elif choice == '2':
                self._multiple_stores_interactive()
            elif choice == '3':
                self._time_series_interactive()
            elif choice == '4':
                self._quick_prediction()
            elif choice == '5':
                print("Exiting Walmart Sales Predictor. Goodbye!")
                break
            else:
                print("Invalid choice. Please enter a number between 1 and 5.")

def main():
    predictor = WalmartPredictor()
    predictor.run_predictor_interface()

if __name__ == "__main__":
    main()