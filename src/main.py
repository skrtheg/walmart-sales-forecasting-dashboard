import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings
warnings.filterwarnings('ignore')
import io
from datetime import datetime, date

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class WalmartSalesForecaster:
    def __init__(self, data_path='data/Walmart.csv'):
        """Initialize the forecaster with data path"""
        self.data_path = data_path
        self.data = None # Raw data
        self.processed_data = None # Processed data for modeling
        self.models = {}
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

        # Define common US holidays for 'IsHoliday' derivation if the column is missing
        self.common_holidays = {
            '2010-02-12', '2010-09-10', '2010-11-26', '2010-12-24',
            '2011-02-11', '2011-09-09', '2011-11-25', '2011-12-23',
            '2012-02-10', '2012-09-07', '2012-11-23', '2012-12-21'
        }

    def load_data(self):
        """Load and display basic information about the dataset"""
        try:
            self.data = pd.read_csv(self.data_path)
            print("✅ Data loaded successfully!")
            print(f"Dataset shape: {self.data.shape}")
            print("\nDataset columns:")
            print(list(self.data.columns))
            print("\nMissing values before preprocessing:")
            print(self.data.isnull().sum())
            return True
        except FileNotFoundError:
            print(f"❌ Error: The file '{self.data_path}' was not found.")
            self.data = None
            return False
        except Exception as e:
            print(f"❌ An error occurred while loading data: {e}")
            self.data = None
            return False

    def explore_data(self):
        """Perform basic data exploration"""
        if self.data is None:
            print("No data to explore. Please load data first.")
            return

        print("\nBasic Data Description:")
        print(self.data.describe())

        # Plot weekly sales distribution
        plt.figure(figsize=(10, 6))
        sns.histplot(self.data['Weekly_Sales'], bins=50, kde=True)
        plt.title('Distribution of Weekly Sales')
        plt.xlabel('Weekly Sales')
        plt.ylabel('Frequency')
        plt.show()

    def preprocess_data(self):
        """
        Preprocess the data: handle dates, holidays, missing values, and encode categorical features.
        """
        if self.data is None:
            print("No data to preprocess. Please load data first.")
            self.processed_data = None
            return

        try:
            self.processed_data = self.data.copy()

            # 1. Convert 'Date' to datetime objects
            self.processed_data['Date'] = pd.to_datetime(self.processed_data['Date'], errors='coerce')

            # Handle NaT values
            if self.processed_data['Date'].isnull().any():
                initial_rows = len(self.processed_data)
                self.processed_data.dropna(subset=['Date'], inplace=True)
                rows_dropped = initial_rows - len(self.processed_data)
                if rows_dropped > 0:
                    print(f"⚠️ Warning: Dropped {rows_dropped} rows due to unparseable 'Date' entries.")
                if self.processed_data.empty:
                    print("Error: No valid date entries remaining.")
                    self.processed_data = None
                    return

            # 2. Handle 'IsHoliday' column
            if 'IsHoliday' not in self.processed_data.columns:
                print("⚠️ 'IsHoliday' column not found. Creating based on common holidays.")
                holiday_dates = {datetime.strptime(d, '%Y-%m-%d').date() for d in self.common_holidays}
                self.processed_data['IsHoliday'] = self.processed_data['Date'].dt.date.isin(holiday_dates)
            else:
                self.processed_data['IsHoliday'] = self.processed_data['IsHoliday'].astype(bool)

            # 3. Extract features from Date - FIXED VERSION
            self.processed_data['Year'] = self.processed_data['Date'].dt.year
            self.processed_data['Month'] = self.processed_data['Date'].dt.month
            self.processed_data['Day'] = self.processed_data['Date'].dt.day
            
            # FIX: Handle Week extraction properly
            try:
                # Use isocalendar() which returns a DataFrame with week, year, and weekday
                week_info = self.processed_data['Date'].dt.isocalendar()
                self.processed_data['Week'] = week_info.week.astype(int)
            except Exception as e:
                print(f"Warning: Issue with week extraction: {e}")
                # Fallback: use week of year
                self.processed_data['Week'] = self.processed_data['Date'].dt.isocalendar().week
                
            self.processed_data['DayOfWeek'] = self.processed_data['Date'].dt.dayofweek

            # 4. Handle missing values for numerical columns
            numerical_cols_to_impute = ['CPI', 'Fuel_Price', 'Unemployment', 'Temperature']
            for col in numerical_cols_to_impute:
                if col in self.processed_data.columns and self.processed_data[col].isnull().any():
                    median_val = self.processed_data[col].median()
                    self.processed_data[col].fillna(median_val, inplace=True)
                    print(f"Filled missing values in '{col}' with median: {median_val:.2f}")

            # 5. Handle categorical columns - IMPROVED VERSION
            categorical_cols = ['Store', 'Dept']
            
            # Check if 'Type' column exists before processing
            if 'Type' in self.processed_data.columns:
                categorical_cols.append('Type')
            else:
                print("⚠️ 'Type' column not found. Creating default 'Type' column.")
                # Create a default Type column if missing
                self.processed_data['Type'] = 'A'  # Default type
                categorical_cols.append('Type')

            # Label encode categorical features
            for col in categorical_cols:
                if col in self.processed_data.columns:
                    # Handle any missing values in categorical columns
                    if self.processed_data[col].isnull().any():
                        mode_val = self.processed_data[col].mode()
                        if len(mode_val) > 0:
                            self.processed_data[col].fillna(mode_val[0], inplace=True)
                        else:
                            self.processed_data[col].fillna('Unknown', inplace=True)
                    
                    # Convert to string to handle mixed types
                    self.processed_data[col] = self.processed_data[col].astype(str)
                    le = LabelEncoder()
                    self.processed_data[col] = le.fit_transform(self.processed_data[col])
                    self.label_encoders[col] = le
                    print(f"Label encoded '{col}'. Unique values: {len(le.classes_)}")

            # 6. Prepare features and target
            features_to_drop = ['Weekly_Sales', 'Date']
            features_to_drop_existing = [col for col in features_to_drop if col in self.processed_data.columns]
            
            self.X = self.processed_data.drop(columns=features_to_drop_existing, errors='ignore')
            self.y = self.processed_data['Weekly_Sales']

            # 7. Scale numerical features
            numerical_cols_to_scale = ['Temperature', 'Fuel_Price', 'CPI', 'Unemployment']
            numerical_cols_to_scale_existing = [col for col in numerical_cols_to_scale if col in self.X.columns]

            if numerical_cols_to_scale_existing:
                self.X[numerical_cols_to_scale_existing] = self.scaler.fit_transform(self.X[numerical_cols_to_scale_existing])
                print(f"Scaled numerical features: {numerical_cols_to_scale_existing}")

            # 8. Split data
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                self.X, self.y, test_size=0.2, random_state=42
            )
            
            print("✅ Data preprocessing complete!")
            print(f"Training data shape: {self.X_train.shape}")
            print(f"Testing data shape: {self.X_test.shape}")
            print(f"Feature columns: {list(self.X.columns)}")
            
        except Exception as e:
            print(f"❌ Error during preprocessing: {e}")
            self.processed_data = None
            return

    def train_models(self):
        """Train various regression models"""
        if self.X_train is None or self.y_train is None:
            print("Training data not available. Please preprocess data first.")
            return

        try:
            print("\n🚀 Training models...")

            # Random Forest Regressor
            print("  - Training Random Forest Regressor...")
            rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
            rf_model.fit(self.X_train, self.y_train)
            self.models['RandomForest'] = rf_model
            print("    Random Forest trained.")

            # Linear Regression
            print("  - Training Linear Regression...")
            lr_model = LinearRegression(n_jobs=-1)
            lr_model.fit(self.X_train, self.y_train)
            self.models['LinearRegression'] = lr_model
            print("    Linear Regression trained.")

            print("✅ Models trained successfully!")
            
        except Exception as e:
            print(f"❌ Error during model training: {e}")
            return

    def evaluate_models(self):
        """Evaluate trained models and print metrics"""
        if not self.models:
            print("No models to evaluate. Please train models first.")
            return

        try:
            print("\n📊 Model Evaluation:")
            results = {}
            for name, model in self.models.items():
                print(f"  - Evaluating {name}...")
                y_pred = model.predict(self.X_test)
                mae = mean_absolute_error(self.y_test, y_pred)
                rmse = np.sqrt(mean_squared_error(self.y_test, y_pred))
                r2 = r2_score(self.y_test, y_pred)

                results[name] = {'MAE': mae, 'RMSE': rmse, 'R²': r2}
                print(f"    {name} - MAE: {mae:.2f}, RMSE: {rmse:.2f}, R²: {r2:.4f}")
            return results
        except Exception as e:
            print(f"❌ Error during model evaluation: {e}")
            return {}

    def get_feature_names(self):
        """Return the list of feature names used for training."""
        if self.X_train is not None:
            return self.X_train.columns.tolist()
        return []

    def prepare_single_prediction_data(self, store_id, date_str, holiday_flag,
                                       temperature, fuel_price, cpi, unemployment, dept_id=1, type_id='A'):
        """
        Prepares a single row of data for prediction - IMPROVED VERSION
        """
        try:
            # Parse date
            date_obj = pd.to_datetime(date_str)
            
            # Determine holiday status
            is_holiday_val = bool(holiday_flag)
            if not is_holiday_val:
                is_holiday_val = date_obj.date() in {datetime.strptime(d, '%Y-%m-%d').date() for d in self.common_holidays}

            # Create prediction data
            new_data = {
                'Store': store_id,
                'Dept': dept_id,
                'Type': str(type_id),  # Ensure string type
                'Temperature': float(temperature),
                'Fuel_Price': float(fuel_price),
                'CPI': float(cpi),
                'Unemployment': float(unemployment),
                'IsHoliday': is_holiday_val,
                'Year': date_obj.year,
                'Month': date_obj.month,
                'Day': date_obj.day,
                'Week': date_obj.isocalendar().week,
                'DayOfWeek': date_obj.dayofweek
            }

            prediction_df = pd.DataFrame([new_data])

            # Apply label encoding with error handling
            for col in ['Store', 'Dept', 'Type']:
                if col in prediction_df.columns and col in self.label_encoders:
                    try:
                        # Convert to string first
                        prediction_df[col] = prediction_df[col].astype(str)
                        prediction_df[col] = self.label_encoders[col].transform(prediction_df[col])
                    except ValueError as e:
                        print(f"Warning: Category '{prediction_df[col].iloc[0]}' for column '{col}' not seen during training.")
                        # Use the most frequent category from training
                        if hasattr(self, 'X_train') and self.X_train is not None:
                            most_freq = self.X_train[col].mode()[0] if col in self.X_train.columns else 0
                            prediction_df[col] = most_freq
                        else:
                            prediction_df[col] = 0

            # Ensure all required columns exist
            if hasattr(self, 'X_train') and self.X_train is not None:
                # Add missing columns with default values
                for col in self.X_train.columns:
                    if col not in prediction_df.columns:
                        if col in ['Store', 'Dept', 'Type']:
                            prediction_df[col] = 0
                        else:
                            prediction_df[col] = self.X_train[col].median()

                # Reorder columns to match training data
                prediction_df = prediction_df[self.X_train.columns]

                # Scale numerical features
                numerical_cols_to_scale = ['Temperature', 'Fuel_Price', 'CPI', 'Unemployment']
                numerical_cols_to_scale_pred = [col for col in numerical_cols_to_scale if col in prediction_df.columns]
                
                if numerical_cols_to_scale_pred:
                    prediction_df[numerical_cols_to_scale_pred] = self.scaler.transform(prediction_df[numerical_cols_to_scale_pred])

            return prediction_df
            
        except Exception as e:
            print(f"❌ Error preparing prediction data: {e}")
            return None

    def predict_sales(self, store_id, date_str, holiday_flag, temperature, fuel_price, cpi, unemployment, dept_id=1, type_id='A', model_name='RandomForest'):
        """
        Predict sales for given parameters - NEW METHOD
        """
        try:
            # Prepare prediction data
            prediction_data = self.prepare_single_prediction_data(
                store_id, date_str, holiday_flag, temperature, fuel_price, cpi, unemployment, dept_id, type_id
            )
            
            if prediction_data is None:
                print("❌ Failed to prepare prediction data")
                return None
            
            # Get model
            model = self.models.get(model_name)
            if model is None:
                print(f"❌ Model '{model_name}' not found")
                return None
            
            # Make prediction
            prediction = model.predict(prediction_data)
            return float(prediction[0])
            
        except Exception as e:
            print(f"❌ Error during prediction: {e}")
            return None

    def get_model(self, model_name='RandomForest'):
        """Return a trained model by name."""
        return self.models.get(model_name)

    def run_complete_analysis(self):
        """Run the complete analysis pipeline"""
        print("🚀 WALMART SALES FORECASTING ANALYSIS")
        print("=" * 50)

        # Step 1: Load data
        if not self.load_data():
            return False

        # Step 2: Preprocess data
        self.preprocess_data()
        if self.processed_data is None:
            print("Analysis halted due to preprocessing errors.")
            return False

        # Step 3: Train models
        self.train_models()
        if not self.models:
            print("Analysis halted due to training errors.")
            return False

        # Step 4: Evaluate models
        results = self.evaluate_models()

        # Summary
        print("\n🎉 ANALYSIS COMPLETE!")
        print("=" * 30)
        if results:
            best_model_name = max(results, key=lambda x: results[x]['R²'])
            print(f"🏆 Best performing model: {best_model_name}")
            print(f"   R² Score: {results[best_model_name]['R²']:.4f}")
            print(f"   RMSE: ${results[best_model_name]['RMSE']:,.2f}")
        
        return True

    def merge_external_features(self, weather_df, trends_df):
        """
        Merge weather and Google Trends data into the main processed_data DataFrame.
        Assumes weather_df has columns: Date, City, Avg_Temperature, Precipitation
        and trends_df has columns: Date, Trends_<keyword1>, ...
        """
        # Merge on Date (and City if available)
        if 'City' in weather_df.columns and 'City' in self.processed_data.columns:
            self.processed_data = pd.merge(self.processed_data, weather_df, how='left', on=['Date', 'City'])
        else:
            self.processed_data = pd.merge(self.processed_data, weather_df, how='left', on='Date')
        self.processed_data = pd.merge(self.processed_data, trends_df, how='left', on='Date')
        print("✅ Merged weather and Google Trends features into processed_data.")

    def get_feature_importance(self, model_name='RandomForest'):
        """Return feature importances for the specified model (default: RandomForest)."""
        model = self.models.get(model_name)
        if model is not None and hasattr(model, 'feature_importances_'):
            return dict(zip(self.X_train.columns, model.feature_importances_))
        else:
            print(f"Model '{model_name}' does not support feature importances.")
            return {}

    def detect_anomalies(self, threshold=2.0, model_name='RandomForest'):
        """Detect anomalies where the absolute error is greater than threshold * std of errors."""
        model = self.models.get(model_name)
        if model is None:
            print(f"Model '{model_name}' not found for anomaly detection.")
            return pd.DataFrame()
        y_pred = model.predict(self.X_test)
        errors = self.y_test - y_pred
        std_error = np.std(errors)
        anomalies = np.abs(errors) > threshold * std_error
        result = self.X_test.copy()
        result['Actual_Sales'] = self.y_test
        result['Predicted_Sales'] = y_pred
        result['Error'] = errors
        result['Anomaly'] = anomalies
        return result[result['Anomaly']]

# Example usage
if __name__ == "__main__":
    forecaster = WalmartSalesForecaster(data_path='data/Walmart.csv')
    
    if forecaster.run_complete_analysis():
        # Test prediction
        predicted_sales = forecaster.predict_sales(
            store_id=1,
            date_str='2012-11-23',
            holiday_flag=1,
            temperature=50.0,
            fuel_price=3.0,
            cpi=215.0,
            unemployment=7.5,
            dept_id=1,
            type_id='A'
        )
        
        if predicted_sales is not None:
            print(f"\n🎯 Sample Prediction: ${predicted_sales:,.2f}")
        else:
            print("\n❌ Sample prediction failed")