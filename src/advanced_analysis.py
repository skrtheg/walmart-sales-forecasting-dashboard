import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures
import warnings
warnings.filterwarnings('ignore')

# Updated import path: from src.module_name import ClassName
from src.main import WalmartSalesForecaster
from sklearn.linear_model import LinearRegression

class AdvancedWalmartAnalyzer:
    def __init__(self):
        """Initialize advanced analyzer"""
        # Ensure data_path is correctly set for the forecaster, pointing to the data/ directory
        self.forecaster = WalmartSalesForecaster(data_path='data/Walmart.csv')
        self.advanced_models = {}
        self.X = None
        self.y = None

    def setup_data(self):
        """Setup data for advanced analysis"""
        print("🔧 Setting up data for advanced analysis...")

        # Use the forecaster's load and preprocess methods
        if not self.forecaster.load_data():
            print("Error: Failed to load data for advanced analysis.")
            return False

        self.forecaster.preprocess_data()

        # IMPORTANT: Ensure processed_data is not None after preprocessing
        if self.forecaster.processed_data is None:
            print("Error: Preprocessing failed in WalmartSalesForecaster. Cannot proceed with advanced analysis.")
            return False

        # Prepare features and target from the forecaster's processed data
        if 'Weekly_Sales' in self.forecaster.processed_data.columns:
            # Use the X and y already prepared by the forecaster
            self.X = self.forecaster.X
            self.y = self.forecaster.y
        else:
            print("Error: 'Weekly_Sales' column not found in processed data. Cannot perform analysis.")
            return False

        if self.X is None or self.y is None:
            print("Error: Features or target are None after forecaster setup. Cannot proceed.")
            return False

        print("✅ Data setup complete for advanced analysis!")
        return True

    def feature_importance_analysis(self):
        """Perform feature importance analysis using RandomForest"""
        if self.X is None or self.y is None:
            print("Data not set up for feature importance. Run setup_data first.")
            return pd.DataFrame()

        print("\n🔎 Performing Feature Importance Analysis...")
        # Use a fresh RandomForestRegressor for feature importance
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        rf_model.fit(self.X, self.y)

        importance_df = pd.DataFrame({
            'Feature': self.X.columns,
            'Importance': rf_model.feature_importances_
        }).sort_values(by='Importance', ascending=False)

        print("Feature Importance:")
        print(importance_df.head(10)) # Print top 10 important features

        plt.figure(figsize=(12, 7))
        sns.barplot(x='Importance', y='Feature', data=importance_df.head(10))
        plt.title('Top 10 Feature Importances')
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.tight_layout()
        plt.show()

        return importance_df

    def hyperparameter_tuning(self):
        """Perform hyperparameter tuning for RandomForestRegressor using GridSearchCV"""
        if self.X is None or self.y is None:
            print("Data not set up for hyperparameter tuning. Run setup_data first.")
            return None, None

        print("\n⚙️ Performing Hyperparameter Tuning for RandomForestRegressor...")
        param_grid = {
            'n_estimators': [50, 100], # Reduced for faster execution
            'max_features': [0.6, 0.8], # Reduced for faster execution
            'min_samples_split': [2, 5]
        }
        rf = RandomForestRegressor(random_state=42, n_jobs=-1)
        grid_search = GridSearchCV(estimator=rf, param_grid=param_grid,
                                   cv=2, n_jobs=-1, verbose=1, scoring='r2') # Reduced cv for speed

        grid_search.fit(self.X, self.y)

        best_params = grid_search.best_params_
        best_score = grid_search.best_score_
        best_model = grid_search.best_estimator_

        self.advanced_models['RandomForest_Tuned'] = best_model

        print(f"✅ Best parameters found: {best_params}")
        print(f"✅ Best cross-validation R² score: {best_score:.4f}")

        return best_model, best_params

    def advanced_models_comparison(self):
        """Compare performance of advanced models (Gradient Boosting, Polynomial Regression)"""
        if self.X is None or self.y is None or self.forecaster.X_train is None:
            print("Data not set up for model comparison or forecaster's train/test split is missing. Run setup_data first.")
            return {}

        print("\n🔬 Comparing Advanced Models...")
        X_train_adv, X_test_adv, y_train_adv, y_test_adv = self.forecaster.X_train, self.forecaster.X_test, self.forecaster.y_train, self.forecaster.y_test

        results = {}

        # Gradient Boosting Regressor
        print("  - Training Gradient Boosting Regressor...")
        gb_model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
        gb_model.fit(X_train_adv, y_train_adv)
        y_pred_gb = gb_model.predict(X_test_adv)
        results['GradientBoosting'] = {
            'MAE': mean_absolute_error(y_test_adv, y_pred_gb),
            'RMSE': np.sqrt(mean_squared_error(y_test_adv, y_pred_gb)),
            'R²': r2_score(y_test_adv, y_pred_gb)
        }
        print(f"    Gradient Boosting - R²: {results['GradientBoosting']['R²']:.4f}")
        self.advanced_models['GradientBoosting'] = gb_model

        # Polynomial Regression (as an example of non-linear approach)
        poly_features = ['Temperature', 'Fuel_Price', 'CPI', 'Unemployment']
        poly_features_existing = [col for col in poly_features if col in X_train_adv.columns]

        if poly_features_existing:
            print("  - Training Polynomial Regression...")
            poly = PolynomialFeatures(degree=2, include_bias=False)
            X_train_poly = poly.fit_transform(X_train_adv[poly_features_existing])
            X_test_poly = poly.transform(X_test_adv[poly_features_existing])

            lr_poly_model = LinearRegression(n_jobs=-1)
            lr_poly_model.fit(X_train_poly, y_train_adv)
            y_pred_lr_poly = lr_poly_model.predict(X_test_poly)
            results['PolynomialRegression'] = {
                'MAE': mean_absolute_error(y_test_adv, y_pred_lr_poly),
                'RMSE': np.sqrt(mean_squared_error(y_test_adv, y_pred_lr_poly)),
                'R²': r2_score(y_test_adv, y_pred_lr_poly)
            }
            print(f"    Polynomial Regression - R²: {results['PolynomialRegression']['R²']:.4f}")
            self.advanced_models['PolynomialRegression'] = lr_poly_model
        else:
            print("  - Skipping Polynomial Regression: Required numerical features not found.")


        print("\nModel Comparison Results:")
        for model_name, metrics in results.items():
            print(f"  {model_name}: R²={metrics['R²']:.4f}, RMSE={metrics['RMSE']:.2f}, MAE={metrics['MAE']:.2f}")

        return results

    def sales_trend_analysis(self):
        """Analyze sales trends (seasonal, yearly)"""
        if self.forecaster.processed_data is None:
            print("Processed data not available for trend analysis. Run setup_data first.")
            return None

        print("\n📈 Performing Sales Trend Analysis...")

        # Ensure 'Weekly_Sales' is present
        if 'Weekly_Sales' not in self.forecaster.processed_data.columns:
            print("Error: 'Weekly_Sales' column not found for trend analysis.")
            return None

        # Yearly Trend
        yearly_sales = self.forecaster.processed_data.groupby('Year')['Weekly_Sales'].sum().reset_index()
        print("\nYearly Sales Trend:")
        print(yearly_sales)
        plt.figure(figsize=(10, 5))
        sns.lineplot(x='Year', y='Weekly_Sales', data=yearly_sales, marker='o')
        plt.title('Yearly Sales Trend')
        plt.xlabel('Year')
        plt.ylabel('Total Weekly Sales')
        plt.grid(True)
        plt.show()

        # Monthly Trend
        monthly_sales = self.forecaster.processed_data.groupby('Month')['Weekly_Sales'].mean().reset_index()
        print("\nMonthly Average Sales Trend:")
        print(monthly_sales)
        plt.figure(figsize=(10, 5))
        sns.lineplot(x='Month', y='Weekly_Sales', data=monthly_sales, marker='o')
        plt.title('Monthly Average Sales Trend')
        plt.xlabel('Month')
        plt.ylabel('Average Weekly Sales')
        plt.grid(True)
        plt.show()

        # Weekly Trend (by ISO week number)
        weekly_sales_iso = self.forecaster.processed_data.groupby('Week')['Weekly_Sales'].mean().reset_index()
        print("\nWeekly Average Sales Trend (ISO Week):")
        print(weekly_sales_iso)
        plt.figure(figsize=(12, 6))
        sns.lineplot(x='Week', y='Weekly_Sales', data=weekly_sales_iso, marker='o')
        plt.title('Weekly Average Sales Trend (ISO Week)')
        plt.xlabel('Week of Year')
        plt.ylabel('Average Weekly Sales')
        plt.grid(True)
        plt.show()

        # Holiday Effect
        if 'IsHoliday' in self.forecaster.processed_data.columns:
            holiday_sales = self.forecaster.processed_data.groupby('IsHoliday')['Weekly_Sales'].mean().reset_index()
            print("\nAverage Sales during Holidays vs. Non-Holidays:")
            print(holiday_sales)
            plt.figure(figsize=(7, 5))
            sns.barplot(x='IsHoliday', y='Weekly_Sales', data=holiday_sales)
            plt.title('Average Weekly Sales: Holiday vs. Non-Holiday')
            plt.xlabel('Is Holiday')
            plt.ylabel('Average Weekly Sales')
            plt.xticks(ticks=[0, 1], labels=['False', 'True'])
            plt.show()
        else:
            print("Cannot analyze Holiday Effect: 'IsHoliday' column not available.")


        return {
            'yearly_sales': yearly_sales,
            'monthly_sales': monthly_sales,
            'weekly_iso_sales': weekly_sales_iso,
            'holiday_sales': holiday_sales if 'IsHoliday' in self.forecaster.processed_data.columns else "N/A"
        }

    def store_performance_analysis(self):
        """Analyze performance across different stores and types"""
        if self.forecaster.processed_data is None:
            print("Processed data not available for store performance analysis. Run setup_data first.")
            return None

        print("\n🏢 Performing Store Performance Analysis...")

        # Total sales per store
        store_sales = self.forecaster.processed_data.groupby('Store')['Weekly_Sales'].sum().sort_values(ascending=False).reset_index()
        print("\nTop 5 Stores by Total Sales:")
        print(store_sales.head())
        print("\nBottom 5 Stores by Total Sales:")
        print(store_sales.tail())

        plt.figure(figsize=(14, 7))
        sns.barplot(x='Store', y='Weekly_Sales', data=store_sales.head(10), palette='viridis')
        plt.title('Top 10 Stores by Total Weekly Sales')
        plt.xlabel('Store ID')
        plt.ylabel('Total Weekly Sales')
        plt.show()

        # Sales by Store Type
        if 'Type' in self.forecaster.processed_data.columns:
            type_sales = self.forecaster.processed_data.groupby('Type')['Weekly_Sales'].mean().reset_index()
            print("\nAverage Sales by Store Type:")
            # If 'Type' was label encoded, decode it for better readability if the encoder is available
            if 'Type' in self.forecaster.label_encoders:
                # Create a temporary series with the encoded values to inverse transform
                temp_series = pd.Series(type_sales['Type'])
                type_sales['Type'] = self.forecaster.label_encoders['Type'].inverse_transform(temp_series)
            print(type_sales)
            plt.figure(figsize=(8, 5))
            sns.barplot(x='Type', y='Weekly_Sales', data=type_sales, palette='plasma')
            plt.title('Average Weekly Sales by Store Type')
            plt.xlabel('Store Type')
            plt.ylabel('Average Weekly Sales')
            plt.show()
        else:
            print("Cannot analyze sales by Store Type: 'Type' column not available.")


        return {
            'store_sales': store_sales,
            'type_sales': type_sales if 'Type' in self.forecaster.processed_data.columns else "N/A"
        }


    def run_complete_advanced_analysis(self):
        """Run the complete advanced analysis pipeline"""
        if not self.setup_data():
            print("Advanced analysis setup failed. Exiting.")
            return {}

        importance_df = self.feature_importance_analysis()

        # Hyperparameter tuning
        best_model, best_params = self.hyperparameter_tuning()

        # Advanced model comparison
        model_results = self.advanced_models_comparison()

        # Sales trend analysis
        trend_results = self.sales_trend_analysis()

        # Store performance analysis
        store_results = self.store_performance_analysis()

        print("\n🎉 ADVANCED ANALYSIS COMPLETE!")
        print("=" * 40)
        print("✅ Feature importance analyzed")
        print("✅ Hyperparameters optimized")
        print("✅ Multiple models compared")
        print("✅ Sales trends identified")
        print("✅ Store performance evaluated")

        return {
            'feature_importance': importance_df,
            'best_model': best_model,
            'model_comparison': model_results,
            'trends': trend_results,
            'store_performance': store_results
        }

def main():
    """Main function"""
    analyzer = AdvancedWalmartAnalyzer()
    results = analyzer.run_complete_advanced_analysis()

    print("\n📋 INSIGHTS SUMMARY:")
    print("=" * 30)
    print("1. Check feature importance to understand key drivers")
    print("2. Review optimized model parameters")
    print("3. Analyze seasonal and yearly trends")
    print("4. Identify top and bottom performing stores")
    print("5. Use insights for strategic decision making")

if __name__ == "__main__":
    main()