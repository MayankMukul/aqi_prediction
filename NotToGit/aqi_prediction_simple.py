"""
Air Quality Index (AQI) Prediction Project
------------------------------------------

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Machine Learning Models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# For visualization
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class AQIPredictor:
    
    def __init__(self):
        self.models = {}
        self.scaler = StandardScaler()
        self.results = {}
        
    def generate_sample_data(self, n_samples=5000):
        """
        Generate sample air quality data
        Replace this with real data by loading CSV:
        df = pd.read_csv('your_aqi_data.csv')
        """
        # Creating sysnthetic dataset
        # print("Generating sample air quality dataset...")
        
        # # Create date range
        # dates = pd.date_range(start='2020-01-01', periods=n_samples, freq='H')
        
        # # Generate pollutant concentrations
        # # PM2.5: Fine particulate matter
        # pm25 = np.random.normal(50, 20, n_samples)
        # pm25 = np.clip(pm25, 0, 500)
        
        # # PM10: Coarse particulate matter
        # pm10 = pm25 * 1.5 + np.random.normal(0, 15, n_samples)
        # pm10 = np.clip(pm10, 0, 600)
        
        # # NO2: Nitrogen dioxide
        # no2 = np.random.normal(40, 15, n_samples)
        # no2 = np.clip(no2, 0, 200)
        
        # # SO2: Sulfur dioxide
        # so2 = np.random.normal(20, 10, n_samples)
        # so2 = np.clip(so2, 0, 100)
        
        # # O3: Ozone
        # o3 = np.random.normal(60, 20, n_samples)
        # o3 = np.clip(o3, 0, 300)
        
        # # CO: Carbon monoxide
        # co = np.random.normal(1.5, 0.5, n_samples)
        # co = np.clip(co, 0, 10)
        
        # # Calculate AQI from pollutants (simplified EPA formula)
        # aqi = self._calculate_aqi_from_pollutants(pm25, pm10, no2, so2, o3, co)
        
        # # Create DataFrame
        # df = pd.DataFrame({
        #     'date': dates,
        #     'pm25': pm25,
        #     'pm10': pm10,
        #     'no2': no2,
        #     'so2': so2,
        #     'o3': o3,
        #     'co': co,
        #     'aqi': aqi
        # })

        # Load real data
        
        
        df = pd.read_csv('aqi_data_final.csv')

        # Filter for Delhi
        df = df[df['city'] == 'Delhi'].copy()

        # Rename columns to match your code
        # df = df.rename(columns={
        #     'Date': 'date',
        #     'PM2.5': 'pm25',
        #     'PM10': 'pm10',
        #     'NO2': 'no2',
        #     'SO2': 'so2',
        #     'O3': 'o3',
        #     'CO': 'co',
        #     'AQI': 'aqi'
        # })
        
        print(f"✓ Generated {len(df)} samples")
        print(f"✓ Date range: {df['date'].min()} to {df['date'].max()}")
        return df
    
    def _calculate_aqi_from_pollutants(self, pm25, pm10, no2, so2, o3, co):
        """Calculate AQI based on PM2.5 (primary pollutant)"""
        pm25_bp = [0, 12, 35.4, 55.4, 150.4, 250.4, 350.4, 500]
        aqi_bp = [0, 50, 100, 150, 200, 300, 400, 500]
        
        aqi = np.zeros_like(pm25)
        for i in range(len(aqi)):
            pm_val = pm25[i]
            for j in range(len(pm25_bp) - 1):
                if pm25_bp[j] <= pm_val <= pm25_bp[j + 1]:
                    aqi[i] = ((aqi_bp[j + 1] - aqi_bp[j]) / (pm25_bp[j + 1] - pm25_bp[j])) * \
                             (pm_val - pm25_bp[j]) + aqi_bp[j]
                    break
            if pm_val > pm25_bp[-1]:
                aqi[i] = 500
        return aqi
    
    def preprocess_data(self, df):
        """
        Preprocess and analyze the dataset using Pandas
        """
        print("\n" + "="*60)
        print("PREPROCESSING AND ANALYZING DATA")
        print("="*60)
        
        # Basic statistics
        print("\n1. Dataset Overview:")
        print("-" * 60)
        print(f"Total samples: {len(df)}")
        print(f"Features: {', '.join(df.columns.tolist())}")
        print(f"\nMissing values:\n{df.isnull().sum()}")
        
        # Statistical summary
        print("\n2. Statistical Summary:")
        print("-" * 60)
        print(df.describe())
        
        # AQI category distribution
        df['aqi_category'] = df['aqi'].apply(self._categorize_aqi)
        print("\n3. AQI Category Distribution:")
        print("-" * 60)
        category_dist = df['aqi_category'].value_counts()
        for category, count in category_dist.items():
            percentage = (count / len(df)) * 100
            print(f"{category:20s}: {count:5d} ({percentage:5.1f}%)")
        
        # Check for outliers
        print("\n4. Data Quality Checks:")
        print("-" * 60)
        for col in ['pm25', 'pm10', 'no2', 'so2', 'o3', 'co']:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            outliers = ((df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR))).sum()
            print(f"{col:10s}: {outliers:4d} outliers detected")
        
        return df
    
    def _categorize_aqi(self, aqi_value):
        """Classify AQI into categories"""
        if aqi_value <= 50:
            return "Good"
        elif aqi_value <= 100:
            return "Moderate"
        elif aqi_value <= 200:
            return "Poor"
        else:
            return "Severe"
    
    def prepare_features(self, df):
        """Prepare features for modeling"""
        print("\n5. Feature Preparation:")
        print("-" * 60)
        
        # Select features (pollutants) and target (AQI)
        feature_columns = ['pm25', 'pm10', 'no2', 'so2', 'o3', 'co']
        X = df[feature_columns]
        y = df['aqi']
        
        print(f"Features: {', '.join(feature_columns)}")
        print(f"Target: AQI")
        print(f"Feature shape: {X.shape}")
        
        return X, y
    
    def split_and_scale_data(self, X, y, test_size=0.2):
        """Split data into train/test and scale features"""
        print("\n6. Data Splitting:")
        print("-" * 60)
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        print(f"Training samples: {len(X_train)} ({(1-test_size)*100:.0f}%)")
        print(f"Testing samples:  {len(X_test)} ({test_size*100:.0f}%)")
        
        # Scale the features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        print("✓ Features scaled using StandardScaler")
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def train_models(self, X_train, y_train):
        """
        Implement Linear Regression, Decision Tree, and Random Forest models
        """
        print("\n" + "="*60)
        print("TRAINING MACHINE LEARNING MODELS")
        print("="*60)
        
        # 1. Linear Regression
        print("\n1. Training Linear Regression...")
        lr_model = LinearRegression()
        lr_model.fit(X_train, y_train)
        self.models['Linear Regression'] = lr_model
        print("✓ Linear Regression trained")
        
        # 2. Decision Tree
        print("\n2. Training Decision Tree...")
        dt_model = DecisionTreeRegressor(
            max_depth=10,
            min_samples_split=10,
            random_state=42
        )
        dt_model.fit(X_train, y_train)
        self.models['Decision Tree'] = dt_model
        print("✓ Decision Tree trained")
        
        # 3. Random Forest
        print("\n3. Training Random Forest...")
        rf_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=15,
            min_samples_split=5,
            random_state=42,
            n_jobs=-1
        )
        rf_model.fit(X_train, y_train)
        self.models['Random Forest'] = rf_model
        print("✓ Random Forest trained")
        
        return self.models
    
    def evaluate_models(self, X_test, y_test):
        """
        Evaluate models using RMSE and R² score
        """
        print("\n" + "="*60)
        print("MODEL EVALUATION (RMSE and R² Score)")
        print("="*60)
        
        for model_name, model in self.models.items():
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            
            # Calculate category accuracy
            y_test_cat = y_test.apply(self._categorize_aqi)
            y_pred_cat = pd.Series(y_pred).apply(self._categorize_aqi)
            category_accuracy = (y_test_cat.values == y_pred_cat.values).mean() * 100
            
            # Store results
            self.results[model_name] = {
                'rmse': rmse,
                'r2': r2,
                'mae': mae,
                'category_accuracy': category_accuracy,
                'predictions': y_pred,
                'actual': y_test
            }
            
            # Print results
            print(f"\n{model_name}:")
            print("-" * 60)
            print(f"RMSE:              {rmse:.2f}")
            print(f"R² Score:          {r2:.4f}")
            print(f"MAE:               {mae:.2f}")
            print(f"Category Accuracy: {category_accuracy:.2f}%")
        
        # Find best model
        best_model = min(self.results.items(), key=lambda x: x[1]['rmse'])
        print("\n" + "="*60)
        print(f"BEST MODEL: {best_model[0]} (RMSE: {best_model[1]['rmse']:.2f})")
        print("="*60)
        
        return self.results
    
    def visualize_results(self, df, save_path='aqi_visualizations.png'):
        """
        Visualize AQI trends using Matplotlib
        """
        print("\n" + "="*60)
        print("CREATING VISUALIZATIONS")
        print("="*60)
        
        fig = plt.figure(figsize=(18, 12))
        
        # 1. AQI Trend Over Time
        ax1 = plt.subplot(3, 3, 1)
        plt.plot(df['date'][:1000], df['aqi'][:1000], linewidth=0.8, alpha=0.7)
        plt.xlabel('Date')
        plt.ylabel('AQI')
        plt.title('AQI Trend Over Time (First 1000 Hours)', fontweight='bold', fontsize=11)
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        # 2. AQI Category Distribution
        ax2 = plt.subplot(3, 3, 2)
        category_counts = df['aqi_category'].value_counts()
        colors = {'Good': '#00E400', 'Moderate': '#FFFF00', 
                  'Poor': '#FF7E00', 'Severe': '#FF0000'}
        bars = plt.bar(category_counts.index, category_counts.values,
                      color=[colors.get(cat, 'gray') for cat in category_counts.index],
                      edgecolor='black')
        plt.xlabel('AQI Category')
        plt.ylabel('Count')
        plt.title('AQI Category Distribution', fontweight='bold', fontsize=11)
        plt.xticks(rotation=45)
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}', ha='center', va='bottom')
        
        # 3. Pollutant Concentrations
        ax3 = plt.subplot(3, 3, 3)
        pollutants = ['pm25', 'pm10', 'no2', 'so2', 'o3', 'co']
        for pollutant in pollutants:
            if pollutant != 'co':
                plt.plot(df['date'][:500], df[pollutant][:500], 
                        label=pollutant.upper(), alpha=0.7, linewidth=1)
        plt.xlabel('Date')
        plt.ylabel('Concentration (μg/m³)')
        plt.title('Pollutant Concentrations Over Time', fontweight='bold', fontsize=11)
        plt.legend(loc='upper right', fontsize=8)
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        # 4. Model Comparison - RMSE
        ax4 = plt.subplot(3, 3, 4)
        models = list(self.results.keys())
        rmse_values = [self.results[m]['rmse'] for m in models]
        bars = plt.bar(models, rmse_values, color=['#3498db', '#2ecc71', '#e74c3c'],
                      edgecolor='black')
        plt.ylabel('RMSE')
        plt.title('Model Comparison - RMSE', fontweight='bold', fontsize=11)
        plt.xticks(rotation=45)
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}', ha='center', va='bottom')
        
        # 5. Model Comparison - R² Score
        ax5 = plt.subplot(3, 3, 5)
        r2_values = [self.results[m]['r2'] for m in models]
        bars = plt.bar(models, r2_values, color=['#3498db', '#2ecc71', '#e74c3c'],
                      edgecolor='black')
        plt.ylabel('R² Score')
        plt.title('Model Comparison - R² Score', fontweight='bold', fontsize=11)
        plt.xticks(rotation=45)
        plt.ylim([0, 1])
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.4f}', ha='center', va='bottom')
        
        # 6. Actual vs Predicted (Best Model)
        ax6 = plt.subplot(3, 3, 6)
        best_model_name = min(self.results.items(), key=lambda x: x[1]['rmse'])[0]
        actual = self.results[best_model_name]['actual']
        predicted = self.results[best_model_name]['predictions']
        
        plt.scatter(actual, predicted, alpha=0.5, s=10)
        plt.plot([actual.min(), actual.max()], [actual.min(), actual.max()], 
                'r--', lw=2, label='Perfect Prediction')
        plt.xlabel('Actual AQI')
        plt.ylabel('Predicted AQI')
        plt.title(f'Actual vs Predicted - {best_model_name}', fontweight='bold', fontsize=11)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 7. Correlation Heatmap
        ax7 = plt.subplot(3, 3, 7)
        corr_data = df[['pm25', 'pm10', 'no2', 'so2', 'o3', 'co', 'aqi']].corr()
        sns.heatmap(corr_data, annot=True, fmt='.2f', cmap='coolwarm', 
                   square=True, linewidths=0.5, cbar_kws={'shrink': 0.8})
        plt.title('Pollutant Correlation Matrix', fontweight='bold', fontsize=11)
        
        # 8. Category Accuracy by Model
        ax8 = plt.subplot(3, 3, 8)
        cat_acc = [self.results[m]['category_accuracy'] for m in models]
        bars = plt.bar(models, cat_acc, color=['#3498db', '#2ecc71', '#e74c3c'],
                      edgecolor='black')
        plt.ylabel('Accuracy (%)')
        plt.title('Category Classification Accuracy', fontweight='bold', fontsize=11)
        plt.xticks(rotation=45)
        plt.ylim([0, 100])
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}%', ha='center', va='bottom')
        
        # 9. AQI Distribution
        ax9 = plt.subplot(3, 3, 9)
        plt.hist(df['aqi'], bins=50, color='steelblue', edgecolor='black', alpha=0.7)
        plt.xlabel('AQI Value')
        plt.ylabel('Frequency')
        plt.title('AQI Distribution', fontweight='bold', fontsize=11)
        plt.axvline(50, color='green', linestyle='--', linewidth=1, label='Good')
        plt.axvline(100, color='yellow', linestyle='--', linewidth=1, label='Moderate')
        plt.axvline(200, color='orange', linestyle='--', linewidth=1, label='Poor')
        plt.legend(fontsize=8)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Visualizations saved to: {save_path}")
        
        return fig


def main():
    """
    Main function to run the complete AQI prediction pipeline
    """
    print("\n")
    print("╔" + "="*58 + "╗")
    print("║" + " "*12 + "AIR QUALITY INDEX (AQI) PREDICTION" + " "*12 + "║")
    print("╚" + "="*58 + "╝")
    
    # Initialize predictor
    predictor = AQIPredictor()
    
    # Step 1: Generate/Load data
    df = predictor.generate_sample_data(n_samples=5000)
    
    # Step 2: Preprocess and analyze data using Pandas
    df = predictor.preprocess_data(df)
    
    # Step 3: Prepare features
    X, y = predictor.prepare_features(df)
    
    # Step 4: Split and scale data
    X_train, X_test, y_train, y_test = predictor.split_and_scale_data(X, y)
    
    # Step 5: Train models (Linear Regression, Decision Tree, Random Forest)
    models = predictor.train_models(X_train, y_train)
    
    # Step 6: Evaluate models using RMSE and R² score
    results = predictor.evaluate_models(X_test, y_test)
    
    # Step 7: Visualize AQI trends using Matplotlib
    predictor.visualize_results(df)
    
    print("\n" + "="*60)
    print("PROJECT COMPLETED SUCCESSFULLY!")
    print("="*60)
    print("\n✓ All tasks completed as per resume bullet points:")
    print("  1. Preprocessed and analyzed air quality datasets using Pandas")
    print("  2. Implemented Linear Regression, Decision Tree, and Random Forest")
    print("  3. Evaluated models using RMSE and R² score")
    print("  4. Classified AQI into Good, Moderate, Poor, and Severe categories")
    print("  5. Visualized AQI trends using Matplotlib")
    print("\n" + "="*60 + "\n")


if __name__ == "__main__":
    main()
