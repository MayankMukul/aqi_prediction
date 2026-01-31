"""
Air Quality Index (AQI) Prediction 
----------------------------------
"""

from sklearn.impute import KNNImputer
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Machine Learning

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class AQIPredictorWithNaNHandling:

    def __init__(self):
        self.models = {}
        self.scaler = StandardScaler()
        self.results = {}
        self.missing_report = {}

    def load_real_data(self, filepath):

        print("\n" + "="*70)
        print("LOADING DATA")
        print("="*70)

        try:
            df = pd.read_csv(filepath)
            print(f"✓ Loaded {len(df)} rows from {filepath}")
            print(f"✓ Columns: {', '.join(df.columns.tolist())}")

            # Standardize column names
            df.columns = df.columns.str.lower().str.strip()

            # Handle different date formats
            date_cols = [col for col in df.columns if 'date' in col]
            if date_cols:
                df[date_cols[0]] = pd.to_datetime(
                    df[date_cols[0]], errors='coerce')
                if 'date' not in df.columns:
                    df = df.rename(columns={date_cols[0]: 'date'})

            return df

        except Exception as e:
            print(f"✗ Error loading file: {e}")
            print("  Using sample data instead...")
            return self.generate_sample_data_with_missing()

    def generate_sample_data_with_missing(self, n_samples=5000, missing_rate=0.15):
        """
        Generate sample data with realistic missing values
        """
        print(
            f"\nGenerating {n_samples} samples with {missing_rate*100:.0f}% missing values...")

        # Generate base data
        dates = pd.date_range(start='2020-01-01', periods=n_samples, freq='H')

        df = pd.DataFrame({
            'date': dates,
            'pm25': np.random.normal(50, 20, n_samples),
            'pm10': np.random.normal(75, 25, n_samples),
            'no2': np.random.normal(40, 15, n_samples),
            'so2': np.random.normal(20, 10, n_samples),
            'o3': np.random.normal(60, 20, n_samples),
            'co': np.random.normal(1.5, 0.5, n_samples)
        })

        # Clip to realistic ranges
        df['pm25'] = np.clip(df['pm25'], 0, 500)
        df['pm10'] = np.clip(df['pm10'], 0, 600)
        df['no2'] = np.clip(df['no2'], 0, 200)
        df['so2'] = np.clip(df['so2'], 0, 100)
        df['o3'] = np.clip(df['o3'], 0, 300)
        df['co'] = np.clip(df['co'], 0, 10)

        # Calculate AQI
        df['aqi'] = self._calculate_aqi_from_pm25(df['pm25'].values)

        # Introduce realistic missing patterns
        # 1. Random missing
        for col in ['pm25', 'pm10', 'no2', 'so2', 'o3', 'co']:
            mask = np.random.random(n_samples) < missing_rate
            df.loc[mask, col] = np.nan

        # 2. Continuous missing blocks (sensor failure)
        start_idx = np.random.randint(0, n_samples - 100, 3)
        for idx in start_idx:
            col = np.random.choice(['pm25', 'pm10', 'no2'])
            df.loc[idx:idx+50, col] = np.nan

        print(f"✓ Generated {len(df)} samples")
        print(f"✓ Total missing values: {df.isnull().sum().sum()}")

        return df

    def _calculate_aqi_from_pm25(self, pm25):

        pm25_bp = [0, 12, 35.4, 55.4, 150.4, 250.4, 350.4, 500]
        aqi_bp = [0, 50, 100, 150, 200, 300, 400, 500]

        aqi = np.zeros_like(pm25)
        for i in range(len(aqi)):
            pm_val = pm25[i]
            if np.isnan(pm_val):
                aqi[i] = np.nan
                continue
            for j in range(len(pm25_bp) - 1):
                if pm25_bp[j] <= pm_val <= pm25_bp[j + 1]:
                    aqi[i] = ((aqi_bp[j + 1] - aqi_bp[j]) / (pm25_bp[j + 1] - pm25_bp[j])) * \
                             (pm_val - pm25_bp[j]) + aqi_bp[j]
                    break
            if pm_val > pm25_bp[-1]:
                aqi[i] = 500
        return aqi

    def analyze_missing_values(self, df):

        print("\n" + "="*70)
        print("STEP 1: MISSING VALUE ANALYSIS")
        print("="*70)

        # Count missing values
        missing_count = df.isnull().sum()
        missing_percent = (missing_count / len(df)) * 100

        missing_df = pd.DataFrame({
            'Column': missing_count.index,
            'Missing_Count': missing_count.values,
            'Missing_Percent': missing_percent.values
        })
        missing_df = missing_df[missing_df['Missing_Count'] > 0].sort_values(
            'Missing_Percent', ascending=False
        )

        print("\n1. Missing Values by Column:")
        print("-" * 70)
        if len(missing_df) == 0:
            print("✓ No missing values found - data is complete!")
        else:
            for _, row in missing_df.iterrows():
                bar = '█' * int(row['Missing_Percent'] / 2)
                print(
                    f"{row['Column']:10s}: {row['Missing_Count']:5.0f} ({row['Missing_Percent']:5.1f}%) {bar}")

        # Overall statistics
        total_cells = len(df) * len(df.columns)
        missing_cells = df.isnull().sum().sum()
        complete_rows = (~df.isnull().any(axis=1)).sum()

        print("\n2. Overall Statistics:")
        print("-" * 70)
        print(f"Total cells:        {total_cells:,}")
        print(
            f"Missing cells:      {missing_cells:,} ({missing_cells/total_cells*100:.1f}%)")
        print(
            f"Complete rows:      {complete_rows:,} ({complete_rows/len(df)*100:.1f}%)")
        print(
            f"Incomplete rows:    {len(df)-complete_rows:,} ({(len(df)-complete_rows)/len(df)*100:.1f}%)")

        self.missing_report = {
            'total_missing': missing_cells,
            'missing_percent': (missing_cells/total_cells*100),
            'complete_rows': complete_rows
        }

        return missing_df

    def handle_missing_values(self, df, method='auto'):

        print("\n" + "="*70)
        print("STEP 2: HANDLING MISSING VALUES")
        print("="*70)

        missing_percent = (df.isnull().sum().sum() /
                           (len(df) * len(df.columns))) * 100

        # Auto-select method
        if method == 'auto':
            if missing_percent < 5:
                method = 'drop'
                print(
                    f"\nAuto-selected: DROP (missing = {missing_percent:.1f}% < 5%)")
            elif 'date' in df.columns and missing_percent < 20:
                method = 'interpolation'
                print(
                    f"\nAuto-selected: INTERPOLATION (time series, missing = {missing_percent:.1f}%)")
            elif missing_percent < 30:
                method = 'knn'
                print(
                    f"\nAuto-selected: KNN (missing = {missing_percent:.1f}% < 30%)")
            else:
                method = 'mean'
                print(
                    f"\nAuto-selected: MEAN (missing = {missing_percent:.1f}%)")

        print(f"Selected method: {method.upper()}")
        print("-" * 70)

        df_cleaned = df.copy()
        initial_missing = df_cleaned.isnull().sum().sum()
        initial_rows = len(df_cleaned)

        # Apply selected method
        if method == 'drop':
            # Drop rows with any missing values
            df_cleaned = df_cleaned.dropna()
            print(
                f"✓ Dropped {initial_rows - len(df_cleaned)} rows with missing values")

        elif method == 'mean':
            # Fill with column mean
            numeric_cols = df_cleaned.select_dtypes(
                include=[np.number]).columns
            for col in numeric_cols:
                if df_cleaned[col].isnull().sum() > 0:
                    mean_val = df_cleaned[col].mean()
                    filled = df_cleaned[col].isnull().sum()
                    df_cleaned[col] = df_cleaned[col].fillna(mean_val)
                    print(
                        f"  {col:10s}: Filled {filled:4d} values with mean = {mean_val:.2f}")

        elif method == 'median':
            # Fill with column median
            numeric_cols = df_cleaned.select_dtypes(
                include=[np.number]).columns
            for col in numeric_cols:
                if df_cleaned[col].isnull().sum() > 0:
                    median_val = df_cleaned[col].median()
                    filled = df_cleaned[col].isnull().sum()
                    df_cleaned[col] = df_cleaned[col].fillna(median_val)
                    print(
                        f"  {col:10s}: Filled {filled:4d} values with median = {median_val:.2f}")

        elif method == 'interpolation':
            # Linear interpolation
            numeric_cols = df_cleaned.select_dtypes(
                include=[np.number]).columns
            for col in numeric_cols:
                if df_cleaned[col].isnull().sum() > 0:
                    before = df_cleaned[col].isnull().sum()
                    df_cleaned[col] = df_cleaned[col].interpolate(
                        method='linear', limit_direction='both')
                    after = df_cleaned[col].isnull().sum()
                    print(
                        f"  {col:10s}: Filled {before-after:4d} values via interpolation")
            # Fill remaining with forward/backward fill
            df_cleaned = df_cleaned.fillna(
                method='ffill').fillna(method='bfill')

        elif method == 'knn':
            # KNN imputation
            numeric_cols = df_cleaned.select_dtypes(
                include=[np.number]).columns.tolist()
            if numeric_cols:
                print(
                    f"  Applying KNN imputation (k=5) to {len(numeric_cols)} numeric columns...")
                imputer = KNNImputer(n_neighbors=5)
                df_cleaned[numeric_cols] = imputer.fit_transform(
                    df_cleaned[numeric_cols])
                print(f"  ✓ KNN imputation complete")

        elif method == 'forward_fill':
            # Forward fill for time series
            df_cleaned = df_cleaned.fillna(
                method='ffill').fillna(method='bfill')
            print(f"  ✓ Applied forward fill")

        final_missing = df_cleaned.isnull().sum().sum()
        final_rows = len(df_cleaned)

        print("\nResults:")
        print("-" * 70)
        print(f"Initial missing:    {initial_missing}")
        print(f"Final missing:      {final_missing}")
        print(f"Values filled:      {initial_missing - final_missing}")
        print(
            f"Rows retained:      {final_rows}/{initial_rows} ({final_rows/initial_rows*100:.1f}%)")

        if final_missing > 0:
            print(f"\n⚠ Warning: {final_missing} missing values remain")
            print("  Consider using a different method or dropping these rows")
        else:
            print("\n✓ All missing values handled successfully!")

        return df_cleaned

    def preprocess_data(self, df):

        print("\n" + "="*70)
        print("STEP 3: DATA PREPROCESSING")
        print("="*70)

        # Add AQI category
        if 'aqi' in df.columns:
            df['aqi_category'] = df['aqi'].apply(self._categorize_aqi)

            print("\nAQI Category Distribution:")
            print("-" * 70)
            category_dist = df['aqi_category'].value_counts()
            for category, count in category_dist.items():
                percentage = (count / len(df)) * 100
                bar = '█' * int(percentage / 2)
                print(f"{category:20s}: {count:5d} ({percentage:5.1f}%) {bar}")

        return df

    def _categorize_aqi(self, aqi_value):

        if pd.isna(aqi_value):
            return "Unknown"
        elif aqi_value <= 50:
            return "Good"
        elif aqi_value <= 100:
            return "Moderate"
        elif aqi_value <= 200:
            return "Poor"
        else:
            return "Severe"

    def prepare_features(self, df):
        """Prepare features and target"""
        print("\n" + "="*70)
        print("STEP 4: FEATURE PREPARATION")
        print("="*70)

        # Select features
        feature_columns = []
        possible_features = ['pm25', 'pm10', 'no2', 'so2', 'o3', 'co']

        for col in possible_features:
            if col in df.columns:
                feature_columns.append(col)

        if not feature_columns:
            raise ValueError("No pollutant features found in dataset!")

        X = df[feature_columns]

        # Select target
        if 'aqi' in df.columns:
            y = df['aqi']
        else:
            raise ValueError("AQI column not found in dataset!")

        print(f"Features: {', '.join(feature_columns)}")
        print(f"Target: AQI")
        print(f"Shape: {X.shape}")

        return X, y, feature_columns

    def split_and_scale_data(self, X, y, test_size=0.2):
        """Split and scale data"""
        print("\n" + "="*70)
        print("STEP 5: DATA SPLITTING & SCALING")
        print("="*70)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )

        print(f"Training samples: {len(X_train)} ({(1-test_size)*100:.0f}%)")
        print(f"Testing samples:  {len(X_test)} ({test_size*100:.0f}%)")

        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        print("✓ Features scaled using StandardScaler")

        return X_train_scaled, X_test_scaled, y_train, y_test

    def train_models(self, X_train, y_train):
        """Train all models"""
        print("\n" + "="*70)
        print("STEP 6: MODEL TRAINING")
        print("="*70)

        # Linear Regression
        print("\n1. Training Linear Regression...")
        lr = LinearRegression()
        lr.fit(X_train, y_train)
        self.models['Linear Regression'] = lr
        print("✓ Complete")

        # Decision Tree
        print("\n2. Training Decision Tree...")
        dt = DecisionTreeRegressor(
            max_depth=10, min_samples_split=10, random_state=42)
        dt.fit(X_train, y_train)
        self.models['Decision Tree'] = dt
        print("✓ Complete")

        # Random Forest
        print("\n3. Training Random Forest...")
        rf = RandomForestRegressor(
            n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
        rf.fit(X_train, y_train)
        self.models['Random Forest'] = rf
        print("✓ Complete")

        return self.models

    def evaluate_models(self, X_test, y_test):
        """Evaluate all models"""
        print("\n" + "="*70)
        print("STEP 7: MODEL EVALUATION")
        print("="*70)

        for model_name, model in self.models.items():
            y_pred = model.predict(X_test)

            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)

            y_test_cat = y_test.apply(self._categorize_aqi)
            y_pred_cat = pd.Series(y_pred).apply(self._categorize_aqi)
            category_accuracy = (y_test_cat.values ==
                                 y_pred_cat.values).mean() * 100

            self.results[model_name] = {
                'rmse': rmse,
                'r2': r2,
                'mae': mae,
                'category_accuracy': category_accuracy
            }

            print(f"\n{model_name}:")
            print("-" * 70)
            print(f"RMSE:              {rmse:.2f}")
            print(f"R² Score:          {r2:.4f}")
            print(f"MAE:               {mae:.2f}")
            print(f"Category Accuracy: {category_accuracy:.2f}%")

        best_model = min(self.results.items(), key=lambda x: x[1]['rmse'])
        print("\n" + "="*70)
        print(
            f"BEST MODEL: {best_model[0]} (RMSE: {best_model[1]['rmse']:.2f})")
        print("="*70)

        return self.results


def main():
    """Main execution"""
    print("\n")
    print("╔" + "="*68 + "╗")
    print("║" + " "*8 + "AQI PREDICTION WITH MISSING VALUE HANDLING" + " "*18 + "║")
    print("╚" + "="*68 + "╝")

    predictor = AQIPredictorWithNaNHandling()

    # Option 1: Load data
    df = predictor.load_real_data('city_day.csv')

    # Analyze missing values
    predictor.analyze_missing_values(df)

    # Handle missing values
    df_cleaned = predictor.handle_missing_values(df, method='auto')

    # Preprocess
    df_cleaned = predictor.preprocess_data(df_cleaned)

    # Prepare features
    X, y, features = predictor.prepare_features(df_cleaned)

    # Split and scale
    X_train, X_test, y_train, y_test = predictor.split_and_scale_data(X, y)

    # Train models
    predictor.train_models(X_train, y_train)

    # Evaluate
    predictor.evaluate_models(X_test, y_test)

    # Save cleaned data
    df_cleaned.to_csv('aqi_data_final.csv', index=False)

    print("\n" + "="*70)
    print("PROCESS COMPLETE!")
    print("="*70)
    print("\n✓ Missing values handled successfully")
    print("✓ Models trained and evaluated")
    print("✓ Cleaned data saved to: aqi_data_final.csv")
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    main()
