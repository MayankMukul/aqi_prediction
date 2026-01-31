"""
Comprehensive Missing Value (NaN) Handling for AQI Data
-------------------------------------------------------
This module provides multiple strategies to handle missing values in air quality datasets.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import warnings
warnings.filterwarnings('ignore')


class MissingValueHandler:
    """
    Handle missing values in AQI datasets with multiple strategies
    """
    
    def __init__(self):
        self.missing_report = {}
        
    def analyze_missing_values(self, df):
        """
        Comprehensive analysis of missing values
        """
        print("\n" + "="*70)
        print("MISSING VALUE ANALYSIS")
        print("="*70)
        
        # 1. Count missing values
        missing_count = df.isnull().sum()
        missing_percent = (df.isnull().sum() / len(df)) * 100
        
        missing_df = pd.DataFrame({
            'Column': missing_count.index,
            'Missing_Count': missing_count.values,
            'Missing_Percent': missing_percent.values
        })
        missing_df = missing_df[missing_df['Missing_Count'] > 0].sort_values(
            'Missing_Percent', ascending=False
        )
        
        print("\n1. Missing Value Summary:")
        print("-" * 70)
        if len(missing_df) == 0:
            print("✓ No missing values found!")
        else:
            print(missing_df.to_string(index=False))
        
        # 2. Missing value patterns
        print("\n2. Missing Value Patterns:")
        print("-" * 70)
        total_rows = len(df)
        rows_with_missing = df.isnull().any(axis=1).sum()
        complete_rows = total_rows - rows_with_missing
        
        print(f"Total rows:           {total_rows}")
        print(f"Complete rows:        {complete_rows} ({complete_rows/total_rows*100:.1f}%)")
        print(f"Rows with missing:    {rows_with_missing} ({rows_with_missing/total_rows*100:.1f}%)")
        
        # 3. Missing value by pollutant
        pollutant_cols = ['pm25', 'pm10', 'no2', 'so2', 'o3', 'co']
        available_pollutants = [col for col in pollutant_cols if col in df.columns]
        
        if available_pollutants:
            print("\n3. Missing Values by Pollutant:")
            print("-" * 70)
            for col in available_pollutants:
                missing = df[col].isnull().sum()
                percent = (missing / len(df)) * 100
                print(f"{col.upper():10s}: {missing:6d} missing ({percent:5.1f}%)")
        
        # Store report
        self.missing_report = {
            'total_rows': total_rows,
            'complete_rows': complete_rows,
            'rows_with_missing': rows_with_missing,
            'missing_by_column': missing_df
        }
        
        return missing_df
    
    def visualize_missing_values(self, df, save_path='/home/claude/missing_values_viz.png'):
        """
        Create visualizations for missing value patterns
        """
        print("\n4. Creating Missing Value Visualizations...")
        print("-" * 70)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. Missing value heatmap
        ax1 = axes[0, 0]
        # Select only numeric columns with some missing values
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        cols_with_missing = [col for col in numeric_cols if df[col].isnull().sum() > 0]
        
        if cols_with_missing:
            missing_data = df[cols_with_missing].isnull()
            if len(missing_data) > 1000:
                missing_data = missing_data.sample(1000)  # Sample for better visualization
            
            sns.heatmap(missing_data, yticklabels=False, cbar=True, 
                       cmap='viridis', ax=ax1)
            ax1.set_title('Missing Value Heatmap (Yellow = Missing)', fontweight='bold')
            ax1.set_xlabel('Columns')
        else:
            ax1.text(0.5, 0.5, 'No Missing Values!', 
                    ha='center', va='center', fontsize=16)
            ax1.set_title('Missing Value Heatmap', fontweight='bold')
        
        # 2. Missing value bar chart
        ax2 = axes[0, 1]
        missing_count = df.isnull().sum()
        missing_count = missing_count[missing_count > 0].sort_values(ascending=False)
        
        if len(missing_count) > 0:
            missing_count.plot(kind='barh', ax=ax2, color='coral', edgecolor='black')
            ax2.set_xlabel('Number of Missing Values')
            ax2.set_title('Missing Values by Column', fontweight='bold')
            ax2.grid(True, alpha=0.3, axis='x')
        else:
            ax2.text(0.5, 0.5, 'No Missing Values!', 
                    ha='center', va='center', fontsize=16)
            ax2.set_title('Missing Values by Column', fontweight='bold')
        
        # 3. Missing percentage pie chart
        ax3 = axes[1, 0]
        complete_rows = (~df.isnull().any(axis=1)).sum()
        incomplete_rows = df.isnull().any(axis=1).sum()
        
        if incomplete_rows > 0:
            sizes = [complete_rows, incomplete_rows]
            labels = [f'Complete\n({complete_rows})', f'Incomplete\n({incomplete_rows})']
            colors = ['#2ecc71', '#e74c3c']
            explode = (0, 0.1)
            
            ax3.pie(sizes, explode=explode, labels=labels, colors=colors,
                   autopct='%1.1f%%', shadow=True, startangle=90)
            ax3.set_title('Row Completeness', fontweight='bold')
        else:
            ax3.text(0.5, 0.5, '100% Complete!', 
                    ha='center', va='center', fontsize=16, color='green')
            ax3.set_title('Row Completeness', fontweight='bold')
        
        # 4. Missing value timeline (if date column exists)
        ax4 = axes[1, 1]
        if 'date' in df.columns:
            df_temp = df.copy()
            df_temp['date'] = pd.to_datetime(df_temp['date'], errors='coerce')
            df_temp = df_temp.set_index('date')
            
            # Count missing values per day
            missing_per_day = df_temp.isnull().sum(axis=1).resample('D').sum()
            
            if missing_per_day.sum() > 0:
                missing_per_day.plot(ax=ax4, color='red', linewidth=1.5)
                ax4.set_xlabel('Date')
                ax4.set_ylabel('Missing Values per Day')
                ax4.set_title('Missing Values Over Time', fontweight='bold')
                ax4.grid(True, alpha=0.3)
            else:
                ax4.text(0.5, 0.5, 'No Missing Values!', 
                        ha='center', va='center', fontsize=16)
                ax4.set_title('Missing Values Over Time', fontweight='bold')
        else:
            ax4.text(0.5, 0.5, 'No Date Column', 
                    ha='center', va='center', fontsize=14)
            ax4.set_title('Missing Values Over Time', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Visualization saved to: {save_path}")
        
        return fig
    
    def method_1_drop_rows(self, df, threshold=0.5):
        """
        Method 1: Drop rows with too many missing values
        """
        print("\n" + "="*70)
        print("METHOD 1: DROP ROWS WITH MISSING VALUES")
        print("="*70)
        
        initial_rows = len(df)
        
        # Drop rows where more than threshold of values are missing
        min_non_missing = int(threshold * len(df.columns))
        df_cleaned = df.dropna(thresh=min_non_missing)
        
        rows_dropped = initial_rows - len(df_cleaned)
        
        print(f"\nThreshold: Keep rows with at least {threshold*100:.0f}% non-missing values")
        print(f"Initial rows:     {initial_rows}")
        print(f"Rows dropped:     {rows_dropped} ({rows_dropped/initial_rows*100:.1f}%)")
        print(f"Remaining rows:   {len(df_cleaned)} ({len(df_cleaned)/initial_rows*100:.1f}%)")
        
        return df_cleaned
    
    def method_2_drop_columns(self, df, threshold=0.3):
        """
        Method 2: Drop columns with too many missing values
        """
        print("\n" + "="*70)
        print("METHOD 2: DROP COLUMNS WITH MISSING VALUES")
        print("="*70)
        
        initial_cols = len(df.columns)
        
        # Calculate missing percentage for each column
        missing_percent = (df.isnull().sum() / len(df))
        
        # Drop columns with more than threshold missing
        cols_to_drop = missing_percent[missing_percent > threshold].index.tolist()
        df_cleaned = df.drop(columns=cols_to_drop)
        
        print(f"\nThreshold: Drop columns with > {threshold*100:.0f}% missing values")
        print(f"Initial columns:    {initial_cols}")
        print(f"Columns dropped:    {len(cols_to_drop)}")
        if cols_to_drop:
            print(f"Dropped columns:    {', '.join(cols_to_drop)}")
        print(f"Remaining columns:  {len(df_cleaned.columns)}")
        
        return df_cleaned
    
    def method_3_forward_fill(self, df):
        """
        Method 3: Forward Fill (use previous value)
        Good for time series data
        """
        print("\n" + "="*70)
        print("METHOD 3: FORWARD FILL (Time Series)")
        print("="*70)
        
        df_cleaned = df.copy()
        
        # Forward fill for each column
        pollutant_cols = ['pm25', 'pm10', 'no2', 'so2', 'o3', 'co', 'aqi']
        available_cols = [col for col in pollutant_cols if col in df_cleaned.columns]
        
        for col in available_cols:
            before_missing = df_cleaned[col].isnull().sum()
            df_cleaned[col] = df_cleaned[col].fillna(method='ffill')
            after_missing = df_cleaned[col].isnull().sum()
            filled = before_missing - after_missing
            
            if before_missing > 0:
                print(f"{col.upper():10s}: Filled {filled:5d} values ({filled/before_missing*100:.1f}%)")
        
        # Fill any remaining NaN at start with backward fill
        df_cleaned = df_cleaned.fillna(method='bfill')
        
        remaining_missing = df_cleaned.isnull().sum().sum()
        print(f"\n✓ Remaining missing values: {remaining_missing}")
        
        return df_cleaned
    
    def method_4_mean_imputation(self, df):
        """
        Method 4: Fill with mean/median
        Simple but can introduce bias
        """
        print("\n" + "="*70)
        print("METHOD 4: MEAN/MEDIAN IMPUTATION")
        print("="*70)
        
        df_cleaned = df.copy()
        
        # Impute numeric columns
        numeric_cols = df_cleaned.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if df_cleaned[col].isnull().sum() > 0:
                before_missing = df_cleaned[col].isnull().sum()
                
                # Use median for robustness to outliers
                median_value = df_cleaned[col].median()
                df_cleaned[col] = df_cleaned[col].fillna(median_value)
                
                print(f"{col:15s}: Filled {before_missing:5d} values with median = {median_value:.2f}")
        
        remaining_missing = df_cleaned.isnull().sum().sum()
        print(f"\n✓ Remaining missing values: {remaining_missing}")
        
        return df_cleaned
    
    def method_5_interpolation(self, df):
        """
        Method 5: Linear Interpolation
        Good for time series with smooth trends
        """
        print("\n" + "="*70)
        print("METHOD 5: LINEAR INTERPOLATION")
        print("="*70)
        
        df_cleaned = df.copy()
        
        # Interpolate numeric columns
        numeric_cols = df_cleaned.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if df_cleaned[col].isnull().sum() > 0:
                before_missing = df_cleaned[col].isnull().sum()
                
                # Linear interpolation
                df_cleaned[col] = df_cleaned[col].interpolate(method='linear', limit_direction='both')
                
                after_missing = df_cleaned[col].isnull().sum()
                filled = before_missing - after_missing
                
                print(f"{col:15s}: Filled {filled:5d} values using interpolation")
        
        # Fill any remaining with forward/backward fill
        df_cleaned = df_cleaned.fillna(method='ffill').fillna(method='bfill')
        
        remaining_missing = df_cleaned.isnull().sum().sum()
        print(f"\n✓ Remaining missing values: {remaining_missing}")
        
        return df_cleaned
    
    def method_6_knn_imputation(self, df, n_neighbors=5):
        """
        Method 6: KNN Imputation
        Uses similar samples to fill missing values
        Advanced and accurate
        """
        print("\n" + "="*70)
        print(f"METHOD 6: KNN IMPUTATION (k={n_neighbors})")
        print("="*70)
        
        df_cleaned = df.copy()
        
        # Select only numeric columns
        numeric_cols = df_cleaned.select_dtypes(include=[np.number]).columns.tolist()
        
        if not numeric_cols:
            print("No numeric columns to impute")
            return df_cleaned
        
        # Apply KNN imputation
        imputer = KNNImputer(n_neighbors=n_neighbors)
        
        before_missing = df_cleaned[numeric_cols].isnull().sum().sum()
        
        df_cleaned[numeric_cols] = imputer.fit_transform(df_cleaned[numeric_cols])
        
        after_missing = df_cleaned[numeric_cols].isnull().sum().sum()
        filled = before_missing - after_missing
        
        print(f"\nTotal missing values filled: {filled}")
        print(f"Method: K-Nearest Neighbors (k={n_neighbors})")
        print(f"✓ All numeric columns imputed")
        
        remaining_missing = df_cleaned.isnull().sum().sum()
        print(f"✓ Remaining missing values: {remaining_missing}")
        
        return df_cleaned
    
    def method_7_iterative_imputation(self, df):
        """
        Method 7: Iterative Imputation (MICE)
        Most advanced - models each feature with missing values
        """
        print("\n" + "="*70)
        print("METHOD 7: ITERATIVE IMPUTATION (MICE Algorithm)")
        print("="*70)
        
        df_cleaned = df.copy()
        
        # Select only numeric columns
        numeric_cols = df_cleaned.select_dtypes(include=[np.number]).columns.tolist()
        
        if not numeric_cols:
            print("No numeric columns to impute")
            return df_cleaned
        
        # Apply iterative imputation
        imputer = IterativeImputer(max_iter=10, random_state=42)
        
        before_missing = df_cleaned[numeric_cols].isnull().sum().sum()
        
        df_cleaned[numeric_cols] = imputer.fit_transform(df_cleaned[numeric_cols])
        
        after_missing = df_cleaned[numeric_cols].isnull().sum().sum()
        filled = before_missing - after_missing
        
        print(f"\nTotal missing values filled: {filled}")
        print(f"Method: Multiple Imputation by Chained Equations (MICE)")
        print(f"✓ All numeric columns imputed")
        
        remaining_missing = df_cleaned.isnull().sum().sum()
        print(f"✓ Remaining missing values: {remaining_missing}")
        
        return df_cleaned
    
    def compare_methods(self, df):
        """
        Compare all imputation methods
        """
        print("\n" + "="*70)
        print("COMPARING ALL IMPUTATION METHODS")
        print("="*70)
        
        results = {}
        
        # Store original missing count
        original_missing = df.isnull().sum().sum()
        original_rows = len(df)
        
        methods = {
            'Drop Rows (50%)': lambda x: self.method_1_drop_rows(x, threshold=0.5),
            'Drop Columns (30%)': lambda x: self.method_2_drop_columns(x, threshold=0.3),
            'Forward Fill': self.method_3_forward_fill,
            'Mean Imputation': self.method_4_mean_imputation,
            'Interpolation': self.method_5_interpolation,
            'KNN (k=5)': lambda x: self.method_6_knn_imputation(x, n_neighbors=5),
            'Iterative (MICE)': self.method_7_iterative_imputation
        }
        
        print("\nMethod Comparison:")
        print("-" * 70)
        print(f"{'Method':<25} {'Rows':<12} {'Missing':<12} {'Data Loss %':<12}")
        print("-" * 70)
        
        for name, method in methods.items():
            try:
                df_temp = df.copy()
                df_cleaned = method(df_temp)
                
                remaining_missing = df_cleaned.isnull().sum().sum()
                remaining_rows = len(df_cleaned)
                data_loss = ((original_rows - remaining_rows) / original_rows) * 100
                
                results[name] = {
                    'rows': remaining_rows,
                    'missing': remaining_missing,
                    'data_loss': data_loss
                }
                
                print(f"{name:<25} {remaining_rows:<12} {remaining_missing:<12} {data_loss:<12.1f}")
                
            except Exception as e:
                print(f"{name:<25} ERROR: {str(e)[:30]}")
        
        print("-" * 70)
        
        return results
    
    def recommend_method(self, df):
        """
        Recommend best method based on data characteristics
        """
        print("\n" + "="*70)
        print("METHOD RECOMMENDATION")
        print("="*70)
        
        missing_percent = (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
        has_date = 'date' in df.columns or 'datetime' in df.columns
        n_rows = len(df)
        
        print(f"\nData Characteristics:")
        print(f"  Total missing:  {missing_percent:.1f}%")
        print(f"  Has date:       {has_date}")
        print(f"  Rows:           {n_rows}")
        
        print("\n" + "-" * 70)
        print("RECOMMENDATION:")
        print("-" * 70)
        
        if missing_percent < 5:
            print("✓ Missing < 5%")
            print("  → Use: DROP ROWS or MEAN IMPUTATION")
            print("  → Reason: Simple methods work well with little missing data")
            recommended = 'drop_rows'
            
        elif missing_percent < 20 and has_date:
            print("✓ Missing 5-20% with time series")
            print("  → Use: INTERPOLATION or FORWARD FILL")
            print("  → Reason: Time series methods preserve temporal patterns")
            recommended = 'interpolation'
            
        elif missing_percent < 30:
            print("✓ Missing 20-30%")
            print("  → Use: KNN IMPUTATION")
            print("  → Reason: Uses similar samples for accurate imputation")
            recommended = 'knn'
            
        elif missing_percent < 50:
            print("⚠ Missing 30-50%")
            print("  → Use: ITERATIVE IMPUTATION (MICE)")
            print("  → Reason: Most sophisticated for high missing rates")
            recommended = 'iterative'
            
        else:
            print("⚠ Missing > 50%")
            print("  → Use: DROP COLUMNS or COLLECT MORE DATA")
            print("  → Reason: Too much missing data for reliable imputation")
            recommended = 'drop_columns'
        
        return recommended


def demonstrate_all_methods():
    """
    Demonstrate all missing value handling methods
    """
    print("\n")
    print("╔" + "="*68 + "╗")
    print("║" + " "*15 + "MISSING VALUE HANDLING DEMONSTRATION" + " "*17 + "║")
    print("╚" + "="*68 + "╝")
    
    # Create sample data with missing values
    print("\nGenerating sample AQI data with missing values...")
    
    np.random.seed(42)
    n_samples = 1000
    
    df = pd.DataFrame({
        'date': pd.date_range(start='2023-01-01', periods=n_samples, freq='H'),
        'pm25': np.random.normal(50, 20, n_samples),
        'pm10': np.random.normal(75, 25, n_samples),
        'no2': np.random.normal(40, 15, n_samples),
        'so2': np.random.normal(20, 10, n_samples),
        'o3': np.random.normal(60, 20, n_samples),
        'co': np.random.normal(1.5, 0.5, n_samples),
        'aqi': np.random.normal(100, 30, n_samples)
    })
    
    # Introduce missing values randomly (20%)
    missing_mask = np.random.random(df.shape) < 0.15
    df = df.mask(missing_mask)
    
    print(f"✓ Created dataset with {df.isnull().sum().sum()} missing values")
    
    # Initialize handler
    handler = MissingValueHandler()
    
    # Step 1: Analyze
    handler.analyze_missing_values(df)
    
    # Step 2: Visualize
    handler.visualize_missing_values(df)
    
    # Step 3: Get recommendation
    recommended = handler.recommend_method(df)
    
    # Step 4: Compare methods
    handler.compare_methods(df)
    
    # Step 5: Apply recommended method
    print("\n" + "="*70)
    print(f"APPLYING RECOMMENDED METHOD: {recommended.upper()}")
    print("="*70)
    
    if recommended == 'interpolation':
        df_cleaned = handler.method_5_interpolation(df)
    elif recommended == 'knn':
        df_cleaned = handler.method_6_knn_imputation(df)
    elif recommended == 'iterative':
        df_cleaned = handler.method_7_iterative_imputation(df)
    else:
        df_cleaned = handler.method_1_drop_rows(df)
    
    # Save cleaned data
    df_cleaned.to_csv('/home/claude/aqi_data_cleaned.csv', index=False)
    print(f"\n✓ Cleaned data saved to: aqi_data_cleaned.csv")
    
    print("\n" + "="*70)
    print("PROCESS COMPLETED!")
    print("="*70)
    print("\nFiles generated:")
    print("  1. aqi_data_cleaned.csv - Cleaned dataset")
    print("  2. missing_values_viz.png - Visualizations")
    
    return df_cleaned


if __name__ == "__main__":
    demonstrate_all_methods()
