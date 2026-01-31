# Complete Guide: Handling Missing Values in AQI Data

## 📊 Why Missing Values Matter

In real-world air quality data, missing values are **extremely common** due to:
- Sensor malfunctions
- Power outages  
- Network connectivity issues
- Maintenance periods
- Data transmission errors

**Typical missing rates in AQI datasets: 10-30%**

---

## 🎯 7 Methods to Handle Missing Values

### Method 1: Drop Rows ❌
**When to use:** Very few missing values (<5%)

**Pros:**
- Simple and fast
- No assumptions about data

**Cons:**
- Loses data
- Not suitable for time series

**Code:**
```python
# Drop all rows with any missing values
df_cleaned = df.dropna()

# Drop rows with >50% missing
df_cleaned = df.dropna(thresh=int(0.5 * len(df.columns)))
```

**Resume bullet:**
> "Removed rows with missing values using dropna(), ensuring data quality for model training"

---

### Method 2: Drop Columns ❌
**When to use:** Specific columns have >50% missing

**Pros:**
- Preserves most data
- Removes unreliable features

**Cons:**
- Loses potentially useful information

**Code:**
```python
# Drop columns with >30% missing
threshold = 0.3
missing_pct = df.isnull().sum() / len(df)
cols_to_drop = missing_pct[missing_pct > threshold].index
df_cleaned = df.drop(columns=cols_to_drop)
```

**Resume bullet:**
> "Analyzed feature quality and removed columns with >30% missing data to improve model reliability"

---

### Method 3: Forward Fill / Backward Fill ✅
**When to use:** Time series data with <20% missing

**Pros:**
- Preserves temporal patterns
- Fast and simple
- Works well for AQI (smooth changes)

**Cons:**
- Can propagate errors
- Not good for large gaps

**Code:**
```python
# Forward fill (use previous value)
df['pm25'] = df['pm25'].fillna(method='ffill')

# Backward fill (use next value)
df['pm25'] = df['pm25'].fillna(method='bfill')

# Combined approach
df = df.fillna(method='ffill').fillna(method='bfill')
```

**Resume bullet:**
> "Applied forward-fill imputation for time-series AQI data, preserving temporal continuity"

---

### Method 4: Mean/Median Imputation ✅
**When to use:** <15% missing, no time dependency

**Pros:**
- Simple and fast
- Doesn't change data distribution
- Median is robust to outliers

**Cons:**
- Reduces variance
- Ignores relationships

**Code:**
```python
# Fill with mean
df['pm25'] = df['pm25'].fillna(df['pm25'].mean())

# Fill with median (better for outliers)
df['pm25'] = df['pm25'].fillna(df['pm25'].median())

# Using sklearn
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='median')
df[['pm25', 'pm10']] = imputer.fit_transform(df[['pm25', 'pm10']])
```

**Resume bullet:**
> "Implemented median imputation for missing pollutant values, maintaining statistical distribution"

---

### Method 5: Interpolation ⭐ RECOMMENDED
**When to use:** Time series with <30% missing

**Pros:**
- Smooth, realistic values
- Uses surrounding points
- Great for AQI trends

**Cons:**
- Requires ordered data
- Can be slow for large datasets

**Code:**
```python
# Linear interpolation
df['pm25'] = df['pm25'].interpolate(method='linear')

# Polynomial interpolation (smoother)
df['pm25'] = df['pm25'].interpolate(method='polynomial', order=2)

# Time-based interpolation
df['date'] = pd.to_datetime(df['date'])
df = df.set_index('date')
df['pm25'] = df['pm25'].interpolate(method='time')
```

**Resume bullet:**
> "Applied linear interpolation to estimate missing AQI values, leveraging temporal continuity in pollution data"

---

### Method 6: KNN Imputation ⭐⭐ RECOMMENDED
**When to use:** 15-30% missing, multiple correlated features

**Pros:**
- Uses similar samples
- Considers feature relationships
- Very accurate

**Cons:**
- Computationally expensive
- Requires complete cases

**Code:**
```python
from sklearn.impute import KNNImputer

# Use 5 nearest neighbors
imputer = KNNImputer(n_neighbors=5)
df[['pm25', 'pm10', 'no2']] = imputer.fit_transform(df[['pm25', 'pm10', 'no2']])
```

**Resume bullet:**
> "Utilized KNN imputation (k=5) to fill missing values by leveraging correlations between pollutants, achieving 95% data retention"

---

### Method 7: Iterative Imputation (MICE) ⭐⭐⭐ ADVANCED
**When to use:** Complex patterns, >20% missing

**Pros:**
- Most sophisticated
- Models each feature
- Handles complex relationships

**Cons:**
- Computationally intensive
- Can overfit

**Code:**
```python
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

# MICE algorithm
imputer = IterativeImputer(max_iter=10, random_state=42)
df[['pm25', 'pm10', 'no2']] = imputer.fit_transform(df[['pm25', 'pm10', 'no2']])
```

**Resume bullet:**
> "Implemented MICE (Multivariate Imputation by Chained Equations) algorithm for sophisticated missing value estimation in multi-pollutant dataset"

---

## 📋 Decision Tree: Which Method to Use?

```
Is missing < 5%?
├─ YES → Drop Rows
└─ NO → Is it time series data?
    ├─ YES → Missing < 20%?
    │   ├─ YES → Use Interpolation or Forward Fill
    │   └─ NO → Use KNN or MICE
    └─ NO → Missing < 15%?
        ├─ YES → Use Mean/Median
        └─ NO → Use KNN or MICE
```

---

## 💡 Best Practices for Your AQI Project

### 1. Always Analyze First
```python
# Check missing patterns
print(df.isnull().sum())
print((df.isnull().sum() / len(df)) * 100)

# Visualize
import seaborn as sns
sns.heatmap(df.isnull(), cbar=True)
```

### 2. Document Your Choice
In your report, explain:
- How much data was missing
- Why you chose this method
- What impact it had

### 3. Compare Methods
```python
# Try multiple methods and compare
methods = {
    'Mean': df.fillna(df.mean()),
    'Interpolation': df.interpolate(),
    'KNN': KNNImputer().fit_transform(df)
}

# Train models on each and compare RMSE
```

### 4. Add to Resume
Pick the most impressive method you used:

**Weak:** ❌
> "Handled missing values in the dataset"

**Strong:** ✅
> "Implemented KNN imputation (k=5) to handle 18% missing values across 6 pollutant features, achieving 95% data retention while preserving inter-pollutant correlations (PM2.5-PM10: r=0.89)"

---

## 🎓 For Different Project Levels

### Undergraduate / Basic Project:
```python
# Simple but effective
df = df.fillna(df.median())  # Method 4
```
**Resume:** "Applied median imputation for missing values"

### Graduate / Intermediate Project:
```python
# More sophisticated
df = df.interpolate(method='linear')  # Method 5
```
**Resume:** "Utilized linear interpolation to preserve temporal patterns in time-series AQI data"

### Advanced / Research Project:
```python
# State-of-the-art
from sklearn.impute import KNNImputer
imputer = KNNImputer(n_neighbors=5)
df_imputed = imputer.fit_transform(df)  # Method 6
```
**Resume:** "Implemented KNN-based imputation algorithm, leveraging pollutant correlations for accurate missing value estimation"

---

## 📊 Example: Complete Workflow

```python
import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer

# 1. Load data
df = pd.read_csv('aqi_data.csv')

# 2. Analyze missing values
print("Missing value analysis:")
print("-" * 50)
for col in df.columns:
    missing = df[col].isnull().sum()
    pct = (missing / len(df)) * 100
    if missing > 0:
        print(f"{col:10s}: {missing:5d} ({pct:5.1f}%)")

# 3. Decide on method
total_missing_pct = (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
print(f"\nTotal missing: {total_missing_pct:.1f}%")

if total_missing_pct < 5:
    print("Using: Drop rows")
    df_clean = df.dropna()
elif total_missing_pct < 20:
    print("Using: Interpolation")
    df_clean = df.interpolate(method='linear')
else:
    print("Using: KNN Imputation")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    imputer = KNNImputer(n_neighbors=5)
    df_clean = df.copy()
    df_clean[numeric_cols] = imputer.fit_transform(df[numeric_cols])

# 4. Verify
print(f"\nBefore: {df.isnull().sum().sum()} missing values")
print(f"After:  {df_clean.isnull().sum().sum()} missing values")

# 5. Save cleaned data
df_clean.to_csv('aqi_data_cleaned.csv', index=False)
print("\n✓ Cleaned data saved!")
```

---

## 🚀 Quick Start for Your Project

### Option 1: Use the Integrated Script
```bash
python aqi_with_nan_handling.py
```
This automatically:
- Analyzes missing values
- Selects best method
- Handles missing data
- Trains models
- Saves cleaned data

### Option 2: Manual Control
```python
from handle_missing_values import MissingValueHandler

handler = MissingValueHandler()

# Analyze
handler.analyze_missing_values(df)

# Visualize
handler.visualize_missing_values(df)

# Get recommendation
recommended = handler.recommend_method(df)

# Apply method
if recommended == 'interpolation':
    df_clean = handler.method_5_interpolation(df)
elif recommended == 'knn':
    df_clean = handler.method_6_knn_imputation(df)
```

---

## 📝 What to Include in Your Report

### Section: Data Preprocessing

**1. Missing Value Analysis**
```
We identified missing values in the dataset as follows:
- PM2.5: 324 values (15.2%)
- PM10: 289 values (13.6%)
- NO2: 198 values (9.3%)
Total missing: 12.4% of all values
```

**2. Method Selection**
```
Given the temporal nature of air quality data and 
missing rate of 12.4%, we selected Linear Interpolation 
as our imputation strategy. This method preserves the 
smooth, continuous nature of pollution trends while 
avoiding the bias introduced by simple mean imputation.
```

**3. Implementation**
```python
df['pm25'] = df['pm25'].interpolate(method='linear', 
                                     limit_direction='both')
```

**4. Validation**
```
After imputation:
- Remaining missing: 0 values
- Data retention: 100%
- Verified no outliers introduced
```

---

## 🎯 Interview Questions & Answers

**Q: How did you handle missing values?**
**A:** "I analyzed the dataset and found 12% missing values across pollutant features. Given the time-series nature of AQI data, I used linear interpolation which estimates missing values based on surrounding time points. This preserved the temporal continuity while maintaining 100% data retention. I validated the approach by checking that no outliers were introduced."

**Q: Why not just drop missing rows?**
**A:** "Dropping rows would have removed 35% of our data, significantly reducing our training set and potentially introducing temporal gaps. Since AQI changes smoothly over time, interpolation provides more accurate estimates than losing this valuable data."

**Q: Did you try other methods?**
**A:** "Yes, I compared forward-fill, mean imputation, and KNN imputation. Interpolation achieved the lowest RMSE (15.2) compared to forward-fill (18.3) and mean imputation (21.1) when validated on a hold-out set."

---

## ✅ Checklist: Before Submitting

- [ ] Analyzed missing value patterns
- [ ] Documented missing percentages
- [ ] Selected appropriate method with justification
- [ ] Implemented chosen method
- [ ] Validated results (no outliers/errors)
- [ ] Updated resume bullets
- [ ] Added section to project report
- [ ] Saved cleaned dataset
- [ ] Can explain choice in interview

---

## 📚 Further Reading

- Scikit-learn Imputation Guide: https://scikit-learn.org/stable/modules/impute.html
- Pandas Missing Data: https://pandas.pydata.org/docs/user_guide/missing_data.html
- MICE Algorithm Paper: https://www.jstatsoft.org/article/view/v045i03

---

**Remember:** Handling missing values properly shows:
✓ Data science maturity
✓ Understanding of real-world challenges
✓ Attention to data quality
✓ Knowledge of advanced techniques

This can be a **major differentiator** in your project presentation!
