# Customer-Churn-Prediction
A machine learning project that predicts which telecom customers are likely to cancel their subscription.  This project simulates a real-world business client  request to identify at-risk customers before they leave  so the company can take action and retain them.





## 🧑‍💼 Business Problem
A telecom company is losing customers every month and 
does not know WHO is about to leave until it is too late. 
They need a data scientist to analyze customer data, 
find out why customers are leaving, and build a model 
that predicts which customers will cancel so they can 
reach out with retention offers before they go.

---

## 📁 Dataset
- **Rows:** 7,043 customers
- **Columns:** 21 features
- **After cleaning:** 7,032 rows, 24 columns

| Column | Description |
|--------|-------------|
| customerID | Unique customer identifier (dropped) |
| gender | Male or Female |
| SeniorCitizen | 1=Senior, 0=Not Senior |
| Partner | Has a partner? Yes/No |
| Dependents | Has dependents? Yes/No |
| tenure | Months with the company |
| PhoneService | Has phone service? Yes/No |
| MultipleLines | Has multiple lines? Yes/No |
| InternetService | DSL, Fiber optic, or No |
| OnlineSecurity | Has online security? Yes/No |
| OnlineBackup | Has online backup? Yes/No |
| DeviceProtection | Has device protection? Yes/No |
| TechSupport | Has tech support? Yes/No |
| StreamingTV | Has streaming TV? Yes/No |
| StreamingMovies | Has streaming movies? Yes/No |
| Contract | Month-to-month, One year, Two year |
| PaperlessBilling | Has paperless billing? Yes/No |
| PaymentMethod | How customer pays |
| MonthlyCharges | Monthly payment amount |
| TotalCharges | Total amount paid |
| Churn | Did customer leave? Yes=1, No=0 |

---

## 🛠️ Tools & Libraries

| Tool | Purpose |
|------|---------|
| Python | Main programming language |
| Pandas | Data loading and manipulation |
| Matplotlib | Data visualization |
| Seaborn | Advanced data visualization |
| Scikit-learn | Machine learning models |
| Imbalanced-learn | SMOTE for balancing data |
| XGBoost | Advanced ML model |
| Google Colab | Development environment |

---

## 📊 Project Steps

### Step 1 — Load & Explore Data

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')
print(df.shape)
df.head()
df.info()
df.describe()
df['Churn'].value_counts()
```

**Findings:**
- 7,043 customer records with 21 columns
- 18 columns were object (text) type — needed encoding
- TotalCharges column had 11 blank rows
- 5,174 customers stayed (73%)
- 1,869 customers churned (27%)
- Dataset is imbalanced — needed SMOTE to fix

---

### Step 2 — Data Cleaning

```python
# Drop customerID
df = df.drop('customerID', axis=1)

# Fix TotalCharges column
df['TotalCharges'] = pd.to_numeric(
    df['TotalCharges'], errors='coerce')
df = df.dropna()

# Convert Churn to 0/1
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

# Convert gender
df['gender'] = df['gender'].map({'Female': 0, 'Male': 1})

# Convert all Yes/No columns
yes_no_columns = ['Partner', 'Dependents', 'PhoneService',
                  'MultipleLines', 'OnlineSecurity',
                  'OnlineBackup', 'DeviceProtection',
                  'TechSupport', 'StreamingTV',
                  'StreamingMovies', 'PaperlessBilling']

for col in yes_no_columns:
    df[col] = df[col].map({'Yes': 1, 'No': 0,
                           'No internet service': 0,
                           'No phone service': 0})

# One Hot Encoding for 3+ option columns
df = pd.get_dummies(df, columns=['InternetService',
                                  'Contract',
                                  'PaymentMethod'],
                                  drop_first=True)

# Fix True/False to 1/0
df = df.astype(
    {col: int for col in df.select_dtypes(bool).columns})

print("Cleaning done!")
print("Shape:", df.shape)
print("Text remaining:",
      df.select_dtypes('object').columns.tolist())
```

**What was cleaned:**
- Dropped customerID — not useful for prediction
- Fixed 11 blank rows in TotalCharges
- Converted all text columns to numbers
- Applied Label Encoding for Yes/No columns
- Applied One Hot Encoding for Contract, InternetService
  and PaymentMethod
- Final dataset: 7,032 rows, 24 columns, zero text remaining

---

### Step 3 — Data Visualization

```python
# Chart 1 - Churn Count
sns.countplot(x='Churn', data=df)
plt.title('Churn Count (0=Stayed, 1=Left)')
plt.show()

# Chart 2 - Churn by Contract Type
sns.countplot(x='Contract_One year', hue='Churn', data=df)
plt.title('Churn by Contract Type')
plt.xlabel('Contract (0=Month-to-month, 1=One Year)')
plt.legend(['Stayed', 'Left'])
plt.show()

# Chart 3 - Churn by Tenure
sns.histplot(data=df, x='tenure', hue='Churn', bins=30)
plt.title('Churn by Tenure (How long customer stayed)')
plt.xlabel('Tenure (Months)')
plt.show()

# Chart 4 - Monthly Charges vs Churn
sns.boxplot(x='Churn', y='MonthlyCharges', data=df)
plt.title('Monthly Charges vs Churn')
plt.xlabel('Churn (0=Stayed, 1=Left)')
plt.ylabel('Monthly Charges ($)')
plt.show()

# Chart 5 - Churn by Internet Service
sns.countplot(x='InternetService_Fiber optic',
              hue='Churn', data=df)
plt.title('Churn by Internet Service')
plt.xlabel('Fiber Optic (0=No, 1=Yes)')
plt.legend(['Stayed', 'Left'])
plt.show()
```

**Findings:**
- 27% of customers churned (1,869 out of 7,032)
- Month-to-month customers churn 10x more than
  yearly contract customers
- New customers (0-10 months) are at highest risk
- Customers paying above $80/month churn more
- Fiber optic customers churn 3x more than DSL customers

---

### Step 4 — Prepare Data for Modeling

```python
from sklearn.model_selection import train_test_split

# Separate features and target
X = df.drop('Churn', axis=1)
y = df['Churn']

# Split 80% train 20% test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

print("Training size:", X_train.shape)
print("Testing size:", X_test.shape)
```

---

### Step 5 — Handle Imbalanced Data with SMOTE

```python
!pip install imbalanced-learn -q
from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42)
X_train_sm, y_train_sm = smote.fit_resample(
    X_train, y_train)

print("Before SMOTE:", y_train.value_counts().to_dict())
print("After SMOTE:", y_train_sm.value_counts().to_dict())
```

**Results:**
- Before SMOTE: 4,130 stayed vs 1,495 churned
- After SMOTE: 4,130 stayed vs 4,130 churned
- SMOTE created 2,635 synthetic churn examples

---

### Step 6 — Scale the Data

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_sm = scaler.fit_transform(X_train_sm)
X_test = scaler.transform(X_test)
print("Scaling done!")
```

---

### Step 7 — Build 3 Machine Learning Models

```python
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
!pip install xgboost -q
from xgboost import XGBClassifier

# Model 1 - Logistic Regression
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train_sm, y_train_sm)
y_pred_lr = lr.predict(X_test)
lr_score = accuracy_score(y_test, y_pred_lr)
print("Logistic Regression:", round(lr_score * 100, 2), "%")

# Model 2 - Random Forest
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train_sm, y_train_sm)
y_pred_rf = rf.predict(X_test)
rf_score = accuracy_score(y_test, y_pred_rf)
print("Random Forest:", round(rf_score * 100, 2), "%")

# Model 3 - XGBoost
xgb = XGBClassifier(random_state=42, eval_metric='logloss')
xgb.fit(X_train_sm, y_train_sm)
y_pred_xgb = xgb.predict(X_test)
xgb_score = accuracy_score(y_test, y_pred_xgb)
print("XGBoost:", round(xgb_score * 100, 2), "%")
```

---

### Step 8 — Evaluate Best Model

```python
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

# Classification report
print("Classification Report - Random Forest")
print(classification_report(y_test, y_pred_rf))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred_rf)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix - Random Forest')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Feature importance
feat_imp = pd.Series(rf.feature_importances_,
                     index=X.columns)
feat_imp.sort_values().plot(kind='barh',
                            color='steelblue',
                            figsize=(8, 10))
plt.title('Most Important Features for Predicting Churn')
plt.xlabel('Importance Score')
plt.show()
```

---

## 📈 Results

### Model Accuracy Comparison

| Model | Accuracy |
|-------|----------|
| XGBoost | 75.3% |
| Logistic Regression | 76.0% |
| **Random Forest** | **76.0%** 🏆 |

### Best Model — Random Forest Evaluation

| Metric | Class 0 (Stayed) | Class 1 (Churned) |
|--------|-----------------|-------------------|
| Precision | 0.84 | 0.55 |
| Recall | 0.83 | 0.56 |
| F1 Score | 0.84 | 0.55 |
| Support | 1,033 | 374 |
| **Overall Accuracy** | | **76%** |

### Confusion Matrix Results

| | Predicted Stayed | Predicted Churned |
|--|-----------------|-------------------|
| **Actual Stayed** | 861 ✅ | 172 ❌ |
| **Actual Churned** | 165 ❌ | 209 ✅ |

### Error Analysis

| Error Type | Count | Percentage | Business Impact |
|------------|-------|-----------|----------------|
| False Positive | 172 | 12.2% | Wasted retention calls |
| False Negative | 165 | 11.7% | Lost customers — dangerous! |
| Correct predictions | 1,070 | 76% | Accurate |

---

## 🔍 Key Findings

### Top Predictors of Churn

| Rank | Feature | Importance |
|------|---------|-----------|
| 🥇 1st | MonthlyCharges | Highest |
| 🥈 2nd | TotalCharges | Very High |
| 🥉 3rd | tenure | Very High |
| 4th | Contract_Two year | High |
| 5th | Contract_One year | High |
| Last | PhoneService | Lowest |

### Visualization Insights

| Finding | Detail |
|---------|--------|
| Churn rate | 27% of all customers |
| Highest risk group | Month-to-month customers |
| Tenure risk | Customers with 0-10 months |
| Charge risk | Customers paying above $80/month |
| Service risk | Fiber optic churns 3x more than DSL |

---

## 💼 Business Recommendations

1. **Offer discounts to new customers** in the first
   3 months — they are the highest risk group
2. **Encourage month-to-month customers** to upgrade
   to yearly contracts with incentives
3. **Review fiber optic pricing** — too expensive
   relative to perceived value
4. **Create loyalty rewards** for customers over
   2 years to reduce long-term churn
5. **Focus retention calls** on customers paying
   above $80/month with less than 10 months tenure

---

## 📝 Portfolio Description
> *Analyzed 7,032 telecom customer records to predict
> customer churn. Cleaned messy real-world data,
> applied Label Encoding and One Hot Encoding,
> handled class imbalance using SMOTE, and built
> 3 ML models achieving 76% accuracy. Identified
> MonthlyCharges, tenure and contract type as top
> churn predictors. Delivered actionable retention
> recommendations for the business.*

---

## 🏷️ Tags
`Python` `Machine Learning` `Classification`
`Business Analytics` `Random Forest` `XGBoost`
`SMOTE` `Data Cleaning` `EDA` `Scikit-learn`
`Data Visualization` `Imbalanced Data`
