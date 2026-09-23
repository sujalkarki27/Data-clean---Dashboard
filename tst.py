
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use("ggplot")
sns.set(font_scale=1)

# ============================================
# Load Dataset
# ============================================

df = pd.read_csv("train.csv")

print("="*50)
print("First 5 Rows")
print(df.head())

print("\nDataset Shape:", df.shape)

print("\nDataset Info")
print(df.info())

# ============================================
# Missing Value Handling
# ============================================

print("\nMissing Values Before Cleaning")
print(df.isnull().sum())

# Numerical Columns
num_cols = df.select_dtypes(include=np.number).columns

# Fill numerical values with median
for col in num_cols:
    df[col].fillna(df[col].median(), inplace=True)

# Categorical Columns
cat_cols = df.select_dtypes(include="object").columns

# Fill categorical values with mode
for col in cat_cols:
    df[col].fillna(df[col].mode()[0], inplace=True)

print("\nMissing Values After Cleaning")
print(df.isnull().sum())

# ============================================
# Remove Outliers using IQR
# ============================================

numeric_columns = df.select_dtypes(include=np.number).columns

for col in numeric_columns:

    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)

    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    df = df[(df[col] >= lower) & (df[col] <= upper)]

print("\nShape After Removing Outliers")
print(df.shape)

# ============================================
# Statistical Summary
# ============================================

print("\nSummary Statistics")
print(df.describe())

# ============================================
# Dashboard
# ============================================

fig = plt.figure(figsize=(20,12))

# -----------------------------
# 1. Missing Values Heatmap
# -----------------------------

plt.subplot(2,3,1)
sns.heatmap(df.isnull(), cbar=False, cmap="viridis")
plt.title("Missing Values")

# -----------------------------
# 2. Correlation Heatmap
# -----------------------------

plt.subplot(2,3,2)
corr = df.corr(numeric_only=True)
sns.heatmap(corr,
            annot=True,
            cmap="coolwarm",
            fmt=".2f")
plt.title("Correlation Matrix")

# -----------------------------
# 3. Distribution
# -----------------------------

plt.subplot(2,3,3)

if "Age" in df.columns:
    sns.histplot(df["Age"], kde=True)
    plt.title("Age Distribution")
else:
    sns.histplot(df[num_cols[0]], kde=True)
    plt.title(num_cols[0])

# -----------------------------
# 4. Survival Count
# -----------------------------

plt.subplot(2,3,4)

if "Survived" in df.columns:
    sns.countplot(x="Survived", data=df)
    plt.title("Survival Count")
else:
    sns.countplot(x=df[df.columns[0]])

# -----------------------------
# 5. Boxplot
# -----------------------------

plt.subplot(2,3,5)

if "Fare" in df.columns:
    sns.boxplot(y=df["Fare"])
    plt.title("Fare Distribution")
else:
    sns.boxplot(y=df[num_cols[1]])

# -----------------------------
# 6. Pair Relationship
# -----------------------------

plt.subplot(2,3,6)

if "Sex" in df.columns and "Survived" in df.columns:
    sns.countplot(x="Sex", hue="Survived", data=df)
    plt.title("Gender vs Survival")
else:
    sns.scatterplot(
        x=df[num_cols[0]],
        y=df[num_cols[1]]
    )
    plt.title("Scatter Plot")

plt.tight_layout()
plt.show()

# ============================================
# Additional Visualizations
# ============================================

# Histogram of all numeric columns
df.hist(figsize=(15,10))
plt.suptitle("Numeric Feature Distribution")
plt.show()

# Pairplot
sns.pairplot(df[numeric_columns])
plt.show()

# ============================================
# Key Findings
# ============================================

print("="*60)
print("KEY FINDINGS")
print("="*60)

print(f"Total Rows : {len(df)}")
print(f"Total Columns : {len(df.columns)}")

print("\nAverage Values")

print(df.mean(numeric_only=True))

print("\nMaximum Values")

print(df.max(numeric_only=True))

print("\nMinimum Values")

print(df.min(numeric_only=True))

print("\nStandard Deviation")

print(df.std(numeric_only=True))

print("\nData Cleaning Completed Successfully!")
print("Dashboard Generated Successfully!")