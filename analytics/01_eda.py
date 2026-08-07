import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Load Titanic dataset ONCE
df = sns.load_dataset("titanic")

# Save offline fallback
df.to_csv("titanic.csv", index=False)

print("Shape:")
print(df.shape)

print("\nInfo:")
print(df.info())

print("\nDescribe:")
print(df.describe())

print("\nMissing percentage:")
missing = df.isnull().sum()
missing_percent = (missing / len(df)) * 100
print(missing_percent[missing_percent > 0])


# ---------------- CLEANING ----------------

clean_df = df.copy()

# Missing value handling
# age, embarked are between 5%-30% -> impute
clean_df["age"] = clean_df["age"].fillna(clean_df["age"].median())
clean_df["embarked"] = clean_df["embarked"].fillna(clean_df["embarked"].mode()[0])

# deck has very high missing values -> drop
clean_df = clean_df.drop(columns=["deck"])


# Save cleaned data
clean_df.to_csv("clean_titanic.csv", index=False)


# ---------------- OUTLIERS ----------------

for col in ["age", "fare"]:
    q1 = clean_df[col].quantile(0.25)
    q3 = clean_df[col].quantile(0.75)
    iqr = q3 - q1

    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    outliers = clean_df[(clean_df[col] < lower) | (clean_df[col] > upper)]

    print(f"{col} outliers:", len(outliers))


# Fare statistics

print("\nFare mean:")
print(clean_df["fare"].mean())

print("Fare median:")
print(clean_df["fare"].median())

print("Fare mode:")
print(clean_df["fare"].mode()[0])


# ---------------- UNIVARIATE PLOTS ----------------

for col in ["age", "fare"]:
    sns.histplot(clean_df[col], kde=True)
    plt.title(f"{col} Histogram")
    plt.savefig(f"{col}_hist.png")
    plt.close()

    sns.boxplot(x=clean_df[col])
    plt.title(f"{col} Boxplot")
    plt.savefig(f"{col}_box.png")
    plt.close()


# ---------------- SURVIVAL ANALYSIS ----------------

print("\nSurvival by sex")
print(clean_df.groupby("sex")["survived"].mean())

print("\nSurvival by class")
print(clean_df.groupby("pclass")["survived"].mean())

print("\nSurvival by sex and class")
print(clean_df.groupby(["sex","pclass"])["survived"].mean())


# ---------------- CORRELATION ----------------

corr_cols = [
    "survived",
    "pclass",
    "age",
    "sibsp",
    "parch",
    "fare"
]

corr = clean_df[corr_cols].corr()

print("\nCorrelation:")
print(corr)

sns.heatmap(corr, annot=True, cmap="coolwarm")
plt.title("Titanic Correlation Heatmap")
plt.savefig("correlation_heatmap.png")
plt.close()


# ---------------- STANDARDIZATION CHECK ----------------

for col in ["age","fare"]:
    clean_df[col+"_z"] = (
        clean_df[col] - clean_df[col].mean()
    ) / clean_df[col].std()

print("\nStandardization check")

print(clean_df[["age_z","fare_z"]].mean())
print(clean_df[["age_z","fare_z"]].std())


print("\nEDA COMPLETE")