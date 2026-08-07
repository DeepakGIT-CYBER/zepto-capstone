import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from imblearn.over_sampling import SMOTE
import joblib
import matplotlib.pyplot as plt


# Load cleaned data
df = pd.read_csv("clean_titanic.csv")


# ---------------- CLASSIFICATION ----------------

features = [
    "pclass",
    "sex",
    "age",
    "sibsp",
    "parch",
    "fare",
    "embarked"
]

X = df[features]
y = df["survived"]


# Stratified split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


numeric_features = [
    "pclass",
    "age",
    "sibsp",
    "parch",
    "fare"
]

categorical_features = [
    "sex",
    "embarked"
]


preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ]),
            numeric_features
        ),
        (
            "cat",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore"))
            ]),
            categorical_features
        )
    ]
)


models = {
    "Logistic Regression":
        LogisticRegression(max_iter=1000),

    "Decision Tree":
        DecisionTreeClassifier(random_state=42),

    "Random Forest":
        RandomForestClassifier(random_state=42)
}


results = []


for name, model in models.items():

    pipe = Pipeline([
        ("prep", preprocessor),
        ("model", model)
    ])

    pipe.fit(X_train, y_train)

    pred = pipe.predict(X_test)

    prob = pipe.predict_proba(X_test)[:,1]

    results.append([
        name,
        accuracy_score(y_test,pred),
        precision_score(y_test,pred),
        recall_score(y_test,pred),
        f1_score(y_test,pred),
        roc_auc_score(y_test,prob)
    ])

    print("\n",name)

    print(confusion_matrix(y_test,pred))


comparison = pd.DataFrame(
    results,
    columns=[
        "Model",
        "Accuracy",
        "Precision",
        "Recall",
        "F1",
        "AUC"
    ]
)

print("\nMODEL COMPARISON")
print(comparison)


# ---------------- RANDOM FOREST TUNING ----------------


rf_pipe = Pipeline([
    ("prep",preprocessor),
    ("model",
     RandomForestClassifier(
         random_state=42,
         oob_score=True
     ))
])


grid = GridSearchCV(
    rf_pipe,
    {
        "model__n_estimators":[100,200],
        "model__max_depth":[5,10,None],
        "model__max_features":["sqrt","log2"]
    },
    cv=5
)


grid.fit(X_train,y_train)


print("\nBEST PARAMETERS")
print(grid.best_params_)

best_model = grid.best_estimator_

print("OOB SCORE")
print(best_model.named_steps["model"].oob_score_)


joblib.dump(
    best_model,
    "titanic_model_pipeline.pkl"
)


# Reload test

loaded = joblib.load(
    "titanic_model_pipeline.pkl"
)

print(
    "Reload prediction:",
    loaded.predict(X_test.head())
)



# ---------------- REGRESSION ----------------


X_reg = df.drop(columns=["fare"])
y_reg = df["fare"]


X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
    X_reg,
    y_reg,
    test_size=0.2,
    random_state=42
)


reg_numeric = X_reg.select_dtypes(
    include=["int64","float64"]
).columns

reg_cat = X_reg.select_dtypes(
    include=["object"]
).columns


reg_pre = ColumnTransformer(
    [
        (
            "num",
            Pipeline([
                ("imputer",SimpleImputer(strategy="median")),
                ("scale",StandardScaler())
            ]),
            reg_numeric
        ),
        (
            "cat",
            Pipeline([
                ("imputer",SimpleImputer(strategy="most_frequent")),
                ("onehot",OneHotEncoder(handle_unknown="ignore"))
            ]),
            reg_cat
        )
    ]
)


reg_model = Pipeline(
    [
        ("prep",reg_pre),
        ("model",LinearRegression())
    ]
)


reg_model.fit(
    X_train_r,
    y_train_r
)


reg_pred = reg_model.predict(
    X_test_r
)


rmse = np.sqrt(
    mean_squared_error(y_test_r,reg_pred)
)


print("\nREGRESSION")
print("MAE:",mean_absolute_error(y_test_r,reg_pred))
print("RMSE:",rmse)
print("R2:",r2_score(y_test_r,reg_pred))


plt.scatter(
    reg_pred,
    y_test_r-reg_pred
)

plt.xlabel("Predicted")
plt.ylabel("Residuals")

plt.title("Residual Plot")

plt.savefig(
    "residual_plot.png"
)

print("\nMODELING COMPLETE")