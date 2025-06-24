import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import joblib # For saving/loading models
import numpy as np

print("Starting model training script...")

# --- 1. Load the Dataset ---
try:
    df = pd.read_csv('beer-servings.csv')
    print("Dataset loaded successfully.")
except FileNotFoundError:
    print("Error: 'beer-servings.csv' not found. Please ensure it's in the same directory.")
    exit()

# --- 2. Data Preprocessing ---
print("Starting data preprocessing...")

if df.columns[0] == 'Unnamed: 0':
    df = df.drop(columns=[df.columns[0]])

numerical_cols = ['beer_servings', 'spirit_servings', 'wine_servings', 'total_litres_of_pure_alcohol']
for col in numerical_cols:
    if df[col].isnull().any():
        median_val = df[col].median()
        df[col] = df[col].fillna(median_val)
        print(f"Filled missing values in '{col}' with median: {median_val}")

categorical_cols = ['country', 'continent']
for col in categorical_cols:
    if df[col].isnull().any():
        df[col] = df[col].fillna('Unknown')
        print(f"Filled missing values in '{col}' with 'Unknown'.")

df_encoded = pd.get_dummies(df, columns=['country', 'continent'], drop_first=True)
print("Categorical features one-hot encoded.")

X = df_encoded.drop('total_litres_of_pure_alcohol', axis=1)
y = df_encoded['total_litres_of_pure_alcohol']

feature_columns = X.columns.tolist()
print(f"Features for training: {feature_columns}")

# --- 3. Split Data ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Data split into training ({len(X_train)} samples) and testing ({len(X_test)} samples).")

# --- 4. Model Development and Evaluation ---
models = {
    'Linear Regression': LinearRegression(),
    'Random Forest Regressor': RandomForestRegressor(random_state=42)
}

best_model = None
best_r2_score = -np.inf
best_model_name = ""

for name, model in models.items():
    print(f"\n--- Training {name} ---")
    if name == 'Random Forest Regressor':
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_features': ['sqrt', 'log2'],
            'max_depth': [5, 10, None]
        }
        grid_search = GridSearchCV(model, param_grid, cv=3, scoring='r2', n_jobs=-1, verbose=1)
        grid_search.fit(X_train, y_train)
        model = grid_search.best_estimator_
        print(f"Best parameters for {name}: {grid_search.best_params_}")
    else:
        model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    print(f"R2-score for {name} on evaluation data: {r2:.4f}")

    if r2 > best_r2_score:
        best_r2_score = r2
        best_model = model
        best_model_name = name

print(f"\nBest model selected for deployment: {best_model_name} with R2-score: {best_r2_score:.4f}")

# --- 5. Save the Best Model and Feature Columns ---
model_filename = 'best_beer_predictor_model.joblib'
joblib.dump(best_model, model_filename)
print(f"Best model saved as '{model_filename}'")

feature_columns_filename = 'feature_columns.joblib'
joblib.dump(feature_columns, feature_columns_filename)
print(f"Feature columns saved as '{feature_columns_filename}'")

print("Model training script finished successfully.")
