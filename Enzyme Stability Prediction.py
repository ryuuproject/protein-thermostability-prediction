# -*- coding: utf-8 -*-


import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from xgboost import XGBRegressor

random_state = 42

def preprocess_data(df, is_test=False):
    if not is_test:
        df[["pH_level", "temperature"]] = df[["temperature", "pH_level"]].where(df["pH_level"] > 14, df[["pH_level", "temperature"]].values)
        df = df[df["temperature"] > 51.5]
    df["sequence_length"] = df["protein_sequence"].apply(len)
    for aa in "ACDEFGHIKLMNPQRSTVWY":
        df[aa] = df["protein_sequence"].str.count(aa)
    drop_columns = ["sequence_id", "source", "protein_sequence", "I"]
    if is_test:
        drop_columns.remove("source")
    df.drop(columns=drop_columns, inplace=True)
    return df

train_file = "/train.csv"
test_file = "/test.csv"

training_data = preprocess_data(pd.read_csv(train_file))
testing_data = preprocess_data(pd.read_csv(test_file), is_test=True)

train_y_values = training_data.pop("temperature").values
train_x_values = training_data.values
test_x_values = testing_data.values

x_train_set, x_test_set, y_train_set, y_test_set = train_test_split(train_x_values, train_y_values, test_size=0.2, random_state=random_state)

# Perform Grid Search
model = XGBRegressor(random_state=random_state)
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 6, 9],
    'learning_rate': [0.01, 0.1, 0.3],
    'subsample': [0.5, 1],
    'colsample_bytree': [0.5, 1],
}

grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
grid_search.fit(x_train_set, y_train_set)

best_params = grid_search.best_params_
print(f"Best parameters found: {best_params}")

best_model = grid_search.best_estimator_
score = best_model.score(x_test_set, y_test_set)
print(f"Best model test score: {score}")