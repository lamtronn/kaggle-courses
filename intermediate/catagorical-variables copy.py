import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# Function for comparing different approaches
def score_dataset(X_train, X_valid, y_train, y_valid):
    model = RandomForestRegressor(n_estimators=10, random_state=0)
    model.fit(X_train, y_train)
    preds = model.predict(X_valid)
    return mean_absolute_error(y_valid, preds)

# Import dataset
body_bmi_dataset = './train.csv'

X = pd.read_csv(body_bmi_dataset, sep=';')
print(X.columns)
y = X.Weight

# Break off validation set from training data
X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2,
                                                      random_state=0)

missing_cols = [col for col in X_train.columns if X_train[col].isnull().any()]

reduced_X_train = X_train.drop(missing_cols, axis = 1)
reduced_X_valid = X_valid.drop(missing_cols, axis = 1)

mae_score = score_dataset(reduced_X_train, reduced_X_valid, y_train, y_valid)

print(mae_score)



