import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
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

y = X.Height

# Break off validation set from training data
X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2,
                                                      random_state=0)

is_catagorical_cols = (X_train.dtypes == "object")
catagorical_cols = list(is_catagorical_cols[is_catagorical_cols].index)


# Approach 1: Drop catagorical cols 
# dropped_X_train = X_train.drop(catagorical_cols, axis = 1)
# dropped_X_valid = X_valid.drop(catagorical_cols, axis = 1)
# mae_score = score_dataset(dropped_X_train, dropped_X_valid, y_train, y_valid)
# print("Approach 1 - Drop catagorical cols: ", mae_score)

# Approach 2: Cordinal Encoding
# label_X_train = X_train.copy()
# label_X_valid = X_valid.copy()
# ordinal_encoder = OrdinalEncoder()
# label_X_train = ordinal_encoder.fit_transform(X_train[catagorical_cols])
# label_X_valid = ordinal_encoder.fit_transform(X_valid[catagorical_cols])
# mae_score = score_dataset(label_X_train, label_X_valid, y_train, y_valid)
# print("Approach 2 - Cordinal Encoding: ", mae_score)

# Approach 3: One-hot Encoding
OH_encoder = OneHotEncoder(handle_unknown='ignore')
OH_X_train = pd.DataFrame(OH_encoder.fit_transform(X_train[catagorical_cols]))
OH_X_valid = pd.DataFrame(OH_encoder.transform(X_valid[catagorical_cols]))

# One-hot encoding removed index; put it back
OH_X_train.index = X_train.index
OH_X_valid.index = X_valid.index

# Remove categorical columns (will replace with one-hot encoding)
num_X_train = X_train.drop(catagorical_cols, axis = 1)
num_X_valid = X_valid.drop(catagorical_cols, axis = 1)

# Add one-hot encoded columns to numerical features
OH_X_train = pd.concat([num_X_train, OH_X_train], axis = 1)
OH_X_valid = pd.concat([num_X_valid, OH_X_valid], axis = 1)

# Ensure all columns have string type
OH_X_train.columns = OH_X_train.columns.astype(str)
OH_X_valid.columns = OH_X_valid.columns.astype(str)

mae_score = score_dataset(OH_X_train, OH_X_valid, y_train, y_valid)

print("Approach 3 - One-hot Encoding: ", mae_score)





