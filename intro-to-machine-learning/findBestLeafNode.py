import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error


# Import dataset
mobile_battery_path = './train.csv'

mobile_battery_data = pd.read_csv(mobile_battery_path)

# Drop missing data
mobile_battery_data = mobile_battery_data.dropna(axis=0)

# Choose target and features 
y = mobile_battery_data.battery_power

mobile_battery_features = ["blue", "dual_sim", "four_g", "m_dep", "mobile_wt", "n_cores"]

X = mobile_battery_data[mobile_battery_features]

# Split dataset into to pieces, training and validation
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=0)

def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(train_X, train_y)
    preds_val = model.predict(val_X)
    mae = mean_absolute_error(val_y, preds_val)
    return(mae)

for max_leaf_nodes in [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 150, 200]:
    my_mae = get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y)
    print("Max leaf nodes: %d  \t\t Mean Absolute Error:  %d" %(max_leaf_nodes, my_mae))
