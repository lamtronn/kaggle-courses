import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
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

# Create model 
mobile_battery_model = RandomForestRegressor(random_state=1)

# Fir the model 
mobile_battery_model.fit(train_X, train_y)

# Make predictation
mobile_battery_preds = mobile_battery_model.predict(val_X)

# Calculate the MAE value 
print(mean_absolute_error(val_y, mobile_battery_preds))





