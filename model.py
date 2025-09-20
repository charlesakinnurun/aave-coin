# %% [markdown]
# Import the neccessary libraries

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression,Ridge,Lasso,ElasticNet
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
import matplotlib.pyplot as plt

# %% [markdown]
# Data Loading

# %%
try:
    df = pd.read_csv("coin_Aave.csv")
    print("Data loaded successfully!")
except FileNotFoundError:
    print("Error: The file 'coin_Aave.csv' was not found. Please make sure it is in the same directory")
    exit()
df.head()

# %% [markdown]
# Data Preprocessing

# %%
# Check for missing value
df_missing = df.isnull().sum()
print("Missing Values")
print(df_missing)

# Check for duplicated rows
df_duplicated = df.duplicated().sum()
print("Duplicated Rows")
print(df_duplicated)

# Rename the columns for clarity and consistency
df.rename(columns={
    "SNo":"serial_number",
    "Name":"name",
    "Symbol":"symbol",
    "Date":"date",
    "High":"high",
    "Low":"low",
    "Open":"open",
    "Close":"close",
    "Volume":"volume",
    "Marketcap":"marketcap"
},inplace=True)

# Convert the data to a simple numerical ordinal value for the model to use
df["date"] = pd.to_datetime(df["date"]).apply(lambda x: x.toordinal())

# %% [markdown]
# Feature Engineering

# %%
# We define our features (X) and our target (y)
# The features are the columns we will use to make predictions
# The target is the column we want to predict ("close")

features = ["high","low","open","volume"]
target = "close"

# Define the feature (X) and the target (y)
b = df["high"]

X = df[features]
y = df[target]

# %% [markdown]
# Data Splitting

# %%
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

# %% [markdown]
# Data Scaling

# %%
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.fit_transform(X_test)

# %% [markdown]
# Model Training

# %%
# We will create a dictionaty to store our models and their names

models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(),
    "Lasso Regression": Lasso(alpha=0.1), # alpha is a regularization paramet
    "ElasticNet Regression":ElasticNet(alpha=0.1,l1_ratio=0.5),
    "SVR": SVR(kernel="rbf") # SVR with a radical basis function kernel
}

# We'll store the results for comparison
results = {}

# %% [markdown]
# Model Evaluation

# %%
print("-----Model Evaluation-----")
for name, model in models.items():
    print(f"Training {name}....")
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    # Calculate evaluation metrics
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Store the Results
    results[name] = {"MAE": mae, "MSE": mse, "R2": r2}

    # Print the Results
    print(f"Mean Absolute Error (MAE): {mae:.2f}")
    print(f"Mean Squared Error (MSE): {mse:.2f}")
    print(f"R-Squared: {r2:.2f}")

# %% [markdown]
# Find the Best Model

# %%
# We find the best model based on the R-squared value, which indicates how
# well the model explains the variance in the data (closer to 1 is better)

best_model_name = ""
best_r2 = -np.inf

for name,metrics in results.items():
    if metrics["R2"] > best_r2:
        best_r2 = metrics["R2"]
        best_model_name = name

print("Model Comparison Summary")
print(f"The best performing model is {best_model_name}")
print(f"It achieved an R-squared score of {best_r2:.2f}")

# %% [markdown]
# Visualization of Results

# %%
fig,ax = plt.subplots(figsize=(10,6))
model_names = list(results.keys())
r2_scores = [results[name]["R2"] for name in model_names]
plt.bar(model_names,r2_scores,color="red")
plt.ylabel("$R^2$ Score")
plt.title("R-Squared Score Comparison of Regression Models")
plt.ylim(0,1) # R2 Score is between 0 and 1
plt.xticks(rotation=45,ha="right")
plt.tight_layout()
plt.show()

# %% [markdown]
# Use the Best Model for Prediction based on User Input

# %%
# This part of the code prompts the user to input values for all features to make a prediction
print("Interactive Prediction using Ridge Regression")
print("Enter the following values to predict the Close price of Aave Coin")

while True:
    try:
        high_price_input = input("Enter the High Price (or type 'exit' to quit):")
        if high_price_input == "exit":
            break
        high_price = float(high_price_input)
        low_price = float(input("Enter the Low Price:"))
        open_price = float(input("Enter the Open Price:"))
        volume_price = float(input("Enter the Trading Volume:"))

        # We must reshape the input to a 2D array for a single sample
        new_prices = np.array([[high_price,low_price,open_price,volume_price]])

        predicted_new_prices = models["Ridge Regression"].predict(new_prices)

        print(f"For the given prices, the predicted Close price is:{predicted_new_prices[0]:.5f}")
    except ValueError:
        print("Invalid input. Please enter valid numbers")


