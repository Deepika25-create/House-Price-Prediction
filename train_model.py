# Import pandas to load dataset
import pandas as pd

# Import Linear Regression model
from sklearn.linear_model import LinearRegression

# Import joblib to save trained model
import joblib

# Load dataset from CSV file
data = pd.read_csv("house_data.csv")

# Separate input features
X = data[['Size','Bedrooms','Age']]

# Target variable
y = data['Price']

# Create Linear Regression model
model = LinearRegression()

# Train model using dataset
model.fit(X,y)

# Save trained model
joblib.dump(model,"house_model.pkl")

print("Model trained successfully and saved as house_model.pkl")