import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score
import joblib


print("\n========================================")
print("   SMART RESTAURANT AI TRAINING SYSTEM")
print("========================================\n")


# Load Dataset
df = pd.read_csv("restaurant_dataset.csv")

print("Dataset Loaded Successfully!")
print(f"Total Records : {len(df)}")
print(f"Total Columns : {len(df.columns)}\n")


# Check Missing Values
print("Checking Missing Values...\n")
print(df.isnull().sum())


# Label Encoding
le_day = LabelEncoder()
le_dish = LabelEncoder()

df["Day"] = le_day.fit_transform(df["Day"])
df["Dish"] = le_dish.fit_transform(df["Dish"])


# Features and Target
X = df[["Day", "Temperature", "Event", "Dish"]]
y = df["Quantity"]


# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


print("\nTraining Dataset :", len(X_train))
print("Testing Dataset  :", len(X_test))


# Create Model
model = RandomForestRegressor(
    n_estimators=500,
    random_state=42
)


print("\nTraining AI Model...\n")


# Train Model
model.fit(X_train, y_train)


# Predictions
predictions = model.predict(X_test)


# Accuracy Score
score = r2_score(y_test, predictions)


print("========================================")
print("         MODEL PERFORMANCE")
print("========================================")

print(f"R2 Score : {round(score, 3)}")

if score >= 0.90:
    print("Model Accuracy Status : Excellent")
elif score >= 0.75:
    print("Model Accuracy Status : Good")
elif score >= 0.60:
    print("Model Accuracy Status : Average")
else:
    print("Model Accuracy Status : Needs Improvement")


# Save Model & Encoders
joblib.dump(model, "restaurant_model.pkl")
joblib.dump(le_day, "day_encoder.pkl")
joblib.dump(le_dish, "dish_encoder.pkl")


print("\n========================================")
print("      FILES SAVED SUCCESSFULLY")
print("========================================")

print("restaurant_model.pkl")
print("day_encoder.pkl")
print("dish_encoder.pkl")


# Show Available Dishes
print("\nAvailable Dishes In Dataset:\n")

for dish in le_dish.classes_:
    print("-", dish)


# Sample Prediction Test
sample = [[0, 35, 1, 0]]

sample_prediction = model.predict(sample)

print("\n========================================")
print("        SAMPLE PREDICTION TEST")
print("========================================")

print("Sample Input : [Day=0, Temp=35, Event=1, Dish=0]")
print("Predicted Quantity :", round(sample_prediction[0]))


print("\nAI Training Completed Successfully!")