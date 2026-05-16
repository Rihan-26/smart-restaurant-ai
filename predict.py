import joblib

# Load trained model and encoders
model = joblib.load("restaurant_model.pkl")
le_day = joblib.load("day_encoder.pkl")
le_dish = joblib.load("dish_encoder.pkl")


# All dishes from dataset
DISHES = [
    "Biryani",
    "Pizza",
    "Burger",
    "Fried Rice",
    "Pasta",
    "Shawarma",
    "Faluda",
    "Curd Rice",
    "Lemon Rice",
    "Normal Meal",
    "Chicken Wings",
    "Paneer Curry",
    "Ice Cream"
]


# User input
today_day = input("Enter Day: ").title()

# Validate day
if today_day not in le_day.classes_:

    print("\nInvalid Day Entered!")
    print("Available Days:")

    for d in le_day.classes_:
        print("-", d)

    exit()


temperature = int(input("Enter Temperature: "))
event = int(input("Special Event? (1 = Yes, 0 = No): "))


# Encode day
day_encoded = le_day.transform([today_day])[0]

results = {}

print("\nGenerating AI Prediction...\n")


# Predict for all dishes
for dish in DISHES:

    # Skip missing dishes
    if dish not in le_dish.classes_:
        continue

    dish_encoded = le_dish.transform([dish])[0]

    prediction = model.predict([
        [day_encoded, temperature, event, dish_encoded]
    ])

    results[dish] = round(prediction[0])


# Sort highest demand first
sorted_results = sorted(
    results.items(),
    key=lambda x: x[1],
    reverse=True
)


# Final Output
print("=" * 45)
print(" SMART RESTAURANT AI PREPARATION PLAN ")
print("=" * 45)

for dish, qty in sorted_results:

    print(f"{dish:<20} → {qty} Plates")

print("=" * 45)


# Top selling dish
top_dish = sorted_results[0]

print(f"\n🔥 Highest Demand Item: {top_dish[0]}")
print(f"📦 Suggested Quantity : {top_dish[1]} Plates")