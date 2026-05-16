from flask import Flask, render_template, request, jsonify
import joblib

app = Flask(__name__)

# Load trained model and encoders
model = joblib.load("restaurant_model.pkl")
le_day = joblib.load("day_encoder.pkl")
le_dish = joblib.load("dish_encoder.pkl")


# All dishes from new dataset
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


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    try:

        # Receive frontend data
        data = request.get_json()

        print("Received Data:", data)

        day = data["day"]
        temperature = int(data["temperature"])
        event = int(data["event"])

        # Encode day
        day_encoded = le_day.transform([day])[0]

        results = {}

        # Predict all dishes
        for dish in DISHES:

            # Skip if dish not in encoder
            if dish not in le_dish.classes_:
                continue

            dish_encoded = le_dish.transform([dish])[0]

            prediction = model.predict([
                [day_encoded, temperature, event, dish_encoded]
            ])

            results[dish] = round(prediction[0])

        # Sort predictions highest to lowest
        results = dict(
            sorted(
                results.items(),
                key=lambda item: item[1],
                reverse=True
            )
        )

        print("Predictions:", results)

        return jsonify(results)

    except Exception as e:

        print("ERROR:", e)

        return jsonify({
            "error": str(e)
        })


if __name__ == "__main__":
    app.run(debug=True)