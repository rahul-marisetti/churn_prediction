from flask import Flask, render_template, request
import pandas as pd
import pickle

app = Flask(__name__)

# Load Model and Label Encoders When App Starts
with open("model3.pkl", "rb") as file:
    model = pickle.load(file)

with open("label_encoders3.pkl", "rb") as file:
    label_encoders = pickle.load(file)

@app.route('/')
def index():
    return render_template("index (1).html")

@app.route('/form')
def form():
    return render_template("form.html")

@app.route('/submit', methods=['POST'])
def submit():
    if request.method == 'POST':
        try:
            # Get user input
            user_data = {
                "Age": float(request.form.get("age")),
                "Login_Frequency": float(request.form.get("login")),
                "Location": request.form.get("location"),
                "Customer_Rating": float(request.form.get("rating")),
                "Subscription_Duration": float(request.form.get("subscription")),
                "Activity_Time": float(request.form.get("activity"))
            }

            # Encode Location Properly
            if user_data["Location"] in label_encoders["Location"].classes_:
                user_data["Location"] = label_encoders["Location"].transform([user_data["Location"]])[0]
            else:
                user_data["Location"] = -1  # Handle unknown locations

            # Convert to DataFrame
            input_df = pd.DataFrame([user_data])

            # Ensure all expected columns are present
            expected_features = list(model.feature_names_in_)
            for col in expected_features:
                if col not in input_df.columns:
                    input_df[col] = 0  # Fill missing features with default value

            # Ensure column order matches training data
            input_df = input_df[expected_features]

            # Debugging: Print feature names
            print("Model Features:", model.feature_names_in_)
            print("Input Features:", input_df.columns)

            # Make Prediction
            prediction = model.predict(input_df)[0]

            # Display Result
            result = "Likely to Churn" if prediction == 1 else "Not Likely to Churn"
            return render_template("result.html", result=result)

        except Exception as e:
            return f"Error: {str(e)}"

if __name__ == '__main__':
    app.run(debug=True, port=4000)
