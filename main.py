from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd
from pymongo import MongoClient 

app = Flask(__name__)

# 🔹 Load Trained Model
with open("model_final.pkl", "rb") as file:
    model = pickle.load(file)

# 🔹 Load encoders
with open("encoders.pkl", "rb") as file:
    encoders = pickle.load(file)

# ✅ MongoDB Atlas Connection (Replace <username>, <password>, <cluster-url>)
MONGO_URI = "mongodb+srv://user:54321piyush@cluster0.rpbxt.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
client = MongoClient(MONGO_URI)
db = client["fraud_Database"]  # ✅ Database Name
collection = db["predictions"]  # ✅ Collection Name

@app.route("/")
def home():
    return render_template("index1.html")  # HTML Page Load Karega

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # ✅ User Input Retrieve Karna
        merchant_name = request.form["merchant_name"]
        transaction_amount = float(request.form["transaction_amount"])
        customer_city = request.form["customer_city"]
        customer_job = request.form["customer_job"]
        age = int(request.form["age"])
        customer_state_state_full = request.form["customer_state_state_full"] 
        merchant_category = request.form["merchant_category"]
        transaction_month = int(request.form["transaction_month"])

        # ✅ Categorical Columns Encode Karna
        merchant_name = encoders["merchant_name"].transform([merchant_name])[0]
        customer_city = encoders["customer_city"].transform([customer_city])[0]
        customer_job = encoders["customer_job"].transform([customer_job])[0]
        # ✅ Encode the correct column name
        customer_state_state_full = encoders["customer_state_state_full"].transform([customer_state_state_full])[0]
        merchant_category = encoders["merchant_category"].transform([merchant_category])[0]
        

        # ✅ Final Input Array
        input_features = pd.DataFrame([[merchant_name, transaction_amount, customer_city, 
                                customer_job,age, customer_state_state_full, merchant_category, transaction_month]],
                                columns=['merchant_name', 'transaction_amount', 'customer_city', 
                                       'customer_job','age', 'customer_state_state_full', 'merchant_category', 'transaction_month'])
        # ✅ Prediction Karna
        prediction = model.predict(input_features)
        result = "Fraud" if prediction[0] == 1 else "Not Fraud"

         # ✅ MongoDB Me Data Save Karna
        data_to_store = {
            "merchant_name": request.form["merchant_name"],
            "transaction_amount": transaction_amount,
            "customer_city": request.form["customer_city"],
            "customer_job": request.form["customer_job"],
            "age": age,
            "customer_state_state_full": request.form["customer_state_state_full"],
            "merchant_category": request.form["merchant_category"],
            "transaction_month": transaction_month,
            "prediction": result  # ✅ Prediction Store Karega
        }
        collection.insert_one(data_to_store)  # ✅ MongoDB Me Insert Karna

        return render_template("index1.html", prediction=result)

    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == "__main__":
    app.run(debug=True)
