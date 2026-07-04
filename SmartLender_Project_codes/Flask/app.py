from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load Model
model = pickle.load(open("xgb_model.pkl", "rb"))
scaler = pickle.load(open("scale1.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    Gender = float(request.form["Gender"])
    Married = float(request.form["Married"])
    Dependents = float(request.form["Dependents"])
    Education = float(request.form["Education"])
    Self_Employed = float(request.form["Self_Employed"])
    ApplicantIncome = float(request.form["ApplicantIncome"])
    CoapplicantIncome = float(request.form["CoapplicantIncome"])
    LoanAmount = float(request.form["LoanAmount"])
    Loan_Amount_Term = float(request.form["Loan_Amount_Term"])
    Credit_History = float(request.form["Credit_History"])
    Property_Area = float(request.form["Property_Area"])

    data = np.array([[Gender,
                      Married,
                      Dependents,
                      Education,
                      Self_Employed,
                      ApplicantIncome,
                      CoapplicantIncome,
                      LoanAmount,
                      Loan_Amount_Term,
                      Credit_History,
                      Property_Area]])

    data = scaler.transform(data)

    prediction = model.predict(data)

    if prediction[0] == 1:
        result = "Loan Approved ✅"
    else:
        result = "Loan Rejected ❌"

    return render_template("result.html", prediction=result)


if __name__ == "__main__":
    app.run(debug=True)