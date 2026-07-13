from flask import Flask, render_template, request
import joblib

app = Flask(
    __name__,
    template_folder="../Templates",
    static_folder="../Static"
)

# Load trained model
model = joblib.load("../Models/loan_model.pkl")


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

    data = [[
        Gender,
        Married,
        Dependents,
        Education,
        Self_Employed,
        ApplicantIncome,
        CoapplicantIncome,
        LoanAmount,
        Loan_Amount_Term,
        Credit_History,
        Property_Area
    ]]

    prediction = model.predict(data)

    if prediction[0] == 1:
        result = "Loan Approved"
    else:
        result = "Loan Rejected"

    return render_template("result.html", prediction=result)


if __name__ == "__main__":
    app.run(debug=True)