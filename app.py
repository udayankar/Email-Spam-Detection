from flask import Flask, request, render_template, redirect, url_for
import joblib


# Load trained model and vectorizer

model = joblib.load("spam_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")


# Flask App

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html", prediction=None)

@app.route("/predict", methods=["POST"])
def predict():
    message = request.form["message"]
    data = vectorizer.transform([message]).toarray()
    prediction = model.predict(data)[0]
    result = "Spam" if prediction == 1 else "Not Spam"
    return redirect(url_for("show_result", prediction=result))

@app.route("/result/<prediction>")
def show_result(prediction):
    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
