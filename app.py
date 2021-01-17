from flask import Flask,render_template, request, url_for, redirect
import pickle
import numpy as np

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=['GET', 'POST'])
def predict():
    Age = request.form.get('Age')
    Sex = request.form.get('Sex')
    Pclass = request.form.get('Pclass')
    Fare = request.form.get('Fare')
    Embarked = request.form.get('Embarked')
    Title = request.form.get('Title')
    IsAlone = request.form.get('IsAlone')

    variables = [Age,Sex,Pclass,Fare,Embarked,Title,IsAlone]
    for i in variables:
        if i == "6":
            return redirect(url_for("index"))
        
    with open('model/model.pkl', 'rb') as f:
        model = pickle.load(f)
    arr = np.array([Age,Sex,Pclass,Fare,Embarked,Title,IsAlone]).reshape(1,-1)
    predict = model.predict(arr)
    if predict == 0:
        return render_template("result.html", data=predict)
    else:
        return render_template("result2.html", data=predict)


if __name__ == "__main__":
    app.run(debug=True)