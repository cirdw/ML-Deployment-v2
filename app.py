from flask import Flask, render_template, request
import pickle

vectorizer = pickle.load(open('models/cv.pkl', 'rb'))
model = pickle.load(open('models/clf.pkl', 'rb'))

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods = ['POST'])
def predict() :
    text = request.form.get('email-content')
    tokenized_text = vectorizer.transform([text])
    prediction = model.predict(tokenized_text)
    prediction = 1 if prediction == 1 else -1
    return render_template('index.html', prediction = prediction, text = text)

if __name__ == '__main__' :
    app.run(debug = True)