from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)
classifier = joblib.load('classifier.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    prediction = classifier.predict([data['features']])
    return jsonify({'prediction': prediction.tolist()})

if __name__ == '__main__':
    app.run(debug=True)
