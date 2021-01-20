# Boilerplate code to serve a web app

from flask import Flask, request

app = Flask(__name__)


@app.route('/')
def status():
    # Health check endpoint
    return 'Ok'


@app.route('/train')
def train():
    # Implement method to train
    return 'Training...'


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Implement method to train
        return 'Predicting...'


if __name__ == "__main__":
    app.run(debug=True, port=8000)
    # Use gunicorn for serving in production
