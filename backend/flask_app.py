from flask import Flask

app = Flask(__name__)

@app.route("/datapoint", methods=["GET"])
def get_datapoint():
    return "<p>Hello, World!</p>"

@app.route("/prediction", methods=["GET"])
def get_prediction():
    return "<p>Hello, World!</p>"

@app.route("/message", methods=["POST"])
def post_message():
    return "<p>Hello, World!</p>"