from flask import Flask

app = Flask(__name__)

# TODO: get a datapoint, including all columns
@app.route("/datapoint", methods=["GET"])
def get_datapoint():
    return "<p>Hello, World!</p>"

# TODO: get experts opinion on datapoint given datapoint id
@app.route("/expert/<datapoint_id>", methods=["GET"])
def get_expert(datapoint_id):
    return f'<p>Hello, World! {datapoint_id} </p>'

# TODO: get prediction from xai.py for given datapoint id
@app.route("/prediction/<datapoint_id>", methods=["GET"])
def get_prediction(datapoint_id):
    return f'<p>Hello, World! {datapoint_id} </p>'

# TODO: langchain prompt OpenAI with chat message. require slug for tracking in backend
@app.route("/message/<slug>", methods=["POST"])
def post_message(slug):
    return "<p>Hello, World!</p>"