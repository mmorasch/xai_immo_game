from flask import Flask
from flask_cors import CORS

from experiment_manager import ExperimentManager

app = Flask(__name__)

CORS(app)

manager = ExperimentManager()


@app.route("/datapoint", methods=["GET"])
def get_datapoint():
    next_instance_dict = manager.get_next_instance()
    return next_instance_dict


@app.route("/threshold", methods=["GET"])
def get_threshold():
    threshold = manager.get_threshold()
    return {"threshold": str(threshold) + "€"}


# TODO: get experts opinion on datapoint given datapoint id
@app.route("/expert/<datapoint_id>", methods=["GET"])
def get_expert(datapoint_id):
    return {"result": "expert opinion"}  # TODO 0 or 1?


@app.route("/prediction/<datapoint_id>", methods=["GET"])
def get_prediction(datapoint_id):
    # TODO: Only works if get_datapoint was called before
    prediction = manager.get_current_prediction()
    return {"result": str(prediction)}


# TODO: langchain prompt OpenAI with chat message. require slug for tracking in backend
@app.route("/message/<slug>", methods=["POST"])
def post_message(slug):
    return "<p>Hello, World!</p>"

if __name__ == "__main__":
    app.run(debug=False, port=4455, host='0.0.0.0')