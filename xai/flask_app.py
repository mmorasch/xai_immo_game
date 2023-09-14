from flask import Flask
from flask_cors import CORS

from experiment_manager import ExperimentManager

app = Flask(__name__)

CORS(app)

manager = ExperimentManager()


@app.route("/datapoint", methods=["GET"])
def get_datapoint():
    result_dict = manager.get_next_instance()
    threshold = manager.get_threshold()
    expert_opinion = manager.get_expert_opinion()
    prediction = manager.get_current_prediction()
    result_dict["expert_opinion"] = str(expert_opinion)
    result_dict["threshold"] = str(threshold) + "€"
    result_dict["prediction"] = str(prediction)
    return result_dict


@app.route("/threshold", methods=["GET"])
def get_threshold():
    threshold = manager.get_threshold()
    return {"threshold": str(threshold) + "€"}


@app.route("/expert/<datapoint_id>", methods=["GET"])
def get_expert(datapoint_id):
    expert_opinion = manager.get_expert_opinion()
    return {"result": str(expert_opinion)}


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