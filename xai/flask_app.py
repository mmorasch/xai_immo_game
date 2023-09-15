from flask import Flask, request
from flask_cors import CORS
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage, AIMessage
from llm_prompts import create_system_message
from experiment_manager import ExperimentManager

app = Flask(__name__)

CORS(app)

manager = ExperimentManager()

chat = ChatOpenAI(openai_api_key='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')
sys_msg1 = SystemMessage(content=manager.get_llm_context_prompt())


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


@app.route("/start_prompt/<slug>", methods=["GET"])
def get_start_prompt(slug):
    # TODO: Only works if get_datapoint was called before
    userPrediction = request.args.get('prediction')
    start_prompt = SystemMessage(content=manager.get_llm_chat_start_prompt(userPrediction))
    result = chat.predict_messages([sys_msg1, start_prompt])
    return {"messages": [
        {"role": "system", "message": sys_msg1.content},
        {"role": "system", "message": start_prompt.content},
        {"role": "assistant", "message": result.content}
    ]}


# TODO: langchain prompt OpenAI with chat message. require slug for tracking in backend
@app.route("/message/<slug>", methods=["POST"])
def post_message(slug):
    messages = request.json['messages']

    return {"message": "response from backend"}

if __name__ == "__main__":
    app.run(debug=False, port=4455, host='0.0.0.0')