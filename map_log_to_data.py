import pandas as pd
import json

def mapper(log_path: str, output_path: str):
  df = pd.read_csv(log_path, index_col=False, header=None)
  df.columns = ['timestamp', 'chat_id', 'datapoint', 'endpoint', 'conversation', 'total_score']
  print(df.shape)
  output_df = pd.DataFrame(columns=['timestamp', 'chat_id', 'nr_of_datapoint', 'datapoint', 'dialog', 'prediction_correct', 'total_score'])

  # key: chat_id -> dict: datapoint, nr_of_datapoint
  # {chat_id: [
  #   {
  #     chat_id: str
  #     datapoint: {}
  #     nr_of_datapoint: int
  #     timestamp: int
  #     dialog: longest msg array
  #     prediction_correct: score > datapoint before
  #     total_score: score
  #   }
  # ]}
  mapping = {}

  for index, input_row in df.iterrows():
    conversation: str = input_row['conversation']

    if conversation.startswith("{'messages':"):
      conversation = conversation[13:-1]

    conversation = conversation.replace('"', '`')
    conversation = conversation.replace("'", '"')

    conversation_json = json.loads(conversation)

    chat_id = input_row['chat_id']

    if input_row['endpoint'] == 'datapoint':
      if chat_id not in mapping.keys():
        mapping[chat_id] = [{
          'chat_id': chat_id,
          'datapoint': input_row['datapoint'],
          'nr_of_datapoint': 1,
          'timestamp': input_row['timestamp'],
          'dialog': [],
          'total_score': 0
        }]
      else:
        nr_of_datapoint = len(mapping[chat_id]) + 1
        mapping[chat_id].append({
          'chat_id': chat_id,
          'datapoint': input_row['datapoint'],
          'nr_of_datapoint': nr_of_datapoint,
          'timestamp': input_row['timestamp'],
          'dialog': [],
          'total_score': 0
        })

    elif input_row['endpoint'] == 'start_prompt':
      last_datapoint_in_list = mapping[chat_id][-1]
      prediction_correct = False
      if len(mapping[chat_id]) == 1:
        if input_row['total_score'] == 1:
          prediction_correct = True
      else:
        second_last_datapoint_in_list = mapping[chat_id][-2]

        if second_last_datapoint_in_list['total_score'] < input_row['total_score']:
          prediction_correct = True

      last_datapoint_in_list['dialog'] = json.dumps(conversation_json)
      last_datapoint_in_list['prediction_correct'] = prediction_correct
      last_datapoint_in_list['total_score'] = input_row['total_score']

      mapping[chat_id][-1] = last_datapoint_in_list

    elif input_row['endpoint'] == 'message':
      mapping[chat_id][-1]['dialog'] = json.dumps(conversation_json)

  for key in mapping.keys():
    for item in mapping[key]:
      output_df.loc[len(output_df)] = item

  output_df.to_csv(output_path)
  return

if __name__ == '__main__':
  mapper('./log.csv', './data.csv')