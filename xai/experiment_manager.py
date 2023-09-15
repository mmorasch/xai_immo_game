import pickle
import random

import pandas as pd
import yaml

from llm_prompts import create_system_message, create_apartment_with_user_prediction_prompt
from load_immo_data import load_saved_immo_data
from xai_explainer import XaiExplainer


class ExperimentManager:
    def __init__(self):
        self.current_instance = None
        self.setup()

    def setup(self):
        # Load config
        self.config = yaml.load(open('./xai/immo_data_config.yaml'), Loader=yaml.FullLoader)
        # Load Data
        X_train, X_test, y_train, y_test = load_saved_immo_data()
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.test_instances = self.X_test[0:100]
        self.instance_count = 0
        # Load Model
        with open("xai/model.pkl", 'rb') as file:
            model = pickle.load(file)
        self.model = model
        # Load XAI
        self.xai = XaiExplainer(self.config, X_train, y_train, self.model)

    def np_instance_to_dict_with_values(self, instance):
        # Return instance as dict and map column values to column names
        df = pd.DataFrame(instance, columns=self.config['column_names'])
        for col in self.xai.categorical_features:
            df[col] = df[col].astype('int')
        # Map categorical values to names
        for col in self.xai.categorical_features:
            col_id = self.xai.column_names.index(col)
            feature_value_name = self.xai.categorical_names_dict[col_id][int(df[col])]
            df[col] = feature_value_name
        instance_dict = df.to_dict('r')[0]
        return instance_dict

    def get_next_instance(self):
        # Find an instance where model error is small and 'other' is not in the condition column
        model_error = 1000
        while model_error > 100:
            # Get instance and y label and delete from X_test and y_test
            self.current_instance = self.test_instances[0:1]
            # delete current instance from test instances
            self.test_instances = self.test_instances[1:]
            # Store y value of current instance
            self.current_instance_y = self.y_test[0:1]
            # delete current instance y from y_test
            self.y_test = self.y_test[1:]
            # check if 'other' is in current instance
            if 0 == self.current_instance[0][0]:
                continue
            model_error = abs(self.model.predict(self.current_instance)[0] - self.current_instance_y[0])
        instance_dict = self.np_instance_to_dict_with_values(self.current_instance)
        self.instance_count += 1
        return instance_dict

    def get_current_prediction(self):
        return self.model.predict(self.current_instance)[0]

    def get_llm_context_prompt(self):
        return create_system_message(self.xai.categorical_features, self.xai.continuous_features)

    def get_llm_chat_start_prompt(self, user_prediction):
        feature_importances = self.xai.get_feature_importances(self.current_instance)[0]
        target_price_range = [self.get_correct_price() + 100, self.get_correct_price() + 300]
        threshold = self.get_correct_price() - 100
        counterfactuals = self.xai.get_counterfactuals(self.current_instance, target_price_range)
        # Turn current instance into dict
        current_instance_dict = self.np_instance_to_dict_with_values(self.current_instance)
        return create_apartment_with_user_prediction_prompt(current_instance_dict,
                                                            threshold,
                                                            self.get_correct_price(),
                                                            user_prediction,
                                                            feature_importances,
                                                            counterfactuals)

    def get_correct_price(self):
        return self.current_instance_y[0]

    def get_threshold(self):
        if self.instance_count % 2 == 0:
            return round(self.get_correct_price() + 300)  # higher threshold (correc click is lower)
        else:
            return round(self.get_correct_price() - 300)  # lower threshold (correc click is higher)

    def get_expert_opinion(self):
        "Expert opinion is random for now"
        return random.choice([0, 1])  # TODO: 0 and 1 or lower and higher?
