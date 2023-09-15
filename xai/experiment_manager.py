import pickle
import random

import pandas as pd
import yaml

from llm_prompts import create_system_message, create_apartment_with_user_prediction_prompt
from load_immo_data import load_saved_immo_data
from xai_explainer import XaiExplainer


def map_to_german(instance_dict, config):
    """Map english dict to german"""
    english_keys = config["condition_order"]
    german_keys = config["condition_order_german"]
    english_german_mapping = {key: value for key, value in zip(english_keys, german_keys)}

    heating_mapping = {
        "central_heating": "Zentralheizung",
        "combined_heat_and_power_plant": "Blockheizkraftwerk",
        "district_heating": "Fernwärme",
        "electric_heating": "Elektroheizung",
        "floor_heating": "Fußbodenheizung",
        "gas_heating": "Gasheizung",
        "heat_pump": "Wärmepumpe",
        "night_storage_heater": "Nachtspeicherofen",
        "oil_heating": "Ölheizung",
        "self_contained_central_heating": "eigenständige Zentralheizung",
        "solar_heating": "Solarheizung",
        "stove_heating": "Ofenheizung",
        "wood_pellet_heating": "Holzpelletsheizung"
    }

    true_false_mapping = {"true": "Ja", "false": "Nein"}

    instance_dict_german = {}
    for key, value in instance_dict.items():
        german_key = english_german_mapping[key]
        if key == "heatingType":
            german_value = heating_mapping[value]
        elif key == "balcony" or key == "hasKitchen" or key == "newlyConst":
            german_value = true_false_mapping[value]
        else:
            german_value = value
        instance_dict_german[german_key] = german_value
    return instance_dict_german


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
        instance_dict = map_to_german(instance_dict, self.config)
        self.instance_count += 1
        return instance_dict

    def get_current_prediction(self):
        return self.model.predict(self.current_instance)[0]

    def get_llm_context_prompt(self):
        return create_system_message(self.xai.categorical_features, self.xai.continuous_features)

    def get_llm_chat_start_prompt(self, user_prediction):
        feature_importances = self.xai.get_feature_importances(self.current_instance)[0]
        counterfactuals = self.xai.get_counterfactuals(self.current_instance, self.target_price_range)
        # Turn current instance into dict
        current_instance_dict = self.np_instance_to_dict_with_values(self.current_instance)
        return create_apartment_with_user_prediction_prompt(current_instance_dict,
                                                            self.get_threshold(),
                                                            self.get_correct_price(),
                                                            user_prediction,
                                                            self.correct_answer,
                                                            feature_importances,
                                                            counterfactuals)

    def get_correct_price(self):
        return self.current_instance_y[0]

    def get_threshold(self):
        if self.instance_count % 2 == 0:
            self.correct_answer = "0"  # lower
            self.target_price_range = [self.get_correct_price() + 300, self.get_correct_price() + 600]
            return round(self.get_correct_price() + 300)  # higher threshold (correc click is lower)
        else:
            self.correct_answer = "1"  # higher
            self.target_price_range = [self.get_correct_price() - 300, self.get_correct_price() - 600]
            return round(self.get_correct_price() - 300)  # lower threshold (correc click is higher)

    def get_expert_opinion(self):
        "Expert opinion is random for now"
        return random.choice([0, 1])  # TODO: 0 and 1 or lower and higher?
