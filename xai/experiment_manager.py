import pickle
import random

import pandas as pd
import yaml

from llm_prompts_de import create_system_message, create_apartment_with_user_prediction_prompt
from load_immo_data import load_saved_immo_data
from xai_explainer import XaiExplainer


class ExperimentManager:
    def __init__(self, language="de"):
        self.current_instance = None
        self.language = language
        self.setup()

    def setup(self):
        # Load config
        if self.language == "de":
            self.config = yaml.load(open('./xai/immo_data_config_de.yaml'), Loader=yaml.FullLoader)
        elif self.language == "en":
            self.config = yaml.load(open('./xai/immo_data_config_en.yaml'), Loader=yaml.FullLoader)
        # Load Data
        X_train, X_test, y_train, y_test = load_saved_immo_data()
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test[0:30]
        self.test_instances = self.X_test[0:30]
        self.test_instances_with_id = [(instance_id, instance) for instance_id, instance in
                                       enumerate(self.test_instances)]
        self.instance_count = 0
        self.current_instance_id = None
        # Load Model
        with open("xai/model.pkl", 'rb') as file:
            model = pickle.load(file)
        self.model = model
        # Load XAI
        self.xai = XaiExplainer(self.config, X_train, y_train, self.model)
        # check if explanations are precomputed
        try:
            with open('xai/instance_id_explanations_dict.pkl', 'rb') as file:
                self.instance_id_explanations_dict = pickle.load(file)
        except FileNotFoundError:
            self.precompute_explanations()
            # load
            with open('xai/instance_id_explanations_dict.pkl', 'rb') as file:
                self.instance_id_explanations_dict = pickle.load(file)

    def np_instance_to_dict_with_values(self, instance):
        # Return instance as dict and map column values to column names
        df = pd.DataFrame(instance.reshape(1, -1), columns=self.config['column_names'])
        for col in self.xai.categorical_features:
            df[col] = df[col].astype('int')
        # Map categorical values to names
        for col in self.xai.categorical_features:
            col_id = self.xai.column_names.index(col)
            feature_value_name = self.xai.categorical_names_dict[col_id][int(df[col])]
            df[col] = feature_value_name
        instance_dict = df.to_dict('r')[0]
        return instance_dict

    def find_suitable_instance_ids(self):
        suitable_instance_ids = []
        # iterate numpy array
        for (instance_id, instance), y_test in zip(self.test_instances_with_id, self.y_test):
            # make instance 2d array and transpose
            instance = instance.reshape(1, -1)
            model_error = abs(self.model.predict(instance)[0] - y_test)
            if 0 == instance[0][0]:
                continue
            if model_error < 100:
                suitable_instance_ids.append(instance_id)
        self.suitable_instance_ids = suitable_instance_ids

    def get_next_instance(self):
        # Find an instance where model error is small and 'other' is not in the condition column
        if self.current_instance_id is None:
            self.find_suitable_instance_ids()
            self.current_instance_id = self.suitable_instance_ids[0]
        else:
            # get next instance id
            self.current_instance_id = self.suitable_instance_ids[self.suitable_instance_ids.index(
                self.current_instance_id) + 1]
        # get instance
        self.current_instance = self.test_instances[self.current_instance_id].reshape(1, -1)
        self.current_instance_y = self.y_test[self.current_instance_id]
        instance_dict = self.np_instance_to_dict_with_values(self.current_instance)
        self.instance_count += 1
        return instance_dict

    def get_current_prediction(self):
        return self.model.predict(self.current_instance)[0]

    def get_llm_context_prompt(self):
        return create_system_message(self.xai.categorical_features, self.xai.continuous_features)

    def precompute_explanations(self):
        instance_id_explanations_dict = {}
        self.find_suitable_instance_ids()
        for instance_id, instance in self.test_instances_with_id:
            if instance_id not in self.suitable_instance_ids:
                continue
            self.current_instance_y = self.y_test[instance_id]
            instance_2d = instance.reshape(1, -1)
            self.get_threshold()
            self.get_correct_price()
            # self.current_cfs = self.xai.get_counterfactuals(instance_2d, self.target_price_range)
            self.current_feature_importances = self.xai.get_feature_importances(instance_2d)[0]
            instance_id_explanations_dict[instance_id] = {'fis': self.current_feature_importances}
        # pickle
        with open('xai/instance_id_explanations_dict.pkl', 'wb') as handle:
            pickle.dump(instance_id_explanations_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def get_llm_chat_start_prompt(self, user_prediction):
        # Turn current instance into dict
        current_instance_dict = self.np_instance_to_dict_with_values(self.current_instance)
        feature_importances = self.instance_id_explanations_dict[self.current_instance_id]['fis']
        return create_apartment_with_user_prediction_prompt(current_instance_dict,
                                                            self.get_threshold(),
                                                            self.get_correct_price(),
                                                            user_prediction,
                                                            self.correct_answer,
                                                            feature_importances,
                                                            expert_prediction=self.expert_prediction)

    def get_correct_price(self):
        try:
            return self.current_instance_y[0]
        except IndexError:
            return self.current_instance_y

    def get_threshold(self):
        self.threshold = round(self.get_correct_price() + self.get_correct_price() / 6)
        if self.instance_count % 2 == 0:
            self.correct_answer = "0"  # lower
            self.target_price_range = [self.get_correct_price() + self.threshold,
                                       self.get_correct_price() + (self.threshold + 100)]
        else:
            self.correct_answer = "1"  # higher
            self.target_price_range = [self.get_correct_price() - (self.threshold + 100),
                                       self.get_correct_price() - self.threshold]
        return self.threshold

    def get_expert_opinion(self):
        "Expert opinion is random for now"
        self.expert_prediction = str(random.choice([0, 1]))  # TODO: 0 and 1 or lower and higher?
        return self.expert_prediction
