import pickle

import pandas as pd
import yaml

from load_immo_data import load_saved_immo_data
from xai_explainer import XaiExplainer


class ExperimentManager:
    def __init__(self):
        self.current_instance = None
        self.setup()

    def setup(self):
        # Load config
        self.config = yaml.load(open('xai/immo_data_config.yaml'), Loader=yaml.FullLoader)
        # Load Data
        X_train, X_test, y_train, y_test = load_saved_immo_data()
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.test_instances = self.X_test[0:10]
        # Load Model
        with open("model.pkl", 'rb') as file:
            model = pickle.load(file)
        self.model = model
        # Load XAI
        self.xai = XaiExplainer(self.config, X_train, y_train, self.model)

    def get_next_instance(self):
        self.current_instance = self.test_instances[0:1]
        # delete current instance from test instances
        self.test_instances = self.test_instances[1:]
        # Return instance as dict and map column values to column names
        df = pd.DataFrame(self.current_instance, columns=self.config['column_names'])
        for col in self.xai.categorical_features:
            df[col] = df[col].astype('int')
        instance_dict = df.to_dict()
        return instance_dict

    def get_current_prediction(self):
        return self.model.predict(self.current_instance)[0]
