import json
import pickle

import numpy as np
import pandas as pd
import yaml
from lime import lime_tabular
import dice_ml
from dice_ml import Dice

from matplotlib import pyplot as plt

from load_immo_data import load_saved_immo_data

# Load config.yaml
config = yaml.load(open('./immo_data_config.yaml'), Loader=yaml.FullLoader)
column_names = config['column_names']
standard_scaler_columns = config['standard_scaler_columns']


class XaiExplainer:
    def __init__(self,
                 config: dict,
                 X_train: np.ndarray,
                 y_train: np.ndarray,
                 model):
        self.config = config
        self.target_column = config['target_column']
        self.X_train = X_train
        self.y_train = y_train
        self.model = model
        self.categorical_names_dict = self.load_column_mapping()
        # Get categorical features
        self.categorical_features = config['ordinal_columns']
        self.categorical_features.extend(config['one_hot_columns'])
        self.categorical_indices = [column_names.index(col) for col in
                                    self.categorical_features]  # indices of cat features
        # Get continuous features
        self.continuous_features = config['standard_scaler_columns']  # as strings
        self.background_df = self.create_background_df()  # background data for LIME and DICE
        self.column_names_for_df = column_names.copy()
        self.column_names_for_df.append(self.target_column)  # background_df contains target column
        self.column_names_for_np = column_names.copy()  # np array does not contain target column

    def create_background_df(self):
        df = pd.DataFrame(self.X_train, columns=self.config['column_names'])
        df[self.target_column] = self.y_train
        for col in self.categorical_features:
            df[col] = df[col].astype('int')
        return df

    def load_column_mapping(self):
        # load categorical names from json file
        with open('immo_column_id_to_values_mapping.json', 'r') as file:
            categorical_names_dict = json.load(file)
        # Add condition to categorical names (it is not in the json file because it is ordered)
        categorical_names_dict[column_names.index('condition')] = config['condition_order']
        return categorical_names_dict

    def get_feature_importances(self, data_instance):
        # Use LIME to explain instance
        # preprocessor = model.named_steps['preprocessing']
        # background_data = preprocessor.transform(background_data)
        explainer = lime_tabular.LimeTabularExplainer(self.X_train,
                                                      mode="regression",
                                                      categorical_features=self.categorical_indices,
                                                      categorical_names=self.categorical_names_dict,
                                                      feature_names=column_names,
                                                      sample_around_instance=True,
                                                      discretize_continuous=True,
                                                      kernel_width=0.75 * np.sqrt(self.X_train.shape[1]))
        prediction = model.predict(data_instance)
        output = explainer.explain_instance(data_instance.flatten(),
                                            model.predict,
                                            num_features=data_instance.shape[1])
        # TODO: What to return here?
        # Map output to feature_names and values
        intercept = output.intercept[0]
        results = output.as_list()
        # output.as_pyplot_figure()
        # Make y labels fit
        plt.tight_layout()
        plt.show()

    def get_counterfactual_explanation(self, data_instance):
        d_housing = dice_ml.Data(dataframe=self.background_df, continuous_features=self.continuous_features,
                                 outcome_name=self.target_column)
        # We provide the type of model as a parameter (model_type)
        m_housing = dice_ml.Model(model=self.model, backend="sklearn", model_type='regressor')

        exp_genetic_housing = Dice(d_housing, m_housing, method="genetic")

        # predict price
        prediction = self.model.predict(data_instance)
        # turn data instance to df if it is not already
        if isinstance(data_instance, np.ndarray):
            data_instance = pd.DataFrame(data_instance, columns=self.column_names_for_np)
            # make categorical columns to integer
            for col in self.categorical_features:
                data_instance[col] = data_instance[col].astype('int')

        counterfactuals = exp_genetic_housing.generate_counterfactuals(data_instance,
                                                                       total_CFs=3,
                                                                       desired_range=[500, 700])
        print(counterfactuals.visualize_as_list())
        # counterfactuals.visualize_as_dataframe(show_only_changes=True)
        return counterfactuals


# Load trained model
with open("model.pkl", 'rb') as file:
    model = pickle.load(file)

# Load Data Splits from pkl
X_train, X_test, y_train, y_test = load_saved_immo_data()  # np arrays

# Get random instance to explain
data_instance = X_test[2:3]
xai = XaiExplainer(config, X_train, y_train, model)
# xai.get_feature_importances(data_instance)
xai.get_counterfactual_explanation(data_instance)
