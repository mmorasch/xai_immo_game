import json
import pickle

import numpy as np
import pandas as pd
import yaml
from lime import lime_tabular
import dice_ml
from dice_ml import Dice

from matplotlib import pyplot as plt


class XaiExplainer:
    def __init__(self,
                 config: dict,
                 X_train: np.ndarray,
                 y_train: np.ndarray,
                 model):
        self.config = config
        self.target_column = config['target_column']
        self.categorical_features = config['ordinal_columns']
        self.categorical_features.extend(config['one_hot_columns'])
        self.column_names = config['column_names']
        self.X_train = X_train
        self.y_train = y_train
        self.model = model
        self.categorical_names_dict = self.load_column_mapping()
        # Get categorical features
        self.categorical_indices = [self.column_names.index(col) for col in
                                    self.categorical_features]  # indices of cat features
        # Get continuous features
        self.continuous_features = config['standard_scaler_columns']  # as strings
        self.background_df = self.create_background_df()  # background data for LIME and DICE
        self.column_names_for_df = self.column_names.copy()
        self.column_names_for_df.append(self.target_column)  # background_df contains target column
        self.column_names_for_np = self.column_names.copy()  # np array does not contain target column

    def create_background_df(self):
        df = pd.DataFrame(self.X_train, columns=self.config['column_names'])
        df[self.target_column] = self.y_train
        for col in self.categorical_features:
            df[col] = df[col].astype('int')
        return df

    def load_column_mapping(self):
        # load categorical names from json file
        with open('./immo_column_id_to_values_mapping.json', 'r') as file:
            categorical_names_dict = json.load(file)
        # Add condition to categorical names (it is not in the json file because it is ordered)
        categorical_names_dict[self.column_names.index('condition')] = self.config['condition_order']
        # turn keys to int
        categorical_names_dict = {int(k): v for k, v in categorical_names_dict.items()}
        return categorical_names_dict

    def get_feature_importances(self, data_instance):
        # Use LIME to explain instance
        # preprocessor = model.named_steps['preprocessing']
        # background_data = preprocessor.transform(background_data)
        explainer = lime_tabular.LimeTabularExplainer(self.X_train,
                                                      mode="regression",
                                                      categorical_features=self.categorical_indices,
                                                      categorical_names=self.categorical_names_dict,
                                                      feature_names=self.column_names,
                                                      sample_around_instance=True,
                                                      discretize_continuous=True,
                                                      kernel_width=0.75 * np.sqrt(self.X_train.shape[1]))
        # prediction = self.model.predict(data_instance)
        output = explainer.explain_instance(data_instance.flatten(),
                                            self.model.predict,
                                            num_features=data_instance.shape[1])

        # Map output to feature_names and values
        intercept = output.intercept[0]
        results = output.as_list()
        return results, intercept

    def get_counterfactuals(self,
                            data_instance,
                            target_range,
                            num_cf=3,
                            as_string=True):
        """
        Get counterfactual explanations for a given instance using DICE with a target range for the target column
        e.g. target_range = [900, 1000] means that the target column should be between 900 and 1000.
        :param data_instance: instance to explain
        :param target_range: range for target column
        :param num_cf: number of counterfactuals to generate
        :param as_string: Whether to return the change string or the cf instance.
        :return: counterfactuals
        """

        def get_final_cfes(instance_id, explanation):
            """
            Returns the final cfes as pandas df and their ids for a given data instance.
            """
            cfe = explanation.cf_examples_list[instance_id]
            final_cfes = cfe.final_cfs_df
            final_cfe_ids = list(final_cfes.index)

            if self.target_column in final_cfes.columns:
                final_cfes.pop(self.target_column)
            return final_cfes, final_cfe_ids

        def get_change_string(cfe, original_instance):
            rounding_precision = 2
            """Builds a string describing the changes between the cfe and original instance."""
            original_features = list(original_instance.columns)
            change_string = ""
            for feature in original_features:
                feature_index = original_features.index(feature)
                orig_f = original_instance[feature].values[0]
                cfe_f = cfe[feature].values[0]

                if isinstance(cfe_f, str):
                    cfe_f = float(cfe_f)

                if orig_f != cfe_f:
                    if cfe_f > orig_f:
                        inc_dec = "Increasing"
                    else:
                        inc_dec = "Decreasing"
                    # Turn feature to categorical name if possible
                    if self.categorical_names_dict is not None:
                        try:
                            cfe_f = self.categorical_names_dict[feature_index][int(cfe_f)]
                            inc_dec = "Changing"
                        except KeyError:
                            pass  # feature is numeric and not in categorical mapping
                        except IndexError:
                            print("Index error in DICE explanation encountered...")
                    # round cfe_f if it is float and turn to string to print
                    if isinstance(cfe_f, float):
                        cfe_f = str(round(cfe_f, rounding_precision))
                    change_string += f"{inc_dec} feature {feature} to {cfe_f}"
                    change_string += " and "
            # Strip off last and
            change_string = change_string[:-5]
            return change_string

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
                                                                       total_CFs=num_cf,
                                                                       desired_range=target_range)
        print(counterfactuals.visualize_as_list())
        # counterfactuals.visualize_as_dataframe(show_only_changes=True)
        instance_id = data_instance.index[0]
        cfes = get_final_cfes(instance_id, explanation=counterfactuals)
        change_strings = []
        # Itterate over df
        for index, cfe in cfes[0].iterrows():
            cfe = pd.DataFrame(cfe).T
            change_strings.append(get_change_string(cfe, data_instance))
        return change_strings
