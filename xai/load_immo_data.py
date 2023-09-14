import json
import pickle

import numpy as np
import pandas as pd
import yaml
from sklearn.preprocessing import LabelEncoder


def _save_column_id_to_value_index_mapping(data: np.ndarray,
                                           column_ids: [int]):
    """
    Uses LabelEncoder on the data to save the mapping from column id to value index.
    Takes in a dataset in numpy and a list of column indices to create a nested dict from the column indices to
    indexed unique values in a column of the dataset. {col_id: {value_id: unique_value}}
    Needed for XAI methods that handle categorical features this way.
    """
    categorical_names = {}
    for column_id in column_ids:
        # As in LIME example "Categorical features"
        # https://marcotcr.github.io/lime/tutorials/Tutorial%20-%20continuous%20and%20categorical%20features.html
        le = LabelEncoder()
        le.fit(data[:, column_id])
        data[:, column_id] = le.transform(data[:, column_id])
        categorical_names[column_id] = le.classes_.tolist()

    with open("immo_column_id_to_values_mapping.json", "w") as f:
        json.dump(categorical_names, f)

    return data, categorical_names


def missing_values(df, norows):  # input by the df and the number of rows that you want to show
    total = df.isnull().sum().sort_values(ascending=False)
    percent = ((df.isnull().sum().sort_values(ascending=False) / df.shape[0]) * 100).sort_values(ascending=False)
    missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
    return (missing_data.head(norows))


def load_preprocessed_immo_data():
    df = pd.read_csv('../immo_data.csv')
    # Load config.yaml
    config = yaml.load(open('./immo_data_config.yaml'), Loader=yaml.FullLoader)
    wanted_cols = config['ordinal_columns'] + config['one_hot_columns'] + config['standard_scaler_columns']
    wanted_cols.append('baseRent')
    wanted_cols.append(config['target_column'])
    df = df.loc[:, wanted_cols]

    ###### Handle Missing Values #######
    missing_data = missing_values(df, 20)
    # drop the data where the columns contains more than 30%
    df = df.drop(columns=(missing_data[missing_data['Percent'] > 30]).index)

    # Drop rows where the totalRent is NA
    df.dropna(subset=['totalRent'], inplace=True)

    # Handle condition property
    df['condition'].fillna("Other", inplace=True)  # fill the NA by Other
    otherscondition = df['condition'].value_counts().tail(3).index  # combine the last 3 categories into one
    othersregion = list(df['condition'].value_counts().tail(3).index)

    def editcondition(dflist):
        if dflist in otherscondition:
            return 'Other'
        else:
            return dflist

    df['condition'] = df['condition'].apply(editcondition)

    # Handle yearConstructed property
    # Fill NA of 'yearConstructed' with the mean of each type of condition 'condition' because from my perspective if the
    # apartment is not fully_renovated or refurbished it means that it should have a lot of usage year.
    df["yearConstructed"] = df['yearConstructed'].fillna(
        df.groupby('condition')['yearConstructed'].transform('mean')).round(0)

    ### Handle Regions
    # Change regions that are the last 20 frequent to 'Other'
    # Get columns that start with regio
    regio_col_names = [col for col in df.columns if col.startswith('regio')]
    for col in regio_col_names:
        othersregion = set(df[col].value_counts().iloc[20:].index)
        df[col] = df[col].apply(lambda x: 'Other' if x in othersregion else x)

    # Filter rent outliers
    df = df[(df['baseRent'] > 200) & (df['baseRent'] < 8000)]
    df = df[(df['totalRent'] > 200) & (df['totalRent'] < 9000)]
    df = df[(df['totalRent'] > df['baseRent'])]
    df = df[(df['totalRent'] - df['baseRent']) < 500]

    # Living Space
    df = df[(df['livingSpace'] > 10) & (df['livingSpace'] < 400)]

    # Feature engineering
    # df['pricePm2'] = df['baseRent'] / df['livingSpace']
    # df['additionCost'] = df['totalRent'] - df['baseRent']
    # df['numberOfYear'] = date.today().year - df["yearConstructed"]

    # Count nan values for heatingType
    # Fill it with other
    try:
        df['heatingType'].fillna(df['heatingType'].mode()[0], inplace=True)
    except KeyError:
        pass
    try:
        df['typeOfFlat'].fillna(df['typeOfFlat'].mode()[0], inplace=True)
        # change typeOfFlat 'other' values to 'Other'
        df['typeOfFlat'] = df['typeOfFlat'].apply(lambda x: 'Other' if x == 'other' else x)
        # Check if any column has 'other' and 'Other' values, if yes, print the column name
        for col in df.columns:
            if 'other' in df[col].unique() and 'Other' in df[col].unique():
                print("There are both 'other' and 'Other' values in column: " + col)
    except KeyError:
        pass

    print(missing_values(df, 5))

    # Map categorical values to numerical values
    col_names = list(df.columns)
    categorical_col_ids = [df.columns.get_loc(col) for col in config['one_hot_columns']]
    df_numpy, categorical_mapping = _save_column_id_to_value_index_mapping(df.to_numpy(), categorical_col_ids)
    df = pd.DataFrame(df_numpy, columns=col_names)
    # Map condition to numerical values
    condition_mapping = config['condition_order']
    # Create a dictionary that maps each condition to its index in the ordered list
    mapping_dict = {condition: idx for idx, condition in enumerate(condition_mapping)}
    # Map the 'condition' column in the dataframe using the mapping dictionary
    df['condition'] = df['condition'].map(mapping_dict)
    # Change dtype of columns with id in categorical_col_ids columns to int
    for col_idx, col in enumerate(df.columns):
        if col_idx in categorical_col_ids or col == 'condition':
            df[col] = df[col].astype(int)
        else:
            df[col] = df[col].astype(float)
    return df


def load_saved_immo_data():
    # Load Data Splits from pkl
    with open('data_splits.pkl', 'rb') as file:
        X_train, X_test, y_train, y_test = pickle.load(file)  # np arrays

    return X_train, X_test, y_train, y_test
