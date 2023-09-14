import json
import pickle
import yaml
from sklearn import metrics
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from load_immo_data import load_preprocessed_immo_data
from sklearn.ensemble import GradientBoostingRegressor

model_pkl_name = "model.pkl"


def train():
    # Load config.yaml
    config = yaml.load(open('./xai/immo_data_config.yaml'), Loader=yaml.FullLoader)
    target_column = config['target_column']
    ordinal_columns = config['ordinal_columns']

    one_hot_columns = config['one_hot_columns']
    standard_scaler_columns = config['standard_scaler_columns']

    ### DATA
    df = load_preprocessed_immo_data()

    y_df = df[target_column]
    x_df = df.copy()
    x_df.drop([target_column], axis=1, inplace=True)

    # Delete baseRent as it is highly correlated with totalRent
    x_df.drop(['baseRent'], axis=1, inplace=True)  # Drop correlated rent column
    try:
        standard_scaler_columns.remove('baseRent')
    except ValueError:
        pass  # not in list.
    print(f"Using the following features: {x_df.columns.tolist()}")
    # turn one_hot_columns and standard_scaler_columns into lists of indices to train on numpy arrays and not dfs.
    column_names = list(x_df.columns)
    one_hot_columns = [column_names.index(col) for col in one_hot_columns]
    standard_scaler_columns = [column_names.index(col) for col in standard_scaler_columns]

    transformers = [
        ('one_hot', OneHotEncoder(drop='first', handle_unknown='ignore'), one_hot_columns),
        # drop='first' to avoid multicollinearity
        ('scaler', StandardScaler(), standard_scaler_columns)
    ]

    preprocessor = ColumnTransformer(transformers, remainder='drop')

    ### MODEL

    # Best hyperparameters from Random Search:
    # maxdepth: 16, minsamleaf: 117, n: 73, maxfeat: 10, lr: 0.07
    # Hyperparameters
    md = 16
    msl = 117
    n = 73
    mf = 10
    lr = 0.07

    gbm = GradientBoostingRegressor(n_estimators=n, random_state=1111,
                                    max_depth=md, max_features=mf,
                                    min_samples_leaf=msl, learning_rate=lr
                                    )

    pipeline = Pipeline([('preprocessing', preprocessor),
                         ('model', gbm)])

    # check if any record has the value 'Other'
    for col in ordinal_columns:
        if 'Other' in x_df[col].values:
            print(col, 'has other')

    # scores = cross_val_score(pipeline, x, y, cv=10, scoring='neg_mean_squared_error')

    """# Cross-validation returns negative values for scoring methods that are loss functions (lower is better).
    # Mean squared error is one such function. We'll negate the scores and then compute the RMSE.
    rmse_scores = [pow(-score, 0.5) for score in scores]

    print("RMSE for each fold:")
    print(rmse_scores)
    print("\nAverage RMSE across all folds:")
    print(sum(rmse_scores) / len(rmse_scores))"""

    X_train, X_test, y_train, y_test = train_test_split(x_df, y_df, test_size=0.30,
                                                        random_state=1)
    # convert dataframes to numpy arrays
    column_names = x_df.columns.tolist()
    X_train = X_train.to_numpy()
    X_test = X_test.to_numpy()
    y_train = y_train.to_numpy()
    y_test = y_test.to_numpy()
    pipeline.fit(X_train, y_train)

    ### Evaluating and Saving the model
    # If you'd like to save a model trained on the entire dataset:
    with open(model_pkl_name, 'wb') as file:
        pickle.dump(pipeline, file)

    # Save Data Splits to pkl
    with open('data_splits.pkl', 'wb') as file:
        pickle.dump([X_train, X_test, y_train, y_test], file)

    # Save column names to config.yaml
    config['column_names'] = column_names
    with open('./immo_data_config.yaml', 'w') as file:
        yaml.dump(config, file)


def test():
    # Load trained model
    with open(model_pkl_name, 'rb') as file:
        pipeline = pickle.load(file)

    # Load Data Splits from pkl
    with open('data_splits.pkl', 'rb') as file:
        X_train, X_test, y_train, y_test = pickle.load(file)

    y_pred = pipeline.predict(X_test)
    rmse_test = mean_squared_error(y_pred, y_test, squared=False)
    print("RMSE on test set: ", rmse_test)
    rmse_train = mean_squared_error(pipeline.predict(X_train), y_train, squared=False)
    print("RMSE on train set: ", rmse_train)

    # calc MAE
    mae = metrics.mean_absolute_error(y_test, y_pred)
    print("MAE: ", mae)


train()
test()
