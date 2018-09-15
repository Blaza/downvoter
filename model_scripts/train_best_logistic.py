"""
This script is used for training the selected logistic regression model, which
turned out to be the plain logistic regression model with default parameters.
"""

from logistic_features import calculate_features
from sklearn.externals import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np

if __name__ == "__main__":
    print("Loading models")
    body_model = joblib.load("./final/body_model.pkl")
    title_model = joblib.load("./final/title_model.pkl")

    print("Loading data")
    data = pd.concat([
        pd.read_csv("/data/SO_data/downvoter/lr_train_processed_data.csv"),
        pd.read_csv("/data/SO_data/downvoter/lr_val_processed_data.csv")
    ])

    body_vecs = np.vstack([
        joblib.load("./final/vectorized_data/train_body_data.pkl"),
        joblib.load("./final/vectorized_data/val_body_data.pkl")
    ])

    title_vecs = np.vstack([
        joblib.load("./final/vectorized_data/train_title_data.pkl"),
        joblib.load("./final/vectorized_data/val_title_data.pkl")
    ])

    print(data.shape)
    print(body_vecs.shape)
    print(title_vecs.shape)


    print("Calculating features and labels")
    X = calculate_features(data, body_vecs, body_model, title_vecs, title_model)
    labels = data.score < 0

    print("Scaling features")
    sc = StandardScaler()
    Xsc = sc.fit_transform(X)

    print("Fitting model")
    model = LogisticRegression()
    model.fit(Xsc, labels)

    joblib.dump(model, "./final/logistic_model.pkl")
    joblib.dump(sc, "./final/logistic_scaler.pkl")

    print("All done!")
