"""
This script is used to evaluate the logistic regression model for various
regularization parameters and class_weight options. Again, AUC and the Brier
score are used for model selection.
"""

from logistic_features import calculate_features
from sklearn.externals import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.linear_model import LogisticRegression
import pandas as pd

if __name__ == "__main__":
    print("Loading models")
    body_model = joblib.load("./final/body_model.pkl")
    title_model = joblib.load("./final/title_model.pkl")

    print("Loading training data")
    train_data = pd.read_csv("/data/SO_data/downvoter/lr_train_processed_data.csv")
    train_body_vecs = joblib.load("./final/vectorized_data/train_body_data.pkl")
    train_title_vecs = joblib.load("./final/vectorized_data/train_title_data.pkl")

    print("Calculating training features")
    train_X = calculate_features(train_data,
                                 train_body_vecs,
                                 body_model,
                                 train_title_vecs,
                                 title_model)
    train_labels = train_data.score < 0


    print("Loading validation data")
    val_data = pd.read_csv("/data/SO_data/downvoter/lr_val_processed_data.csv")
    val_body_vecs = joblib.load("./final/vectorized_data/val_body_data.pkl")
    val_title_vecs = joblib.load("./final/vectorized_data/val_title_data.pkl")

    print("Calculating validation features")
    val_X = calculate_features(val_data,
                               val_body_vecs,
                               body_model,
                               val_title_vecs,
                               title_model)
    val_labels = val_data.score < 0

    print("Scaling data")
    sc = StandardScaler()
    train_Xsc = sc.fit_transform(train_X)
    val_Xsc = sc.transform(val_X)

    print("Running validation")
    results = pd.DataFrame(columns=["C", "class_weight", "auc", "brier"])

    Cs = [10.0 ** i for i in range(-5, 6)]
    cws = [None, "balanced"]

    for C in Cs:
        for cw in cws:
            print("Processing C=%s cw=%s" % (C, cw))
            model = LogisticRegression(C=C, class_weight=cw)

            print("Fitting...")
            model.fit(train_Xsc, train_labels)

            print("Testing...")
            preds = model.predict_proba(val_Xsc)

            auc = roc_auc_score(val_labels, preds[:,1])
            brier = brier_score_loss(val_labels, preds[:,1])

            results = results.append({"C" : C,
                                      "class_weight" : cw,
                                      "auc" : auc,
                                      "brier" : brier},
                                     ignore_index=True)
            print("AUC: %.3f, BRIER: %.3f" % (auc, brier))

    results.to_csv("./logistic_results.csv", index=False)
    print("All done!")

