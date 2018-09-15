"""
A script which tests the final stacked model on the test set.
Also gives classification reports for several threshold values for the logistic
regression predict_proba.

Note for Github. The final model is located in the folder ./final which isn't
present on Github as the models are ~800 MB in size and thus too big to upload.
"""

from sklearn.metrics import roc_auc_score, brier_score_loss,\
                            fbeta_score, classification_report
from sklearn.externals import joblib
from final_model import predict_probas
import pandas as pd

if  __name__ == "__main__":
    data = pd.read_csv("/data/SO_data/downvoter/lr_test_processed_data.csv")
    body_vecs = joblib.load("./final/vectorized_data/test_body_data.pkl")
    title_vecs = joblib.load("./final/vectorized_data/test_title_data.pkl")

    preds = predict_probas(data, body_vecs, title_vecs)

    labels = data.score < 0

    auc = roc_auc_score(labels, preds)
    brier = brier_score_loss(labels, preds)
    print("AUC: %.4f, BRIER: %.4f" % (auc, brier))

    print("Testing several threshold values...")

    for thr in [0.05 * i for i in range(2, 11)]:
        print("Threshold %.2f:" % thr)
        print("F-0.5 score %.4f" % fbeta_score(labels, preds > thr, 0.5))
        print(classification_report(labels, preds > thr))

