"""
A script used to evaluate the BalancedBaggingClassifiers trained on several (12)
word2vec model outputs and using 50 or 100 decision tree estimators in order
to choose the best (word2vec model, n_estimator) configuration. The criteria
which will be used for selection is AUC score and the Brier score if necessary.
"""
from imblearn.ensemble import BalancedBaggingClassifier
from sklearn.externals import joblib
from sklearn.metrics import roc_auc_score, brier_score_loss
import pandas as pd
import glob, os, re

if __name__ == "__main__":
    if os.path.isfile("./classifier_results.csv"):
        results = pd.read_csv("./classifier_results.csv")
    else:
        results = pd.DataFrame(columns=["data_file", "bb_n_est", "auc", "brier"])

    print("Loading data")

    train_data = pd.read_csv("/data/SO_data/downvoter/wv_train_processed_data.csv")
    val_data = pd.read_csv("/data/SO_data/downvoter/wv_val_processed_data.csv")
    train_labels = train_data.score < 0
    val_labels = val_data.score < 0

    # Different word2vec models vectorized data. An all encompasing list for
    # both titles and bodies. Should contain 24 elements if 12 w2v models.
    instances = [re.match("./vectorized_data/train_(.+)", f).group(1)
                  for f in glob.glob("./vectorized_data/train_*")]
    print(instances)

    # We'll get two generators, which will be 'synchronized', one for training
    # and one for vlaidation data.
    train_vecs = ((joblib.load("./vectorized_data/train_%s" % i), i)
                  for i in instances)
    val_vecs = (joblib.load("./vectorized_data/val_%s" % i)
                  for i in instances)

    # BalancedBaggingClassifier n_estimator and n_jobs params
    params = [(100, -1), (50, -1)]

    for data in train_vecs:
        val_vec = next(val_vecs) # 'sync' training and validation data
        for param in params:
            print("Processing %s %s" % (data[1], param[0]))
            bb_model = BalancedBaggingClassifier(n_estimators=param[0],
                                                 n_jobs=param[1],
                                                 ratio="not minority")
            print("Fitting...")
            bb_model.fit(data[0], train_labels)
            print("Testing...")
            preds = bb_model.predict_proba(val_vec)

            auc = roc_auc_score(val_labels, preds[:,1])
            brier = brier_score_loss(val_labels, preds[:,1])

            results = results.append({"data_file" : data[1],
                                      "bb_n_est" : param[0],
                                      "auc" : auc,
                                      "brier" : brier},
                                     ignore_index=True)
            results.to_csv("./classifier_results.csv", index=False)
            print("AUC: %.3f, BRIER: %.3f" % (auc, brier))

    results.to_csv("./classifier_results.csv", index=False)

