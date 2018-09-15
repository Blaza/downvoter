"""
This is a script which trains the BalancedBagginClassifier on top of the best
word2vec model, on the train + validation dataset.
"""
from imblearn.ensemble import BalancedBaggingClassifier
from sklearn.externals import joblib
import pandas as pd

if __name__ == "__main__":
    print("Loading data")
    data = pd.concat([
        pd.read_csv("/data/SO_data/downvoter/wv_train_processed_data.csv"),
        pd.read_csv("/data/SO_data/downvoter/wv_val_processed_data.csv")
    ])
    body_data = joblib.load("./final/vectorized_data/body_data.pkl")
    title_data = joblib.load("./final/vectorized_data/title_data.pkl")

    body_model = BalancedBaggingClassifier(n_estimators=100,
                                           n_jobs=-1,
                                           ratio="not minority")

    title_model = BalancedBaggingClassifier(n_estimators=100,
                                            n_jobs=-1,
                                            ratio="not minority")

    labels = data.score < 0
    print("Fitting body model")
    body_model.fit(body_data, labels)
    print("Fitting title model")
    title_model.fit(title_data, labels)

    joblib.dump(body_model, "./final/body_model.pkl")
    joblib.dump(title_model, "./final/title_model.pkl")

    print("All done!")
