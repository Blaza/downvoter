"""
This script vectorizes the body and title data, using the MeanEmbeddingVectorizer
from w2v_vectorizers, using the chosen best word2vec model for the datasets used
for the final logistic model..
"""
from w2v_vectorizers import MeanEmbeddingVectorizer
from gensim.models.word2vec import Word2Vec
from sklearn.externals import joblib
import pandas as pd
from multiprocessing import Process

def dump_body(data, path, prefix, ext):
    print("Body %s data..." % prefix)
    print(data.shape)
    joblib.dump(vectorizer.transform(data.body),
                "".join([path, prefix, "_body_data", ext]))

def dump_title(data, path, prefix, ext):
    print("Title %s data..." % prefix)
    print(data.shape)
    joblib.dump(vectorizer.transform(data.title),
                "".join([path, prefix, "_title_data", ext]))

if __name__ == "__main__":
    data_list = [
        pd.read_csv("/data/SO_data/downvoter/lr_train_processed_data.csv"),
        pd.read_csv("/data/SO_data/downvoter/lr_val_processed_data.csv"),
        pd.read_csv("/data/SO_data/downvoter/lr_test_processed_data.csv")
    ]
    prefixes = ["train", "val", "test"]

    path = "./final/vectorized_data/"

    wv_model = Word2Vec.load("./final/word_model.w2v")

    print("Vectorizing")
    vectorizer = MeanEmbeddingVectorizer(wv_model)

    ext = ".pkl"

    processes = [Process(target = dumper, args=(data, path, prefix, ext))
                 for dumper in [dump_body, dump_title]
                 for (data, prefix) in zip(data_list, prefixes)]
    for p in processes:
        p.start()
    for p in processes:
        p.join()

    print("All done!")
