"""
This script vectorizes the body and title data, using the MeanEmbeddingVectorizer
from w2v_vectorizers, for each the best word2vec model, which will be used to
train the BalancedBaggingClassifiers for body and title data on the
train + validation dataset.
"""
from w2v_vectorizers import MeanEmbeddingVectorizer
from gensim.models.word2vec import Word2Vec
from sklearn.externals import joblib
import pandas as pd
from multiprocessing import Process

def dump_body(data, path, ext):
    print("Body...")
    joblib.dump(vectorizer.transform(data.body),
                "".join([path, "body_data", ext]))

def dump_title(data, path, ext):
    print("Title...")
    joblib.dump(vectorizer.transform(data.title),
                "".join([path, "title_data", ext]))

if __name__ == "__main__":
    data = pd.concat([
        pd.read_csv("/data/SO_data/downvoter/wv_train_processed_data.csv"),
        pd.read_csv("/data/SO_data/downvoter/wv_val_processed_data.csv")
    ])
    print(data.shape)

    path = "./final/vectorized_data/"

    wv_model = Word2Vec.load("./final/word_model.w2v")

    print("Vectorizing")
    vectorizer = MeanEmbeddingVectorizer(wv_model)

    ext = ".pkl"

    p1 = Process(target = dump_body, args=(data, path, ext))
    p1.start()
    p2 = Process(target = dump_title, args=(data, path, ext))
    p2.start()
    p1.join()
    p2.join()

    print("All done!")
