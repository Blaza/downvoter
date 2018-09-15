"""
This script vectorizes the body and title data, using the MeanEmbeddingVectorizer
from w2v_vectorizers, for each of the 12 trained word2vec models, to be used for
evaluation of BalancedBaggingClassifiers trained on top of them.
"""

from w2v_vectorizers import MeanEmbeddingVectorizer
from gensim.models.word2vec import Word2Vec
from sklearn.externals import joblib
from multiprocess import Pool
import pandas as pd
import glob, os

if __name__ == "__main__":
    train_data = pd.read_csv("/data/SO_data/downvoter/wv_train_processed_data.csv")
    val_data = pd.read_csv("/data/SO_data/downvoter/wv_val_processed_data.csv")

    wv_models = [Word2Vec.load(f) for f in glob.glob("./word_models/*.model*")]
    path = "/data/SO_data/downvoter/vectorized_data/"

    def process_model(wv_model):
        size = wv_model.vector_size
        window = wv_model.window

        print("Vectorizing s=%d, w=%d" % (size, window))
        vectorizer = MeanEmbeddingVectorizer(wv_model)

        ext = ".w2v.s%d.w%d.pkl" % (size, window)

        print("Body train set...")
        if not os.path.isfile("".join([path, "train_body", ext])):
            joblib.dump(vectorizer.transform(train_data.body),
                        "".join([path, "train_body", ext]))
        print("Body val set...")
        if not os.path.isfile("".join([path, "val_body", ext])):
            joblib.dump(vectorizer.transform(val_data.body),
                        "".join([path, "val_body", ext]))

        print("Title train set...")
        if not os.path.isfile("".join([path, "train_title", ext])):
            joblib.dump(vectorizer.transform(train_data.title),
                        "".join([path, "train_title", ext]))
        print("Title val set...")
        if not os.path.isfile("".join([path, "val_title", ext])):
            joblib.dump(vectorizer.transform(val_data.title),
                        "".join([path, "val_title", ext]))

    with Pool(4) as p:
        p.map(process_model, wv_models)

    print("All done!")
