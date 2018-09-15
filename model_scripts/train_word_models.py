"""
A script which trains word2vec models for combinations of parameters:
    vector sizes: 50, 100, 150, 200,
    window sizes: 5, 15, 30.
Results in 12 models and takes a long long time.
"""

from gensim.models.word2vec import Word2Vec, LineSentence
from multiprocess import Pool
import pandas as pd
import numpy as np
import os

print("Loading data...")
data = pd.read_csv("/data/SO_data/downvoter/wv_train_processed_data.csv").body
print(data.shape)

# save data to one line per doc file
np.savetxt("data/wdocfile.txt", data.values, fmt="%s")
tagged_data = LineSentence("data/wdocfile.txt")


max_epochs = 50
alpha = 0.025

def train_single(vec_size, window, model_file=None):
    if model_file is None:
        model_file = "word_models/word2vec.model.s%d.w%d" % (vec_size, window)
        if os.path.isfile(model_file):
            print("Model s=%d, w=%d Saved" % (vec_size, window))
            return None
        model = Word2Vec(size=vec_size,
                         alpha=alpha,
                         min_alpha=0.01,
                         min_count=25,
                         window=window,
                         workers=6)

        print("Building the vocabulary...")
        model.build_vocab(tagged_data)
    else:
        model = Word2Vec.load(model_file)
    print(len(model.wv.vocab))

    print("Starting training...")
    for epoch in range(max_epochs):
        print("Model s=%d, w=%d: iteration %d" % (vec_size, window, epoch))
        model.train(tagged_data,
                    total_examples=model.corpus_count,
                    epochs=model.iter)
        # decrease the learning rate
        model.alpha -= 0.0002
        # fix the learning rate, no decay
        model.min_alpha = model.alpha
        if epoch % 10 == 0:
            model.save(model_file)

    model.save(model_file)
    print("Model s=%d, w=%d Saved" % (vec_size, window))

    return None

print("Building the models...")
#train_single(100)#, "models/word2vec.model150")

vec_sizes = [50, 100, 150, 200]
windows = [5, 15, 30]
with Pool(6) as p:
    p.starmap(train_single, [(i,j) for i in vec_sizes for j in windows])

print("All done!")
