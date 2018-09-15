"""
The best word2vec model turned out to be of dimension 50 and context window 30.
This script is used to train the model on the train + validation dataset.
"""
from gensim.models.word2vec import Word2Vec, LineSentence
from multiprocess import Pool
import pandas as pd
import numpy as np
import os

if __name__ == "__main__":
    print("Loading data...")
    data = pd.concat([
        pd.read_csv("/data/SO_data/downvoter/wv_train_processed_data.csv").body,
        pd.read_csv("/data/SO_data/downvoter/wv_val_processed_data.csv").body
    ])
    print(data.shape)

    # save data to one line per doc file
    np.savetxt("data/wdocfile.txt", data.values, fmt="%s")
    tagged_data = LineSentence("data/wdocfile.txt")


    max_epochs = 50
    alpha = 0.025

    model_file = "./final/word_model.w2v"

    model = Word2Vec(size=50,
                     alpha=alpha,
                     min_alpha=0.01,
                     min_count=25,
                     window=30,
                     workers=16)

    print("Building the vocabulary...")
    model.build_vocab(tagged_data)
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

    print("All done!")
