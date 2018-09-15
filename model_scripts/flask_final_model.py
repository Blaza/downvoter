# The same as final_model.py, just with relative imorts so it works in the app
from sklearn.externals import joblib
from .preprocessing import process_body, process_title
from .w2v_vectorizers import MeanEmbeddingVectorizer
from .logistic_features import calculate_features
import os

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

wv_model = joblib.load("./final/word_model.w2v")
vectorizer = MeanEmbeddingVectorizer(wv_model)

body_model = joblib.load("./final/body_model.pkl")
title_model = joblib.load("./final/title_model.pkl")

logistic_model = joblib.load("./final/logistic_model.pkl")
logistic_scaler = joblib.load("./final/logistic_scaler.pkl")

def predict_proba(question):
    """
    Estimate the probability that the given question is of low quality.

    Arguments:
    question - an object with the following attributes:
                 question.body - the body of the question, enclosed in <p> tags
                 question.title - the title of the question
                 question.reputation - the reputation of the user posting
    """
    processed_body = process_body(question["body"])
    processed_title = process_title(question["title"])

    body_vec = vectorizer.transform([processed_body])
    title_vec = vectorizer.transform([processed_title])

    features = calculate_features({"body" : [processed_body],
                                   "title" : [processed_title],
                                   "reputation" : [question["reputation"]]},
                                  body_vec, body_model,
                                  title_vec, title_model)

    scaled = logistic_scaler.transform(features)

    return logistic_model.predict_proba(scaled)[:,1]

def predict_probas(questions, body_vecs, title_vecs):
    """
    Estimate the probabilities that given questions are of low quality.

    Arguments:
    questions - pandas dataframe with the questions data
    body_vecs - vectorized form of question bodies
    title_vecs - vectorized form of question titles
    """
    features = calculate_features(questions,
                                  body_vecs, body_model,
                                  title_vecs, title_model)

    scaled = logistic_scaler.transform(features)

    return logistic_model.predict_proba(scaled)[:,1]


