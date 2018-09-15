import numpy as np

def cosine_similarity(x, y):
    """
    Calculate the cosine similarity of corresponding row vectors of x and y.
    """
    return np.einsum('ij,ij->i', x, y) / (
            np.linalg.norm(x, axis=1) * np.linalg.norm(y, axis=1)
    )

def calculate_features(data,
                       body_vectors, body_model,
                       title_vectors, title_model):
    """
    Calculate the features used for the logistic regression model.
    Those are:
        - The probability prediction of the body nlp-based model,
        - The probability prediction of the title nlp-based model,
        - The cosine similarity of the title vector and the body vector,
        - The number of words in the question body,
        - The reputatio of the user posting the question.
    """
    body_preds = body_model.predict_proba(body_vectors)[:, 1]
    title_preds = title_model.predict_proba(title_vectors)[:, 1]

    cosines = cosine_similarity(body_vectors, title_vectors)
    cosines[np.isnan(cosines)] = 0

    body_word_counts = np.array([len(s.split()) for s in data["body"]])

    return np.column_stack([body_preds,
                            title_preds,
                            cosines,
                            body_word_counts,
                            data["reputation"]])
