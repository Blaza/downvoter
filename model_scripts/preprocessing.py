from bs4 import BeautifulSoup

import spacy
nlp = spacy.load("en", disable=["parser", "tagger", "ner"])

safe_tags = ["p", "li"] + ["h%d"%i for i in range(1, 7)]
def process_html(raw_text):
    """
    Process raw html code by taking only the text content of <p>, <li> or <h*>
    html tags, ignoring everything else.
    """
    soup = BeautifulSoup(raw_text, "html5lib")
    tags = soup(safe_tags)
    return " ".join([tag.get_text() for tag in tags])

def process_nlp(text):
    """
    Process textual data by making it all lowercase, removing punctuation and
    lemmatizing the words using the spacy nlp package.
    """
    text = text.lower()
    doc = nlp(text)
    processed = [t.lemma_.strip() for t in doc \
                 if not t.is_punct]

    return " ".join(processed)

def process_body(body):
    """
    Process the body of the question which is in html format by taking the text
    content from <p>, <li> and <h*> tags and processing the resulting text via
    the process_nlp function.
    """
    nohtml = process_html(body)

    processed = process_nlp(nohtml)

    if processed == "":
        processed = "empty_body"

    return processed

def process_title(title):
    """
    Process the title of the question which is a simple string vie process_nlp.
    """
    processed = process_nlp(title)

    if processed == "":
        processed = "empty_title"

    return processed
