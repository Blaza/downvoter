# Downvoter
StackOverflow question quality assessment.

A project for the machine learning course at the Faculty of Mathematics in Belgrade. ([kratka prezentacija](https://blaza.github.io/downvoter))

## Quick overview of the model

The aim was to develop a model which, given a StackOverflow question, decides whether it is a 'bad' question or not. In other words, decides if the question should be downvoted. We'll deem the question 'bad' if it's score is negative.

The developed model has two layers.

1. Firstly, two models, one for the body of the question and one for the title, were developed which give an estimate of the probability that the given question should be downvoted, based only on the textual content of the title/body. A vector representation is created for the text by averaging the word2vec representations of the words within the text. That is then passed to a bagging classifier which scores the question.

2. On top of that, a logistic regression is trained which gives the final estimate of the question 'badness', given the features of the question:
  
    * The score from the body model from the first layer,
    * The score from the title model from the first layer,
    * The cosine distance of the vector representations of the title and the body,
    * The word count of the question body,
    * The reputation of the user posting the question.

The dataset used is the StackOverflow data available at https://archive.org/details/stackexchange (only 'Posts' and 'Users' were used). I used 'Post' data from 01.01.2016. to the end (somewhere around June 2018).

The dataset used is very large and also the models created are quite big in size (~800MB) and so I didn't upload them to Github, given the finiteness of the human lifespan and the limits of the upload bandwidth.

Only the scripts which were used to train and arrive at the models are commited and are located in the `model_scripts`directory. The scripts were run in a sequence like this:

    process_data -> train_word_models -> vectorize_classifier_data -> eval_classifiers ->
    train_best_word_model -> vectorize_best_classifier_data -> train_best_classifier ->
    vectorize_logistic_data -> eval_logistic -> train_best_logistic

## A demo app

I developed a simple Flask web app to demonstrate the decisions made by the model.  

It is available at http://downvoter.duckdns.org:8387 and the usage is simple:

First, you write a question just like you would on StackOverflow, i.e. write a title, write the body (markdown supported), and possibly add the StackOverflow reputation of the user posting.
Then click on the button "Rate the question!" and the downvoter will think about the question and give it's verdict, along with the "badness score", which is the model's probability estimate (the threshold value is set at 0.275).

***OR***

The easiest way is to click on "Fetch from StackOverflow?", paste a StackOverflow URL of a question and click on "Fetch" which will populate all fields in a couple of seconds. Then run "Rate the question!" and wait for the results.

