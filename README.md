# Downvoter
StackOverflow question quality assessment. A project for the machine learning course at the Faculty of Mathematics in Belgrade.

## Quick overview of the model

The aim was to develop a model which, given a StackOverflow question, decides whether it is a 'bad' question or not. In other words, decides if the question should be downvoted. We deem the question 'bad' if it's score is negative.

The developed model has two layers.

1. Firstly, two models, one for the body of the question and one for the title, were developed which give an estimate of the probability that the given question should be downvoted, based only on the textual content of the title/body. A vector representation is created for the text by averaging the word2vec representations of the words within the text. That is then passed to a bagging classifier which scores the question.

2. On top of that, a logistic regression is trained which gives the final estimate of the question 'badness', given the features of the question:
  
    * The score from the body model from the first layer,
    * The score from the title model from the first layer,
    * The cosine distance of the vector representations of the title and the body,
    * The word count of the question body,
    * The reputation of the user posting the question.

The dataset used is the StackOverflow data available at https://archive.org/download/stackexchange (only 'Posts' and 'Users' were used). We used 'Post' data from 01.01.2016. to the end (somewhere around June 2018).

