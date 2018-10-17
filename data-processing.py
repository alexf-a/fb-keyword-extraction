from bs4 import BeautifulSoup as bs
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import datetime
import csv
import spacy
import nltk
import logging
import pandas as pd
import re

TRAIN_PATH = "data/Train.csv"
TEST_PATH = "data/Test.csv"
N = 20000
TOKEN_PATTERN = r'(?u)\b\w+\b'


def cleaned_text(csv_file):
    '''Return a tuple of (titles, HTML-stripped full posts, labels) from posts in csv file located at path csv_file.'''
    posts = []
    titles = []
    labs = []
    with open(csv_file, 'r') as file:
        reader = csv.reader(file)
        for i, row in enumerate(reader):
            if i == 0:
                continue
            if N != None and i > N:
                break
            title = row[1]
            soup = bs(row[2], 'html.parser')
            body = soup.get_text()
            post = title + " " + body
            posts.append(re.sub(r"[^a-zA-Z0-9]+", ' ', post))
            titles.append(re.sub(r"[^a-zA-Z0-9]+", ' ', title))
            labs.append(row[3])
    return (np.array(titles), np.array(posts), np.array(labs))

def title_weights(vectorizer, titles, weights = (3, 9)):
    '''Return matrix of weights for TFIDF title-word indices, with:
        - non-named-entities receiving weight of weights[0]
        - named-entities receiving weight of weights[1]
        - all other entries having a value of 1.
    '''
    nlp = spacy.load('en')
    #anonymous functions for tokenization and named entity extraction
    tokenizer = vectorizer.build_tokenizer()
    pprocessor = vectorizer.build_preprocessor()
    flatten = lambda l: [item for sublist in l for item in sublist]
    named_ents = lambda ents: flatten([tokenizer(ent.text) for ent in ents])
    #get mapping of words to TFIDF-indices
    feature_names = vectorizer.get_feature_names()
    #create matrix of ones with shape (N, len(feature_names))
    ttl_wghts = np.ones((N, len(feature_names)))
    #create mapping of words to weights...
    for i, ttl in enumerate(titles):
        ##split title into tokens and named entities
        ttl = str(ttl)
        spcy_doc = nlp(ttl)
        tknzd = tokenizer(pprocessor(ttl))
        ents = named_ents(spcy_doc.ents)
        ##update this vector of title weights with the appropriate weight
        for tok in tknzd:
            if tok in ents:
                try:
                    ttl_wghts[i][feature_names.index(tok)] = np.multiply(ttl_wghts[i][feature_names.index(tok)], weights[1])
                except IndexError as e:
                    logging.exception(e)
                    continue
            else:
                try:
                    ttl_wghts[i][feature_names.index(tok)] = np.multiply(ttl_wghts[i][feature_names.index(tok)], weights[0])
                except IndexError as e:
                    logging.exception(e)
                    continue

    return ttl_wghts


def clean_duplicates(X, y):
    '''Return tuple of (data, labels).
        - data: Numpy matrix formed by removing duplicates from Numpy matrix X of TFIDF word-vectors.
        - labels: Numpy matrix formed from y by removing labels for duplicates in X, with each remaining label
        y[i] being the union of all labels for duplicates of X[i]. Assumes a Bag-of-Words encoding for labels,
        where y[i][j] = 1 iff vocabulary item j appears in label i.
    '''

    # get sets of duplicate indices
    ##get reconstruction mapping
    _, reconstruct = np.unique(X, return_inverse=True, axis=0)
    ##find unique values in reconstruction mapping
    unq_inds = np.unique(reconstruct)
    ##map each unique index to tuple of indices in reconstruct that have that value
    dupe_inds = []
    for ind in unq_inds:
        dupe_inds.append(np.where(reconstruct == ind))

    # for each set of duplicate indices (i1, ... , in):
    for dupe_set in dupe_inds:
        first_ind = dupe_set[0]
        to_drop = dupe_set[1:]
        # set y[i1] = union of y[i1] ... y[in]
        y_i = np.zeros(len(y[0]))
        for dupe in dupe_set:
            y_i = y_i + y[dupe]
        y_i = np.nan_to_num(y_i, copy=False)
        # drop X[i2, ... ,in], y[i1, ... , in]
        np.delete(X, to_drop, 0)
        np.delete(y, to_drop, 0)


    return (X, y)




if __name__ == "__main__":
    train = cleaned_text(TRAIN_PATH)
    titles = train[0]
    full_posts = train[1]
    labs = train[2]
    tfidf_vectorizer = TfidfVectorizer(token_pattern=TOKEN_PATTERN)
    tfidf_vectorizer.fit(full_posts)
    bow_vectorizer = CountVectorizer(token_pattern=TOKEN_PATTERN)
    bow_vectorizer.fit(labs)
    X = tfidf_vectorizer.transform(full_posts)
    y = bow_vectorizer.transform(labs)
    X = X.toarray()
    y = y.toarray()
    X, y = clean_duplicates(X, y)
    weights = title_weights(tfidf_vectorizer, titles)
    X = np.multiply(X, weights)
    print(X.shape)
    print(y.shape)

### NOTES: X is getting converted into 0-dim by np.asanarray.