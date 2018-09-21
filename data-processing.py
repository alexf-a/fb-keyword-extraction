from bs4 import BeautifulSoup as bs
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import datetime
import csv
import spacy
import nltk
import logging


TRAIN_PATH = "data/Train.csv"
TEST_PATH = "data/Test.csv"
N = 50000


def cleaned_posts(csv_file):
    '''Return a tuple of (titles, HTML-stripped full posts) from posts in csv file located at path csv_file.'''
    posts = []
    titles = []
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
            post = title + "\n" + body
            posts.append(post)
            titles.append(title)
    return (titles, posts)

def title_weights(vectorizer, titles, weights = (3, 9)):
    '''Return matrix of weights for TFIDF title-word indices, with:
        - non-named-entities receiving weight of weights[0]
        - named-entities receiving weight of weights[1]
        - all other entries having a value of 1.
    '''
    nlp = spacy.load('en')
    #anonymous functions for tokenization and named entity extraction
    tokenizer = lambda str: vectorizer.build_tokenizer(vectorizer.build_preprocessor(str))
    flatten = lambda l: [item for sublist in l for item in sublist]
    named_ents = lambda ents: flatten([tokenizer(ent.text) for ent in ents])
    #get mapping of words to TFIDF-indices
    feature_names = vectorizer.get_feature_names()
    #create matrix of ones with shape (N, len(feature_names))
    ttl_wghts = np.ones((N, len(feature_names())))
    #create mapping of words to weights...
    for i, ttl in enumerate(titles):
        ##split title into tokens and named entities
        spcy_doc = nlp(ttl)
        tknzd = tokenizer(ttl)
        ents = named_ents(spcy_doc.ents)
        ##update this vector of title weights with the appropriate weight
        for tok in tknzd:
            if tok in ents:
                try:
                    ttl_wghts[i][feature_names.index(tok)] = ttl_wghts[i][feature_names.index(tok)] * weights[1]
                except IndexError as e:
                    logging.exception(e)
                    continue
            else:
                try:
                    ttl_wghts[i][feature_names.index(tok)] = ttl_wghts[i][feature_names.index(tok)] * weights[0]
                except IndexError as e:
                    logging.exception(e)
                    continue

    return ttl_wghts





