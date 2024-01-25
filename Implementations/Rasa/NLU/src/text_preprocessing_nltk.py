#!/usr/bin/env python

'''

Text preprocessing to prepare it for machine learning model

'''
# Text Preprocessing using nltk

from string import punctuation
from itertools import chain
import data_generation
import pandas as pd
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences


# text preprocessing

def tokenize_text_sequence(column, uncase=True):
    '''
    tokenize the questions into text sequences using nltk
    '''
    if uncase:
        column = [x.lower() for x in list(column)]
    tokens = [word_tokenize(x) for x in list(column)]
    return tokens


def remove_punctuation(tokens):
    '''
    remove the punctuation from the text sequences
    '''
    sentences = []
    for token in tokens:
        x = [word for word in token if word not in punctuation]
        sentences.append(x)
    return sentences


def lemmatize_sequence(tokens):
    '''
    lemmatize the text sequence using nltk wordnetlemmatizer library
    '''
    lm = WordNetLemmatizer()
    lem = []
    for token in tokens:
        x = [lm.lemmatize(word) for word in token]
        lem.append(x)
    return lem


def stem_sequence(tokens):
    '''stem the text sequences using porterstemmer'''
    stemmer = PorterStemmer()
    stemmed = []
    for token in tokens:
        x = [stemmer.stem(word) for word in token]
        stemmed.append(x)
    return stemmed


def remove_stopwords(tokens):
    '''
    remove the stopwords from the text sequences of english language
    '''
    stop_words = stopwords.words('english')
    sentences = []
    for token in tokens:
        x = [word for word in token if word not in stop_words]
        sentences.append(x)
    return sentences


def split_data(X, Y, test_size=0.2, val_size=0.1, random_state=42):
    '''split the data into training set, test set and validation set using the
    given parameter for the size'''
    # calculating the test size and val_size
    val_len = val_size*len(X)
    total_test_len = (test_size+val_size)*len(X)
    val_prp = val_len/total_test_len

    # training testing split
    x_train, x_test, y_train, y_test = train_test_split(
        X, Y, test_size=(test_size+val_size), random_state=random_state
    )
    x_test, x_val, y_test, y_val = train_test_split(
        x_test, y_test, test_size=val_prp)
    train = pd.DataFrame({X.name: x_train, Y.name: y_train})
    test = pd.DataFrame({X.name: x_test, Y.name: y_test})
    val = pd.DataFrame({X.name: x_val, Y.name: y_val})
    train.to_csv('../data/intent_classification/train_set.csv', index=False)
    test.to_csv('../data/intent_classification/test_set.csv', index=False)
    val.to_csv('../data/intent_classification/val_set.csv', index=False)
    return train, test, val


def get_train_set():
    '''Get the dataframe of the splited data '''
    train = pd.read_csv('../data/intent_classification/train_set.csv')
    return train


def get_test_set():
    '''Get the dataframe of the splited data '''
    test = pd.read_csv('../data/intent_classification/test_set.csv')
    return test


def get_val_set():
    '''Get the dataframe of the splited data '''
    val = pd.read_csv('../data/intent_classification/val_set.csv')
    return val


def encode_train(X, Y):
    '''
        X is the data to be fitted and Y is the data we want to transform the fitting.
        It return the Label encoder object and the transformed list

        X and Y both are tokens in the form of ['hello', 'there', 'mates']
    '''
    le = LabelEncoder()
    flatX = list(chain.from_iterable(X))
    le.fit(flatX)
    #func=lambda s : '<oov>' if s not in le.classes_ else s
    Y_dash = []
    for sentence in Y:
        s = ['<oov>' if word not in le.classes_ else word for word in sentence]
        Y_dash.append(s)
    le.classes_ = np.append(le.classes_, '<oov>')
    Y = [list(le.transform(sen)) for sen in Y_dash]
    return le, Y


def pad_encoded(encoder, encoded_sequence, max_len):
    '''pad the encoded sequence with default of post truncating and post padding '''
    return pad_sequences(encoded_sequence,
                         padding='post',
                         maxlen=max_len,
                         truncating='post',
                         value=np.where(encoder.classes_ == '<oov>'))


def preprocess_sequence(feature,
                        encode_into,
                        remove_punct=True,
                        lemmatize=True,
                        rem_stopwords=True,
                        stem=False,
                        pad=False,
                        encode=True,
                        max_pad_len=7):
    '''Pipeline for the preprocessing of the feature'''
    tokens = tokenize_text_sequence(feature)

    if remove_punct:
        tokens = remove_punctuation(tokens)
    if rem_stopwords:
        tokens = remove_stopwords(tokens)
    if lemmatize:
        tokens = lemmatize_sequence(tokens)
    if stem:
        tokens = stem_sequence(tokens)
    if encode:
        # doing the same for encoded into what we applied for the feature
        y_tokens = tokenize_text_sequence(encode_into)
        if remove_punct:
            y_tokens = remove_punctuation(y_tokens)
        if rem_stopwords:
            y_tokens = remove_stopwords(y_tokens)
        if lemmatize:
            y_tokens = lemmatize_sequence(y_tokens)
        if stem:
            y_tokens = stem_sequence(y_tokens)
        tokens = encode_train(tokens, y_tokens)
    if pad:
        tokens = pad_encoded(tokens[0], tokens[1], max_pad_len)
    return tokens

# def encode_labels():


if __name__ == '__main__':
    nltk.download('all-corpora')
    feature = data_generation.import_dataframe(
        "../data/final_data.csv")['Questions']
    #split_data(feature['Questions'], feature['Intent'])
    train = get_train_set()['Questions']
    test = get_test_set()['Questions']
    # print(train)
    preprocessed = preprocess_sequence(train, test, pad=True)
    print(preprocessed)
    print()
    print(str(len(preprocessed)) + " = "+str(len(test)))
