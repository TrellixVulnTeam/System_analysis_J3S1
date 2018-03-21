import numpy as np
import pandas as pd
import nltk
import re
import os
import codecs
from sklearn import feature_extraction
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer

# import mpld3

files = [['textmining1.txt', 'textmining2.txt', 'textmining3.txt', 'textmining4.txt', 'textmining5.txt']]


def get_file_data(file_name):
    file = open(file_name, 'r')
    title = file.readline()
    data = file.read()
    return title, data


def transform_text(texts):
    # tokenize_and_stem: tokenizes (splits the texts into a list of its respective words (or tokens)
    #            and also stems each token)
    # tokenize_only: tokenizes the texts only
    def tokenize_and_stem(text):
        # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
        tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
        filtered_tokens = []
        # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
        for token in tokens:
            if re.search('[а-яА-Я]', token):
                filtered_tokens.append(token)
        stems = [stemmer.stem(t) for t in filtered_tokens]
        return stems

    def tokenize_only(text):
        # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
        tokens = [word.lower() for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
        filtered_tokens = []
        # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
        for token in tokens:
            if re.search('[а-яА-Я]', token):
                filtered_tokens.append(token)
        return filtered_tokens

    # stopwords = nltk.corpus.stopwords.words('russian')
    stemmer = SnowballStemmer("russian")

    totalvocab_stemmed = []
    totalvocab_tokenized = []
    for i in texts:
        allwords_stemmed = tokenize_and_stem(i)  # for each item in 'texts', tokenize/stem
        totalvocab_stemmed.extend(allwords_stemmed)  # extend the 'totalvocab_stemmed' list

        allwords_tokenized = tokenize_only(i)
        totalvocab_tokenized.extend(allwords_tokenized)

    return totalvocab_stemmed, totalvocab_tokenized


def vectorization():
    tfidf_vectorizer = TfidfVectorizer(max_df=0.8, max_features=200000,
                                       min_df=0.2, stop_words='english',
                                       use_idf=True, tokenizer=tokenize_and_stem, ngram_range=(1, 3))

    tfidf_matrix = tfidf_vectorizer.fit_transform(synopses)

def main():
    titles = []
    texts = []
    for i in files:
        for text in i:
            title, text_data = get_file_data('../files/' + text)
            titles.append(title)
            texts.append(text_data)

    totalvocab_stemmed, totalvocab_tokenized = transform_text(texts)

    vocab_frame = pd.DataFrame({'words': totalvocab_tokenized}, index=totalvocab_stemmed)
    #print('there are ' + str(vocab_frame.shape[0]) + ' items in vocab_frame')
    print(vocab_frame.head())


if __name__ == '__main__':
    main()
