import numpy as np
import pandas as pd
import nltk
import re
import os
import codecs
from sklearn import feature_extraction
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans

# import mpld3

files = [['textmining1.txt', 'textmining2.txt', 'textmining3.txt', 'textmining4.txt', 'textmining5.txt'],
         ['genetics1.txt', 'genetics2.txt', 'genetics3.txt', 'genetics4.txt', 'genetics5.txt'],
         ['physics1.txt', 'physics2.txt', 'physics3.txt', 'physics4.txt', 'physics5.txt']]

num_clusters = 3


def get_file_data(file_name):
    file = open(file_name, 'r')
    title = file.readline()
    data = file.read()
    return title, data


# tokenize_and_stem: tokenizes (splits the texts into a list of its respective words (or tokens)
#            and also stems each token)
def tokenize_and_stem(text):
    # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
    stemmer = SnowballStemmer("russian")
    tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
    for token in tokens:
        if re.search('[а-яА-Я]', token):
            filtered_tokens.append(token)
    stems = [stemmer.stem(t) for t in filtered_tokens]
    return stems


# tokenize_only: tokenizes the texts only
def tokenize_only(text):
    # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
    tokens = [word.lower() for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
    for token in tokens:
        if re.search('[а-яА-Я]', token):
            filtered_tokens.append(token)
    return filtered_tokens


def transform_text(texts):
    # stopwords = nltk.corpus.stopwords.words('russian')

    totalvocab_stemmed = []
    totalvocab_tokenized = []
    for i in texts:
        allwords_stemmed = tokenize_and_stem(i)  # for each item in 'texts', tokenize/stem
        totalvocab_stemmed.extend(allwords_stemmed)  # extend the 'totalvocab_stemmed' list

        allwords_tokenized = tokenize_only(i)
        totalvocab_tokenized.extend(allwords_tokenized)

    return totalvocab_stemmed, totalvocab_tokenized


def vectorization(data):
    tfidf_vectorizer = TfidfVectorizer(max_df=0.8, max_features=200000,
                                       min_df=0.2, stop_words='english',
                                       use_idf=True, tokenizer=tokenize_and_stem, ngram_range=(1, 3))

    tfidf_matrix = tfidf_vectorizer.fit_transform(data)
    terms = tfidf_vectorizer.get_feature_names()
    dist = 1 - cosine_similarity(tfidf_matrix)

    return tfidf_matrix


def clustering(data, n_clust):
    km = KMeans(n_clusters=n_clust)
    km.fit(data)
    clusters = km.labels_.tolist()
    return clusters


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

    vect_data = vectorization(texts)
    clusters = clustering(vect_data, num_clusters)

    p_texts = { 'title': titles, 'cluster': clusters}
    frame = pd.DataFrame(p_texts, index=[clusters], columns=['title', 'cluster'])

    print(clusters)


if __name__ == '__main__':
    main()
