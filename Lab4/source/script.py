import pandas as pd
import nltk
import re
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.manifold import MDS

# import mpld3

files = [['genetics1.txt', 'textmining2.txt', 'physics3.txt', 'textmining4.txt', 'physics5.txt'],
         ['textmining1.txt', 'genetics2.txt', 'genetics3.txt', 'genetics4.txt', 'textmining5.txt'],
         ['physics1.txt', 'physics2.txt', 'textmining3.txt', 'physics4.txt', 'genetics5.txt']]

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

    return tfidf_matrix, terms, dist


def clustering(data, n_clust):
    km = KMeans(n_clusters=n_clust)
    km.fit(data)
    clusters = km.labels_.tolist()
    order_centroids = km.cluster_centers_.argsort()[:, ::-1]
    return clusters, order_centroids


def visualization(clusters, titles, dist):
    MDS()
    mds = MDS(n_components=3, dissimilarity="precomputed", random_state=1)

    pos = mds.fit_transform(dist)  # shape (n_components, n_samples)

    xs, ys, zs = pos[:, 0], pos[:, 1], pos[:, 2]

    # set up colors per clusters using a dict
    cluster_colors = {0: '#1b9e77', 1: '#d95f02', 2: '#7570b3'}

    # set up cluster names using a dict
    cluster_names = {0: 'Genetic',
                     1: 'Text Mining',
                     2: 'Physic'}

    df = pd.DataFrame(dict(x=xs, y=ys, z=zs, label=clusters, title=titles))

    groups = df.groupby('label')

    #set up plot
    fig, ax = plt.subplots(figsize=(13, 7))  # set size
    ax.margins(0.05)  # Optional, just adds 5% padding to the autoscaling

    # iterate through groups to layer the plot
    # note that I use the cluster_name and cluster_color dicts with the 'name' lookup to return the appropriate color/label
    for name, group in groups:
        ax.plot(group.x, group.y, marker='o', linestyle='', ms=12,
                label=cluster_names[name], color=cluster_colors[name],
                mec='none')
        ax.set_aspect('auto')
        ax.tick_params(
            axis='x',  # changes apply to the x-axis
            which='both',  # both major and minor ticks are affected
            bottom='off',  # ticks along the bottom edge are off
            top='off',  # ticks along the top edge are off
            labelbottom='off')
        ax.tick_params(
            axis='y',  # changes apply to the y-axis
            which='both',  # both major and minor ticks are affected
            left='off',  # ticks along the bottom edge are off
            top='off',  # ticks along the top edge are off
            labelleft='off')

    ax.legend(numpoints=1)  # show legend with only 1 point

    # add label in x,y position with the label as the film title
    for i in range(len(df)):
        ax.text(df.ix[i]['x'], df.ix[i]['y'], df.ix[i]['title'], size=6)

    plt.show()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for name, group in groups:
        ax.plot(group.x, group.y, group.z, marker='o', linestyle='', ms=8,
                 label=cluster_names[name], color=cluster_colors[name],
                 mec='none')

    plt.show()



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

    vect_data, terms, dist = vectorization(texts)
    clusters, centroids = clustering(vect_data, num_clusters)

    p_texts = {'title': titles, 'cluster': clusters}
    frame = pd.DataFrame(p_texts, index=[clusters], columns=['title', 'cluster'])

    for i in range(num_clusters):
        print("Cluster %d words:" % i, end=' ')

        for ind in centroids[i, :(num_clusters + 1)]:
            print(' %s' % vocab_frame.ix[terms[ind].split(' ')].values.tolist()[0][0], end=', ')
        print('\n')

        print("Cluster %d texts:\n" % i, end='')
        for title in frame.ix[i]['title'].values.tolist():
            print(' %s' % title, end='')
        print('\n\n')

    visualization(clusters, titles, dist)


if __name__ == '__main__':
    main()
