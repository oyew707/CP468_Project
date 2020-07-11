# imports
import requests
from bs4 import BeautifulSoup
from random import sample
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.feature_extraction import text
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.svm import SVC
from application import save_cat, save_model, save_featurex
import json
import pickle

# functions and cells
def cleanMe(html):  # from https://github.com/jamesacampbell/python-examples/blob/8dfbfc82c2aaa7fb17d0ad0086d781398d5812ae/html2plaintext-example.py
    soup = BeautifulSoup(html, "html.parser")  # create a new bs4 object from the html data loaded
    for script in soup(["script", "style"]):  # remove all javascript and stylesheet code
        script.extract()
    # get text
    text = soup.get_text(" ")
    # break into lines and remove leading and trailing space on each
    lines = (line.strip() for line in text.splitlines())
    # break multi-headlines into a line each
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    # drop blank lines
    text = '\n'.join(chunk for chunk in chunks if chunk)
    return text


def download_pages(search):  # words/phrase to search for in yahoo answers
    search = search.replace(" ", "+")
    url = 'https://ca.answers.search.yahoo.com/search?p=' + search
    page = requests.get(url, timeout=10)
    soup = BeautifulSoup(page.content, 'html.parser')
    data = soup.find_all('div', {'class': "dd algo AnswrsV2"})  # get all answers - div

    for i in data:
        # get the link to the answer, category, and name
        i = i.find_all('a', href=True)
        name, site, category = i[0].text.strip(), i[0]['href'], i[1].text.strip()
        page = requests.get(site, timeout=60)
        # remove scripts and get text
        doc = cleanMe(page.content)
        # get dir, name file and save it
        directory = 'CP468_Folder/' + category.replace(" ", "_")
        parent_dir = os.path.abspath(os.getcwd())
        path = os.path.join(parent_dir, directory)
        if not os.path.isdir(path): os.mkdir(path)
        name = path + "/doc_" + name[:7].replace(" ", "_").replace("/", "_") + ".txt"
        fp = open(name, 'w')

        if len(doc.split()) >= 200:
            fp.write(doc)
        fp.close()
    return

    # random set of words to search




word_site = "http://svnweb.freebsd.org/csrg/share/dict/words?view=co&content-type=text/plain"

response = requests.get(word_site, timeout=10)
WORDS = response.content.decode().splitlines()
search = sample(WORDS, 50)  # how much to search
print(search)
# words to search

# download documents

for i in search:
    download_pages(i)

# read documents and get their categories, save dict of the categories
import os
documents = []
print(os.getcwd())
topics = os.listdir('src/Document_Folder/')
y = []
categories = {}
n = 0
for i in topics:

    if i != '.DS_Store' and i != '.ipynb_checkpoints':
        categories[i] = n
        n += 1
        for j in os.listdir('src/Document_Folder/' + i + '/'):
            try:
                with open('src/Document_Folder/' + i + '/' + j, 'r') as fp:
                    data = fp.read()
                documents.append(data)
                y.append(n)

            except:
                print('src/Document_Folder/' + i + '/' + j)

categories = {value: key for key, value in categories.items()}
# Feature Extraction

my_words = ['question',
            'government',
            'now.',
            'join',
            'education',
            'privacy',
            '・',
            'politics',
            'sign',
            'send',
            'entertainment',
            'mobile',
            'pregnancy',
            'answers',
            'business',
            'partners',
            'recreation',
            '+',
            'get',
            '100',
            'computers',
            'parenting',
            'questions',
            'local',
            'and',
            'in',
            'sites',
            'celebrity',
            'news',
            'pets',
            'by',
            'today.',
            'health',
            'out',
            'garden',
            'mathematics',
            'ask',
            '?',
            'environment',
            'transportation',
            'consumer',
            'relationships',
            'culture',
            'beauty',
            'trending',
            'music',
            '?terms',
            'about',
            '&',
            'login',
            'adchoices',
            'movies',
            'food',
            'science',
            'reference',
            'arts',
            'family',
            'style',
            'finance',
            'weather',
            'help',
            'dining',
            'feedback',
            'asking',
            'games',
            'knowledge',
            'guidelines',
            'all',
            'international',
            'points',
            'drink',
            'social',
            'categories',
            'events',
            'yahoo',
            'electronics',
            'mail',
            'humanities',
            'your',
            'businesses',
            'travel',
            'community',
            'home',
            'internet',
            'products',
            'cars',
            'society',
            'rss',
            'sports',
            'levels',
            'leaderboard']
# Categories and web page navigation, has no real meaning
my_stop_words = text.ENGLISH_STOP_WORDS.union(my_words)

# Feature extraction : 1 - tfidf bag of words, 2 -tfidf bag of char, 3 -tfidf bag of Bi grams, 4 - bag of words
vectorizer1 = TfidfVectorizer(analyzer='word', min_df=1, stop_words=my_stop_words)
vectorizer2 = TfidfVectorizer(analyzer='char', min_df=1, stop_words=my_stop_words)
vectorizer3 = TfidfVectorizer(analyzer='char_wb', ngram_range=(1, 2), min_df=1, stop_words=my_stop_words)
vectorizer4 = CountVectorizer(strip_accents='ascii', min_df=1, stop_words=my_stop_words)
# X = vectorizer.fit_transform(documents)
# X_train, X_test, y_train, y_test = train_test_split(X, categories, test_size=0.1, random_state=0)
vec = [vectorizer1, vectorizer2, vectorizer3, vectorizer4]

# Models
def my_KNN(X_train, X_test, y_train, y_test):
    #K-nearest Neighbour using cosine-similarity
    model = KNeighborsClassifier(metric = 'cosine', algorithm = 'brute', weights = 'uniform')
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    return accuracy_score(y_test, pred), precision_score(y_test, pred, average = "weighted"), recall_score(y_test, pred, average = "weighted"), model

def my_KNN1(X_train, X_test, y_train, y_test):
    #K-nearest Neighbour using euclidean dist
    #model = KNeighborsClassifier(n_neighbors=len(set(categories)), metric = 'cosine', algorithm = 'brute', weights = 'distance')
    model = KNeighborsClassifier( weights = 'uniform')
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    return accuracy_score(y_test, pred), precision_score(y_test, pred, average = "weighted"), recall_score(y_test, pred, average = "weighted"), model

def my_MNB(X_train, X_test, y_train, y_test):
    #Multinomial Naive bayes
    clf = MultinomialNB()
    clf.fit(X_train.toarray(), y_train)
    pred = clf.predict(X_test.toarray())
    return accuracy_score(y_test, pred), precision_score(y_test, pred, average = "weighted"), recall_score(y_test, pred, average = "weighted"), clf

def my_SVM(X_train, X_test, y_train, y_test):
    svm = SVC(kernel = 'linear')
    svm.fit(X_train, y_train)
    pred = svm.predict(X_test)
    return accuracy_score(y_test, pred), precision_score(y_test, pred, average = "weighted"), recall_score(y_test, pred, average = "weighted"), svm

import warnings

warnings.filterwarnings('ignore')
for i in vec:
    X = i.fit_transform(documents)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)
    print("KNN cosine ->", my_KNN(X_train, X_test, y_train, y_test),
          "\tKNN minkowski ->", my_KNN1(X_train, X_test, y_train, y_test),
          "\tMultinomial Naive bayes ->", my_MNB(X_train, X_test, y_train, y_test),
          "\tSUpport Vector Machine ->", my_SVM(X_train, X_test, y_train, y_test), "\n" )

model = my_SVM(X_train, X_test, y_train, y_test)[-1]
save_cat(categories)
save_model(model)
save_featurex(vectorizer4)