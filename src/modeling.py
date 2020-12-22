# imports
import numpy as np
import os
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.feature_extraction import text
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
import pandas as pd
import warnings
from math import floor
from sklearn.model_selection import learning_curve
import seaborn as sns
from scipy import sparse
from Utilities import save_model, save_cat, save_featurex

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



# read documents and get their categories, save dict of the categories

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
# my_words= ['question','government','now.','join','education','privacy','・','politics',
#            'entertainment','mobile','pregnancy','answers','business','partners','recreation','send',
#            'computers','parenting','questions','local','and','in','sites','celebrity','news','sign',
#            'pets','by','today.','health','out','garden','mathematics','ask','?','environment','get',
#            'transportation','consumer','relationships','culture','beauty','trending','music','+',
#            '?terms','about','&','login','adchoices','movies','food','science','reference','arts','100',
#            'family','style','finance','weather','help','dining','feedback','asking','games','knowledge',
#            'guidelines','all','international','points','drink','social','categories','events','yahoo',
#            'electronics','mail','humanities','your','businesses','travel','community','home','internet',
#            'products','cars','society','rss','sports','levels','leaderboard','terms', 'today']
# Categories and web page navigation, has no real meaning
my_words = ['adchoices','rss','terms', 'today','question','yahoo', 'help', 'about answers', 'community guidelines',
            "leaderboard", "knowledge partners", "points", "levels", "feedback", "international sites"]
my_stop_words = text.ENGLISH_STOP_WORDS.union(my_words)

# Feature extraction : 1 - tfidf bag of words, 2 -tfidf bag of char, 3 -tfidf bag of Bi grams, 4 - bag of words
vectorizer1 = TfidfVectorizer(analyzer = 'word' ,min_df = 1,  stop_words = my_stop_words)
#vectorizer2 = TfidfVectorizer(analyzer = 'char' ,min_df = 1, stop_words = my_stop_words)
#vectorizer3 = TfidfVectorizer(analyzer = 'char_wb',ngram_range = (1,2) ,min_df = 1,  stop_words = my_stop_words)
vectorizer4 = CountVectorizer(strip_accents='ascii',min_df = 1,  stop_words = my_stop_words)
vec = [vectorizer1, vectorizer4]
for i in list(range(1000,5000,1000)) + list(range(5000,25000,5000)):
    vectorizer1 = TfidfVectorizer(analyzer = 'word' ,min_df = 1,  stop_words = my_stop_words, max_features = i)
    #vectorizer2 = TfidfVectorizer(analyzer = 'char' ,min_df = 1, stop_words = my_stop_words,  max_features = i)
    #vectorizer3 = TfidfVectorizer(analyzer = 'char_wb',ngram_range = (1,2) ,min_df = 1,  stop_words = my_stop_words,  max_features = i)
    vectorizer4 = CountVectorizer(strip_accents='ascii',min_df = 1,  stop_words = my_stop_words, max_features = i)
    vec += [vectorizer1, vectorizer4]

warnings.filterwarnings("ignore")
data = []
for i in vec:
    X = i.fit_transform(documents)
    c = cross_val_score(KNeighborsClassifier(metric = 'cosine', algorithm = 'brute', weights = 'uniform'), X, y, cv=5).mean()
    d = cross_val_score(KNeighborsClassifier( weights = 'uniform'), X, y, cv=5).mean()
    e = cross_val_score(MultinomialNB(), X, y, cv=5).mean()
    f = cross_val_score(SVC(kernel='linear'), X, y, cv=5).mean()
    vect = "TFIDF Word" if vec.index(i)%2 == 0 else "Bag of Words"
    data += [["KNN C", c, vect, i.get_params()["max_features"]] , ["KNN M", d, vect, i.get_params()["max_features"]] , ["MNB", e, vect, i.get_params()["max_features"]] , ["SVM", f, vect, i.get_params()["max_features"]] ]
df = pd.DataFrame(data, columns = ["Model", "Cross_Val_score", "Vector", "Max_Feature"])
df = df.fillna("23000+")

sns.lineplot(x = "Model", y = "Cross_Val_score", hue = "Max_Feature", data =df[df["Vector"] == "Bag of Words"])
sns.lineplot(x = "Model", y = "Cross_Val_score", hue = "Max_Feature", data =df[df["Vector"] == "TFIDF Word"])
df.sort_values(by='Cross_Val_score', ascending=False).head(10)

warnings.filterwarnings("ignore")
X = vec[floor(15/4)].transform(documents)
train_sizes, train_scores, valid_scores = learning_curve(SVC(kernel = 'linear'), X, y, train_sizes=list(range(50,550,50)), cv=5)
ax = sns.lineplot(x = list(range(50,550,50)), y = np.mean(train_scores, axis=1))
ax = sns.lineplot(x = list(range(50,550,50)), y = np.mean(valid_scores, axis=1), ax = ax, legend = "brief")
ax

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state = 8)
accuracy_s, precision_s, recall_s, model = my_SVM(X_train, X_test, y_train, y_test)
print("Support Vector Machine ->", accuracy_s, precision_s, recall_s)


# Phase II delivarables
#Save
sparse.save_npz("vector_matrix.npz", X)
#Load
#X = sparse.load_npz("vector_matrix.npz")PaP

save_cat(categories)
save_model(model)
save_featurex(vec[floor(15/4)])