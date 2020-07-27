"""
------------------------------------------------------------------------
[program description]
------------------------------------------------------------------------
Author: Einstein Oyewole
ID:     180517070
Email:  oyew7070@mylaurier.ca
__updated__ = "2020-04-20"
------------------------------------------------------------------------
"""
# Imports
import json
from bs4 import BeautifulSoup
import pickle
import requests


def process(url):
    page = requests.get(url,timeout=60)
    if page.status_code != 200:
        doc = "Error: Cannot read URL/Webpage"
    else:
        doc = cleanMe(page.content)
    return doc


def cleanMe(html):
    soup = BeautifulSoup(html, "html.parser") # create a new bs4 object from the html data loaded
    for script in soup(["script", "style"]): # remove all javascript and stylesheet code
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


def save_cat(categories):
    with open('src/cat.json', 'w') as fp:
        json.dump(categories, fp)
    return


def load_cat():
    with open('src/cat.json', 'r') as fp:
        categories = json.load(fp)
    return categories


def save_model(model):
    fp = open("src/my_model.pkl", "wb")
    pickle.dump(model, fp)
    fp.close()


def load_model():
    fp2 = open("src/my_model.pkl", "rb")
    model = pickle.load(fp2)
    return model


def save_featurex(model):
    fp = open("src/my_extracter.pkl", "wb")
    pickle.dump(model, fp)
    fp.close()


def load_featurex():
    fp2 = open("src/my_extracter.pkl", "rb")
    model = pickle.load(fp2)
    return model


