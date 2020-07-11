from flask import Flask, render_template, request
import json
import pickle
import requests
import re
from bs4 import BeautifulSoup

app = Flask(__name__ ,template_folder= "CP468_Project/templates")

@app.route('/')
def index():
    categories = load_cat()
    return render_template('index.html', categories = categories)

@app.route('/results', methods=['POST'])
def my_prediction():
    categories = load_cat()
    url = request.form["searchbar"]
    doc = process(url)
    vec1 = load_featurex().transform([doc])
    model = load_model()
    vec1.resize((1, model.shape_fit_[1]))
    pred1 = categories[str(model.predict(vec1)[0])]
    doc = re.findall('................................................?', doc)
    print(doc)
    return render_template('layout.html', categories=categories, Document=doc, Category=pred1)

def process(url):
    categories = load_cat()
    page = requests.get(url,timeout=60)
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
    with open('cat.json', 'w') as fp:
        json.dump(categories, fp)
    return

def load_cat():
    with open('cat.json', 'r') as fp:
        categories = json.load(fp)
    return categories

def save_model(model):
    fp = open("my_model.pkl", "wb")
    pickle.dump(model, fp)
    fp.close()

def load_model():
    fp2 = open("my_model.pkl", "rb")
    model = pickle.load(fp2)
    return model

def save_featurex(model):
    fp = open("my_extracter.pkl", "wb")
    pickle.dump(model, fp)
    fp.close()

def load_featurex():
    fp2 = open("my_extracter.pkl", "rb")
    model = pickle.load(fp2)
    return model

if __name__ == '__main__':
    app.run(debug=True)