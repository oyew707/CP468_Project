from flask import Flask, render_template, request
from src.Utilities import load_cat, load_model, process, load_featurex
import re
import os

root_dir = os.path.dirname(os.getcwd())
template_dir = os.path.join(root_dir,'CP468_Project/templates' )
static_dir = os.path.join(root_dir,'CP468_Project/static' )
application = Flask(__name__, template_folder= template_dir, static_folder= static_dir)



@application.route('/')
def index():
    categories = load_cat()
    categories = {key: value.replace("_", " ") for key, value in categories.items()}
    return render_template('main.html', categories = categories)

@application.route('/about')
def model_info():
    return render_template("about.html", dir = str(template_dir))

@application.route('/results', methods=['GET', 'POST'])
def my_prediction():
    categories = load_cat()
    categories = {key: value.replace("_", " ") for key, value in categories.items()}
    url = request.form["searchbar"]
    doc = process(url)
    if doc != "Error: Cannot read URL/Webpage":
        vec1 = load_featurex().transform([doc])
        model = load_model()
        pred1 = categories[str(model.predict(vec1)[0]-1)]
        doc = re.findall('................................................?', doc)
    else:
        doc = [doc]
        pred1 = ""
    return render_template('layout.html', categories=categories, Document=doc, Category=pred1)







