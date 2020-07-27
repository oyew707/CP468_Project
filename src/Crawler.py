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
from bs4 import BeautifulSoup
import os
import requests
from random import sample
from Utilities import cleanMe


def download_pages(search): # words/phrase to search for in yahoo answers
    search = search.replace(" ","+")
    url = 'https://ca.answers.search.yahoo.com/search?p=' + search
    page = requests.get(url,timeout=10)
    soup = BeautifulSoup(page.content, 'html.parser')
    data = soup.find_all('div', {'class':"dd algo AnswrsV2"}) # get all answers - div

    for i in data:
        # get the link to the answer, category, and name
        i = i.find_all('a', href = True)
        name, site,category = i[0].text.strip(),i[0]['href'],i[1].text.strip()
        page = requests.get(site,timeout=60)
        # remove scripts and get text
        doc = cleanMe(page.content)
        # get dir, name file and save it
        directory = 'CP468_Folder/' + category.replace(" ","_")
        parent_dir = os.path.abspath(os.getcwd())
        path = os.path.join(parent_dir, directory)
        if not os.path.isdir(path): os.mkdir(path)
        name = path + "/doc_" + name[:7].replace(" ","_").replace("/","_")+".txt"
        fp = open(name, 'w')

        if len(doc.split()) >= 200:
            fp.write(doc)
        fp.close()
    return None

# random set of words to search
word_site = "http://svnweb.freebsd.org/csrg/share/dict/words?view=co&content-type=text/plain"

response = requests.get(word_site, timeout = 10)
WORDS = response.content.decode().splitlines()
search = sample(WORDS, 50) #how much to search

# download documents
for i in search:
    download_pages(i)

