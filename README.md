# Web page Classifier 

## Motivation and Intro

The aim of the project is to investigate the process of classifying web pages by applying data classification techniques for automatic categorization. This would help in filtering out the responses of a search engine or ranking web pages according to their relevance to a topic specified by the user.

Usually, some of the pages returned are not tagged with category topics. This is where classification techniques come into play. By using the collection of pages available under each topic as examples, we can create category descriptions. Then using these descriptions, we can classify new web pages to one of the existing topics. Another approach by using some metric over text documents, this could help to find the closest document and assign its category to the new web page.

We treated this webpage classification as a problem in supervised learning. A training data set is not available, so we collected some data by searching random words on yahoo answers. We chose to use yahoo answers to collect data as each question on the website is already categorized. This allows us to have an accurate data set and allows us to train and test our model with ease.

## Description

We decided to write a Web Crawler in Python to get data. we chose Yahoo Answers, where the web pages were already categorized so we chose to use some of the categories on the yahoo answers website. We also chose to combine categories that we felt were alike like “Dining Out” and “Food & Drink”. We also removed some categories like “Yahoo Products” as we felt it was too specific of a topic and did not apply to the majority of the web. We used yahoo answers as it had a wide range of topics which are at different levels. We have 21 topics in total and each topic has a minimum 10 documents and each document has at least 100 words.

For the modeling process, We decided to use TFIDF and Bag of Word for Feature extraction and chose to use Support Vector Machine as our supervised learning approach. Training Set is 90%; Test set is 10% of data; Used python function to distribute data at random.

## Usage

The model is currently hosted on an EC2 instance and can be accessed through http://ec2-3-97-109-119.ca-central-1.compute.amazonaws.com:5000/

To use, insert a url of a webpage into the input on the navigation bar to categorize it 
