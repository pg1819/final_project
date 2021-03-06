# final_project

## Description of each file and directory:

back_translation: Contains trained classifier and extracted features using back_translation method on English-French dataset.

dataset: Contains chapters in English, and chapters translated from French, German, and Japanese.

dependency_tree: Contains trained classifier and extracted features using dependency_tree method on English-French dataset.

results: Contains txt files of results for all methods in German and Japanese chapters at document and feature-level.

template: Front-end with js for webapp

word_distribution: Contains trained classifier and extracted features using word_distribution method on English-French dataset.

word_embedding: Contains trained classifier, extracted features, and gloVe word embeddings on 2014 Wikipedia corpus.

app.py: Back-end logic of webapp

back_translation.py: Implemented back_translation method

chapter_parser.py: Script to split a book into chapters.

dependency_tree.py: Implemented dependency_tree method

evaluation_chapter.py: Script and implemented methods to perform evaluation of methods at document-level

evaluation_feature.py: Script and implemented methods to perform evaluation of methods at feature-level

highlighter.py: Implemented method to highlight text in documents for webapp

main.py: Main method called by app.py (for web-app)

script.py: Script to reproduce evaluation results from the report

util.py: Implemented methods for evaluation

word_distribution.py:  Implemented word_distribution method

word_embedding.py:  Implemented word_embedding method

## How to run web app locally
### Note: Change python to python3 and pip to pip3 if encountering any error

1. Change to directory
>cd final_project 

2. Create an environment
>python -m venv venv

3a. Activate the environment (Windows)
>venv\Scripts\activate

3b. Activate the environment (Linux/OSX)
>. venv/bin/activate

4. Install Flask within activated environment
>pip install flask

5. Install Requirements
>pip install -r requirements.txt

6a. If on CMD (Windows)
> set FLASK_APP=app.py

> flask run

6b. If on Bash (Linux/OSX)
>export FLASK_APP=app

>flask run
