import json
import os
import re
import pandas as pd
import spacy
from spacy.language import Language
from spacy_language_detection import LanguageDetector


def get_lang_detector(nlp, name):
    return LanguageDetector(seed=42)  # We use the seed 42


path_to_json = 'data/'
json_files = [pos_json for pos_json in os.listdir(path_to_json) if pos_json.endswith('.json')]

# Get downloaded terms of services
data = []
for json_file in json_files:
    with open(path_to_json + json_file, 'r') as f:
        data.append(json.load(f))

reviewed_terms = []
for doc in data:
    if doc['parameters']['is_comprehensively_reviewed'] is True:
        reviewed_terms.append(doc)

# Create the summaries by merging the quotes
final_data = []
nlp_model = spacy.load('en_core_web_sm')
Language.factory("language_detector", func=get_lang_detector)
nlp_model.add_pipe('language_detector', last=True)

for doc in reviewed_terms:
    legal_contracts = {}

    for point in doc['parameters']['points']:
        if point['quoteStart'] is not None and point['quoteText'] is not None:
            print(point)
            legal_contracts.setdefault(point['document_id'], []).append(point)

    for doc_id, value in legal_contracts.items():
        legal_contracts[doc_id] = sorted(value, key=lambda i: i['quoteStart'])
        print(doc_id)
        plain_text = ""
        summary = ""
        for point in legal_contracts[doc_id]:
            plain_text += point['quoteText'] + " "
            summary += point['title'] + ". "

        # Regex preprocessing
        plain_text = re.sub(r"<[^>]*>", '', plain_text)
        plain_text = re.sub(r"\n", ' ', plain_text)

        # Language check
        doc = nlp_model(plain_text)
        if doc._.language['language'] == 'en':
            final_data.append([plain_text, summary])

df = pd.DataFrame(final_data, columns=['plain_text', 'summary'])
df.to_json('dataset.json', orient='records', lines=True)