import os
import re
import json
import nltk
import string
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import wordnet
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()
from gensim.models import Word2Vec
punctuation_set = set(string.punctuation)
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import KeyedVectors
nltk.download('stopwords')
nltk.download('wordnet')
stopwords_set = set(stopwords.words('english'))

word2vec_model = KeyedVectors.load("model/numberbatch.bin")



def expand_query_with_synonyms(query):
    expanded_query = set()
    for word in query:
        synonyms = set()
        for syn in wordnet.synsets(word):
            synonyms.update([lemma.replace("_", " ") for lemma in syn.lemma_names()])
        expanded_query.update(synonyms)
    temp = list(expanded_query)
    return " ".join(temp)


def extract_headings_and_text(text):
    headings = []
    heading_texts = []
    heading_pattern = r'\*\*(.*?)\*\*'
    headings = re.findall(heading_pattern, text, re.DOTALL)
    heading_pattern = r'\*\*(.*?)\*\*'
    heading_texts = re.findall(heading_pattern, text[2:]+"**", re.DOTALL)
    return headings, [ heading_text.strip() for heading_text in heading_texts]

def retrieve_combined(query, tfidf_vectors, word2vec_model ,tfidf_vectorizer , inverted_index,document_embeddings,document_ids):
    query = preprocess_text(query)
    query_tfidf_vector = tfidf_vectorizer.transform([query])
    tfidf_similarities = cosine_similarity(query_tfidf_vector, tfidf_vectors)

    query_results = set()
    temp = [word2vec_model[word] for word in query.split() if word in word2vec_model.key_to_index]
    query_embedding = np.mean(temp, axis=0)
    for word in query.split():
        if word in inverted_index:
            query_results.update(inverted_index[word])
    if(len(query_results)!=0):
        word2vec_similarities = cosine_similarity([query_embedding], document_embeddings[list(query_results)])
        hybrid_similarities = 0.5 * tfidf_similarities + 0.5 * word2vec_similarities
    else:
        hybrid_similarities = tfidf_similarities

    sorted_indices = np.argsort(hybrid_similarities[0])[::-1]
    sorted_results = [document_ids[idx] for idx in sorted_indices]
    return sorted_results


def preprocess_text(text):
    tokens = word_tokenize(text)
    tokens = [token.lower() for token in tokens if token not in punctuation_set]
    tokens = [token for token in tokens if token not in stopwords_set]
    tokens = [stemmer.stem(lemmatizer.lemmatize(token)) for token in tokens]
    preprocessed_document = ' '.join(tokens)
    return preprocessed_document


def load_dataset():
    documents = []
    document_ids = []
    folder_path = "Dataset"
    i=0
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, "r", encoding="utf-8") as file:
                document = file.read()
                documents.append(document)
                document_ids.append(i)
                i+=1
    return document_ids, documents











#  initilizing the model 

doc_ids , documents = load_dataset()
tfidf_vectorizer = TfidfVectorizer()
tfidf_vectors = tfidf_vectorizer.fit_transform(documents)

preprocessed_documents = [preprocess_text(doc) for doc in documents]
positional_index = defaultdict(list)
for doc_id, doc in enumerate(preprocessed_documents):
    tokens = word_tokenize(doc.lower())
    for pos, token in enumerate(tokens):
        positional_index[token].append((doc_id, pos))

document_embeddings = np.array([np.mean([word2vec_model[word] for word in doc.split() if word in word2vec_model.key_to_index], axis=0) for doc in preprocessed_documents])
inverted_index = defaultdict(list)

for doc_id, embedding in zip(doc_ids, document_embeddings):
    for word in preprocessed_documents[doc_id]:
        inverted_index[word].append(doc_id)


#demo run
ask_query()







def ask_query():
    query = "What is taj mahal in agra"
    query = input()
    retrieved_documents = retrieve_combined(query, tfidf_vectors, word2vec_model ,tfidf_vectorizer , inverted_index,document_embeddings,doc_ids)[:3]
    json_array = []
    for x in retrieved_documents:
        text = documents[x]
        headings, heading_texts = extract_headings_and_text(text)
        data = {}
        for heading, text in zip(headings, heading_texts):
            data[heading] = text
        json_data = json.dumps(data)
        json_array.append(json_data)

    print(json_array)