import nltk
import numpy as np
import math
import pickle
import sys
from bs4 import BeautifulSoup as bsoup

punctuations = ['.', ',', '!', '\'', '\"',
                '(', ')', '[', ']', '{', '}', '?', '\\', '/', '~', '|', '<', '>']

# For printing the whole npy array
np.set_printoptions(threshold=sys.maxsize)


def build_vocabulary_freqdist(vocabulary):
    vocabulary = nltk.FreqDist(vocabulary)
    vocabulary = sorted(vocabulary.items(), key=lambda x: x[0])
    vocabulary = dict(vocabulary)
    return vocabulary


def build_database_vocabulary(filename):
    with open(filename) as f:
        text = f.read().replace('\n', ' ')
    database = []
    vocabulary = []
    doc_count = 0
    i = 0
    doc_titles = []

    while(i < len(text)):

        if(text[i] == '<' and text[i+1] == 'd' and text[i+2] == 'o' and text[i+3] == 'c'):

            # keep finding the end of the doc begin tag
            flag = 0
            title = ""

            while(text[i] != '>'):
                i = i+1
                if(text[i-6] == 't' and text[i-5] == 'i' and text[i-4] == 't' and text[i-3] == 'l' and text[i-2] == 'e' and text[i-1] == '='):
                    flag = 1

                if(flag == 1 and text[i] != '>'):
                    title = title + str(text[i])

            doc_titles.append(title)

            i = i + 2

            document = ''

            while(not (text[i] == '<' and text[i+1] == '/' and text[i+2] == 'd' and text[i+3] == 'o')):
                document = document + str(text[i])
                i = i+1

            while(text[i] != '>'):
                i = i+1

            doc_count = doc_count + 1

            if(doc_count % 200 == 0):
                print("Document " + str(doc_count) + " is being read...")

            # pre-processing
            clean_document = bsoup(document, 'html.parser').get_text()
            clean_document = clean_document.lower()
            # tokens is a list of tokens
            tokens = nltk.word_tokenize(clean_document)
            tokens = [token for token in tokens if token not in punctuations]

            for token in tokens:
                vocabulary.append(token)

            # saving the file / document wise write-back
            # database is a list of list, as mentioned above
            database.append(tokens)

        i = i + 1

    print("Done with the document reading...Database is ready \n")
    print("There are total " + str(doc_count) + " documents!")
    print("There are totla " + str(len(doc_titles)) + " titles!")

    # if(doc_count > 3):
    #     break

    # vocabulary is a dictionary of words/tokens and their corpus frequencies
    vocabulary = build_vocabulary_freqdist(vocabulary)

    print("Done with the Vocabulary building...Vocab is ready \n")

    with open("vocabulary_dict.pkl", "wb") as f:
        pickle.dump(vocabulary, f)

    with open("doc_titles.pkl", "wb") as f:
        pickle.dump(doc_titles, f)

    # print(vocabulary)
    return database, vocabulary


def build_documents_vector(database, vocabulary_words, inverse_vocab_word_dict):
    documents_vector = np.zeros((len(database), len(vocabulary_words)))
    # print("----------------------------------------------")
    # print("----------------------------------------------")
    # print(documents_vector[0])

    inverse_vocab_word_dict = {k: v for v, k in enumerate(vocabulary_words)}

    # populate the documents_vector with the frequency of each vocabulary word for each document
    for doc_id, doc in enumerate(database):
        if(doc_id % 200 == 0):
            print("Reached Doc " + str(doc_id))
        for token in doc:
            documents_vector[doc_id][inverse_vocab_word_dict[token]
                                     ] = documents_vector[doc_id][inverse_vocab_word_dict[token]] + 1

    print("Done with the Docvec build... \n")
    np.save("documents_vector.npy", documents_vector)
    return documents_vector


def process_query_vector(query, vocabulary_keys, inverse_vocab_word_dict, term_document_frequency, N):
    query_text = ''
    for token in query:
        query_text = query_text + ' ' + str(token)
    query = query_text
    print(query)
    query = query.lower()
    query = nltk.word_tokenize(query)
    query_vector = np.zeros(len(vocabulary_keys))

    for token in query:
        if(token not in inverse_vocab_word_dict):
            print("No results found for the given query")
            exit(0)
        query_vector[inverse_vocab_word_dict[token]
                     ] = query_vector[inverse_vocab_word_dict[token]] + 1

    for i in range(query_vector.shape[0]):
        if(query_vector[i] > 0):
            query_vector[i] = 1 + math.log(query_vector[i])

    for i in range(query_vector.shape[0]):
        if(query_vector[i] == 0):
            continue
        query_vector[i] = query_vector[i] * \
            math.log(N/term_document_frequency[i])

    temp_query_vector = np.copy(query_vector)
    temp_query_vector = np.square(temp_query_vector)
    temp_query_vector = np.sum(temp_query_vector)
    temp_query_vector = np.sqrt(temp_query_vector)
    query_vector = np.divide(query_vector, temp_query_vector)
    return query_vector


def calculate_score(query_vector, documents_vector):
    scores = {}
    for id, document_vector in enumerate(documents_vector):
        score = np.dot(query_vector, document_vector)
        score = score/np.linalg.norm(query_vector)
        score = score/np.linalg.norm(document_vector)
        scores[id] = score
    scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    scores = dict(scores)
    return scores


def process_documents_vector(documents_vector):
    for i in range(0, documents_vector.shape[0]):
        for j in range(0, documents_vector.shape[1]):
            if(documents_vector[i][j] > 0):
                documents_vector[i][j] = 1 + math.log(documents_vector[i][j])
    print("Done with the log calc build... \n")

    # print("----------------------------------------------")
    # print("----------------------------------------------")
    # print(documents_vector[0])

    # calculation of cosine normalisation
    temp_documents_vector = np.copy(documents_vector)
    temp_documents_vector = np.square(temp_documents_vector)
    temp_documents_vector = np.sum(temp_documents_vector, axis=1)
    temp_documents_vector = np.sqrt(temp_documents_vector)
    documents_vector = np.divide(documents_vector, temp_documents_vector[0])

    print("Done with the cosine calc build... \n")
    return documents_vector


def scoring(query, documents_vector, vocabulary_keys, inverse_vocab_word_dict, term_document_frequency, N, doc_titles):
    query_vector = process_query_vector(
        query, vocabulary_keys, inverse_vocab_word_dict, term_document_frequency, N)
    scores = calculate_score(query_vector, documents_vector)
    scores = [(key, value) for key, value in scores.items()]
    print("Top 10 Scoring Documents are: ")

    for ind in range(10):
        if(scores[ind][1] != 0):
            print(doc_titles[scores[ind][0]] + " is at rank " +
                  str(ind+1) + " Score: " + str(scores[ind][1]))


def index_construction(filename):
    database, vocabulary = build_database_vocabulary(filename)
    # how many unique words
    vocabulary_words = list(vocabulary.keys())
    inverse_vocab_word_dict = {k: v for v, k in enumerate(vocabulary_words)}
    documents_vector = build_documents_vector(
        database, vocabulary_words, inverse_vocab_word_dict)
    documents_vector = process_documents_vector(documents_vector)
    np.save("database_lnc.npy", documents_vector)
    print("Saved! database_lnc.npy")
    return database, vocabulary


def main():
    # database, vocabulary = index_construction(sys.argv[1])

    documents_vector = np.load("documents_vector.npy")
    N = len(documents_vector)
    vocabulary_dict = pickle.load(open("vocabulary_dict.pkl", "rb"))
    # print(vocabulary_dict)
    vocabulary_keys = list(vocabulary_dict.keys())
    inverse_vocab_word_dict = {k: v for v, k in enumerate(vocabulary_keys)}
    # print(inverse_vocab_word_dict)
    term_document_frequency = np.count_nonzero(documents_vector, axis=0)
    doc_titles = pickle.load(open("doc_titles.pkl", "rb"))

    scoring(sys.argv[2:], documents_vector, vocabulary_keys,
            inverse_vocab_word_dict, term_document_frequency, N, doc_titles)


if __name__ == "__main__":
    if(len(sys.argv) < 3):
        print("Incorrect number of arguements")
        exit(-1)
    main()
