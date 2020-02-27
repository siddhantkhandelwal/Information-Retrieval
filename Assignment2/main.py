import nltk
import numpy as np
import math
import pickle
import sys
from bs4 import BeautifulSoup as bsoup
from spellchecker import SpellChecker
from nltk.stem import PorterStemmer
ps = PorterStemmer()

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
    print("There are total " + str(len(doc_titles)) + " titles!")

    # if(doc_count > 3):
    #     break

    # vocabulary is a dictionary of words/tokens and their corpus frequencies
    vocabulary = build_vocabulary_freqdist(vocabulary)

    print("Done with the Vocabulary building...Vocab is ready and saved !\n")

    with open("vocabulary_dict.pkl", "wb") as f:
        pickle.dump(vocabulary, f)

    with open("doc_titles.pkl", "wb") as f:
        pickle.dump(doc_titles, f)

    stem_the_vocab(vocabulary)

    # print(vocabulary)
    return database, vocabulary


def build_documents_vector(database, vocabulary_words, inverse_vocab_word_dict):
    documents_vector = np.zeros((len(database), len(vocabulary_words)))

    # populate the documents_vector with the frequency of each vocabulary word for each document
    for doc_id, doc in enumerate(database):
        for token in doc:
            documents_vector[doc_id][inverse_vocab_word_dict[token]
                                     ] = documents_vector[doc_id][inverse_vocab_word_dict[token]] + 1
    print("Done with the Documents Vector build... Saving it as numpy file \n")
    np.save("documents_vector.npy", documents_vector)
    return documents_vector


def spell_correct(query):
    spell = SpellChecker()
    misspelled = spell.unknown(query.split())
    if misspelled:
        for word in query.split():
            if word in misspelled:
                print("Correcting " + word + " to " + spell.correction(word))
                query = query.replace(word, spell.correction(word))
    return query


def stem_the_vocab(vocabulary_dict):

    vocabulary_keys = list(vocabulary_dict.keys())
    vocabulary_values = list(vocabulary_dict.values())

    porter = nltk.PorterStemmer()
    stemmed_vocab = {}

    for word_id, word in enumerate(vocabulary_keys):
        if porter.stem(word) not in stemmed_vocab:
            stemmed_vocab[porter.stem(word)] = (
                word, vocabulary_values[word_id])
        else:
            if stemmed_vocab[porter.stem(word)][1] < vocabulary_values[word_id]:
                stemmed_vocab[porter.stem(word)] = (
                    word, vocabulary_values[word_id])

    with open("stemmed_vocab.pkl", "wb") as f:
        pickle.dump(stemmed_vocab, f)


def get_stemmed_token(token):
    porter = nltk.PorterStemmer()
    return porter.stem(token)


def process_query_vector(query, vocabulary_keys, inverse_vocab_word_dict, term_document_frequency, N):
    query_text = ''
    for token in query:
        query_text = query_text + ' ' + str(token)
    query = query_text
    query = query.lower()

    # bonus heuristic
    print("Running Spell Check")
    query = spell_correct(query)

    query_vector = np.zeros(len(vocabulary_keys))
    query = nltk.word_tokenize(query)

    if(len(query) == 1 and query[0] not in inverse_vocab_word_dict):
        print(
            query[0] + " is not found in vocabulary. Using most appropriate substitution using root word analysis! ")
        stemmed_token = get_stemmed_token(query[0])

        # Now the query contains all tokens as it is, only the ones that do not excist
        # in the vocabulary are replaced by their stemmed root versions
        stemmed_vocab = pickle.load(open("stemmed_vocab.pkl", "rb"))
        if(stemmed_token not in stemmed_vocab):
            print("Could not replace, no search results found")
            exit(0)
        fixed_token = stemmed_vocab[token][0]
        print("Did you mean " + fixed_token + "? Press y for yes: ")
        choice = input()
        if choice != 'y':
            print("No search results found")
            exit(0)
        query[query.index(token)] = fixed_token
        # Query has the wrong word replaced by the most common rooted word (with the same root as the wrong word)
        print("Changed query: " + ' '.join(query))
        query_vector[inverse_vocab_word_dict[fixed_token]
                     ] = query_vector[inverse_vocab_word_dict[fixed_token]] + 1
    elif(len(query) >= 1):
        for token in query:
            if(token not in inverse_vocab_word_dict):
                print(token + " not found in vocabulary.")
                stemmed_token = get_stemmed_token(token)
                stemmed_vocab = pickle.load(open("stemmed_vocab.pkl", "rb"))
                if(stemmed_token not in stemmed_vocab):
                    continue
                fixed_token = stemmed_vocab[stemmed_token][0]
                print("Did you mean " + fixed_token + "? Press y for yes: ")
                choice = input()
                if choice != 'y':
                    print("Skipping " + token)
                    continue
                query[query.index(token)] = fixed_token
                print("Changed query: " + ' '.join(query))
                query_vector[inverse_vocab_word_dict[fixed_token]
                             ] = query_vector[inverse_vocab_word_dict[fixed_token]] + 1
            else:
                query_vector[inverse_vocab_word_dict[token]
                             ] = query_vector[inverse_vocab_word_dict[token]] + 1

    # processing log calculations (L)
    for i in range(query_vector.shape[0]):
        if(query_vector[i] > 0):
            query_vector[i] = 1 + math.log(query_vector[i])

    # processing term normalization (T)
    for i in range(query_vector.shape[0]):
        if(query_vector[i] == 0):
            continue
        query_vector[i] = query_vector[i] * \
            math.log(N/term_document_frequency[i])

    # Cosine normalization (C)
    temp_query_vector = np.copy(query_vector)
    temp_query_vector = np.square(temp_query_vector)
    temp_query_vector = np.sum(temp_query_vector)
    temp_query_vector = np.sqrt(temp_query_vector)
    query_vector = np.divide(query_vector, temp_query_vector)

    return query, query_vector


def calculate_score(query_vector, database_lnc):

    scores = []
    for id, document_vector in enumerate(database_lnc):
        score = np.dot(query_vector, document_vector)
        score = score/np.linalg.norm(query_vector)
        score = score/np.linalg.norm(document_vector)
        scores.append([id, score])
    return scores


def process_documents_vector(documents_vector):

    for i in range(0, documents_vector.shape[0]):
        for j in range(0, documents_vector.shape[1]):
            if(documents_vector[i][j] > 0):
                documents_vector[i][j] = 1 + math.log(documents_vector[i][j])

    print("Done with the log calculation build... \n")

    # calculation of cosine normalisation
    temp_documents_vector = np.copy(documents_vector)
    temp_documents_vector = np.square(temp_documents_vector)
    temp_documents_vector = np.sum(temp_documents_vector, axis=1)
    temp_documents_vector = np.sqrt(temp_documents_vector)
    documents_vector = np.divide(
        documents_vector, temp_documents_vector[:, None])

    print("Done with the cosine normalization build... \n")
    return documents_vector


def title_weighting(scores, query, doc_titles):

    title_weight = 0.1
    trivial_words = ["of", "and", "a", "the", "an", "is"]

    for doc_id, doc_title in enumerate(doc_titles):
        count = 0
        for word in query:
            if word in doc_title:
                if(word not in trivial_words):
                    count = count + 1
        # print("Old Score: " + str(scores[doc_id][1]))
        scores[doc_id][1] = scores[doc_id][1] * (1+count*title_weight)
        # print("New Score: " + str(scores[doc_id][1]))

    return scores


def scoring(query, database_lnc, vocabulary_keys, inverse_vocab_word_dict, term_document_frequency, N, doc_titles):
    corrected_query, query_vector = process_query_vector(
        query, vocabulary_keys, inverse_vocab_word_dict, term_document_frequency, N)
    scores = calculate_score(query_vector, database_lnc)
    original_scores = scores.copy()

    original_scores = sorted(original_scores, key=lambda x: x[1], reverse=True)
    print("Top 10 Scoring Documents without H1 are: ")
    for ind in range(10):
        if(original_scores[ind][1] == 0):
            break
        print(doc_titles[original_scores[ind][0]] + " is at rank " +
              str(ind+1) + " Score: " + str(original_scores[ind][1]))

    print("\n-----------------------------------------\n")

    scores = title_weighting(scores, corrected_query, doc_titles)

    scores = sorted(scores, key=lambda x: x[1], reverse=True)
    print("Top 10 Scoring Documents with H1 are: ")
    for ind in range(10):
        if(scores[ind][1] == 0):
            break
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


def main():
    # index_construction(   sys.argv[1])
    database_lnc = np.load("database_lnc.npy")
    N = database_lnc.shape[0]
    vocabulary_dict = pickle.load(open("vocabulary_dict.pkl", "rb"))

    vocabulary_keys = list(vocabulary_dict.keys())
    inverse_vocab_word_dict = {k: v for v, k in enumerate(vocabulary_keys)}
    term_document_frequency = np.count_nonzero(database_lnc, axis=0)
    doc_titles = pickle.load(open("doc_titles.pkl", "rb"))

    scoring(sys.argv[2:], database_lnc, vocabulary_keys,
            inverse_vocab_word_dict, term_document_frequency, N, doc_titles)


if __name__ == "__main__":
    if(len(sys.argv) < 3):
        print("Incorrect number of arguements")
        exit(-1)
    main()
