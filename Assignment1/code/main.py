import re
import sys
import nltk
import string
from nltk.tokenize import word_tokenize
from nltk.util import ngrams
from nltk.probability import FreqDist
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from collections import Counter
from itertools import islice
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
import math

stop_words = set(stopwords.words('english'))


def read_file(filename):
    f = open(filename, "r")
    data = f.read()
    f.close()
    return data


def remove_html_tags(data):
    data = data.lower()
    a_clean_regex = re.compile('<.*?>')
    return re.sub(a_clean_regex, '', data)


def tokenize_data(data):
    return word_tokenize(data)


def remove_stop_words(data):
    return [word for word in data if word not in stop_words and word not in string.punctuation and word not in ["''", '``', "'s", '’']]


def stem_data(data):
    ps = PorterStemmer()
    return [ps.stem(word) for word in data]


def lemmatize_data(data):
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(word) for word in data]


def get_ngrams(data, n):
    return list(ngrams(data, n))


def get_freqdist(ngrams):
    return FreqDist(ngrams)


def master_dict_generator(data):
    master_dict = {}
    for item in data:
        if item in master_dict:
            master_dict[item] += 1
        else:
            master_dict[item] = 1
    return master_dict


def coverage_calculator(data, ngrams, threshold=0.9):
    coverage = 0
    count = 0
    for item in data:
        coverage += item[1]/len(ngrams)
        count += 1
        if coverage > threshold:
            break
    return count


def plot_fd(fd):
    XY = fd.items()
    XY = sorted(XY, key=lambda pair: pair[1], reverse=True)
    X = [x for (x, y) in XY]
    Y = [math.log10(y) for (x, y) in XY]
    nX = [math.log10(i) for i in range(1, len(X)+1)]
    plt.figure()
    plt.plot(nX, Y, label='counts to tokens')
    plt.xticks(nX, X, rotation='vertical')
    plt.xlabel('Tokens')
    plt.ylabel('Counts')
    plt.title('Counts by tokens')
    plt.legend(loc='upper right', fontsize='small')
    plt.grid()
    plt.tight_layout()
    plt.show()


def unigrams_processing(data):
    print("Generating Unigrams")
    unigrams = get_ngrams(data, 1)
    print("Number of Unigrams in the text: " +
          str(len(set(unigrams))) + "\n____________________\n")
    unigrams = list(unigrams)
    master_dict = master_dict_generator(unigrams)
    master = Counter(master_dict).most_common()
    count = coverage_calculator(master, unigrams, 0.9)
    print("Number of Unigrams required to cover 90% of the corpus: " +
          str(count) + "\n____________________\n")
    print("Plotting the Unigram Frequency Distribution \n____________________\n")
    fdist_uni = get_freqdist(unigrams)
    plot_fd(fdist_uni)


def bigrams_processing(data):
    print("Generating Bigrams")
    bigrams = get_ngrams(data, 2)
    print("Number of Bigrams in the text: " +
          str(len(set(bigrams))) + "\n____________________\n")
    bigrams = list(bigrams)
    master_dict = master_dict_generator(bigrams)
    master = Counter(master_dict).most_common()
    count = coverage_calculator(master, bigrams, 0.8)
    print("Number of Bigrams required to cover 80% of the corpus: " +
          str(count) + "\n____________________\n")
    print("Plotting the Bigram Frequency Distribution \n____________________\n")
    fdist_bi = get_freqdist(bigrams)
    plot_fd(fdist_bi)


def trigrams_processing(data):
    print("Generating Trigrams")
    trigrams = get_ngrams(data, 3)
    print("Number of Trigrams in the text: " +
          str(len(set(trigrams))) + "\n____________________\n")
    trigrams = list(trigrams)
    master_dict = master_dict_generator(trigrams)
    master = Counter(master_dict).most_common()
    count = coverage_calculator(master, trigrams, 0.7)
    print("Number of Trigrams required to cover 70% of the corpus: " +
          str(count) + "\n____________________\n")
    print("Plotting the Trigram Frequency Distribution \n____________________\n")
    fdist_tri = get_freqdist(trigrams)
    plot_fd(fdist_tri)


def collocations_calculator(data):
    print("Top 20 bi-gram collocations in the text corpus using Chi-square test: \n")
    collocation = {}
    unigram_dict = {}
    unigrams = get_ngrams(data, 1)
    for unigram in unigrams:
        if unigram[0] in unigram_dict:
            unigram_dict[unigram[0]] += 1
        else:
            unigram_dict[unigram[0]] = 1
    bigram_dict = master_dict_generator(get_ngrams(data, 2))
    for bigram in bigram_dict:
        table = [0, 0, 0, 0]
        table[0] = bigram_dict[bigram]
        table[1] = unigram_dict[bigram[1]] - bigram_dict[bigram]
        table[2] = unigram_dict[bigram[0]] - bigram_dict[bigram]
        table[3] = len(bigram_dict) - table[1] - table[2]
        chisquared = (len(bigram_dict)*(((table[0]*table[3]) - (table[1]*table[2]))**2))/(
            (table[0]+table[1])*(table[0] + table[2])*(table[1] + table[3])*(table[2]+table[3]))
        collocation[bigram] = chisquared
    collocation = Counter(collocation).most_common()
    top_20_collocations = list(islice(collocation, 20))
    return top_20_collocations


def main():
    print("\nReading File\n____________________\n")
    data = read_file(sys.argv[1])

    print("Removing HTML Tags\n____________________\n")
    cleaned_data = remove_html_tags(data)

    print("Tokenizing Data\n____________________\n")
    tokenized_data = tokenize_data(cleaned_data)

    print("Removing Stop Words\n____________________\n")
    tokenized_data = remove_stop_words(tokenized_data)

    unigrams_processing(tokenized_data)
    bigrams_processing(tokenized_data)
    trigrams_processing(tokenized_data)

    print("\n\n____________________________Stemming Data Now____________________________\n\n")

    stemmized_data = stem_data(tokenized_data)
    unigrams_processing(stemmized_data)
    bigrams_processing(stemmized_data)
    trigrams_processing(stemmized_data)

    print("\n\n____________________________Lemmatizing Data Now____________________________\n\n")

    lemmatized_data = lemmatize_data(tokenized_data)
    unigrams_processing(lemmatized_data)
    bigrams_processing(lemmatized_data)
    trigrams_processing(lemmatized_data)

    top_20_collocations = collocations_calculator(tokenized_data)
    print(top_20_collocations)


if __name__ == "__main__":
    if(len(sys.argv) != 2):
        print("Incorrect number of arguements")
        exit(-1)
    main()
