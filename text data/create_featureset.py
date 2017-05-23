'''
lexicon
[char, table, spoon. television]

I pulled the chair up to the table

np.zeros(len(lexicon))

[0 0 0 0]

chair exist in the lexicon so 
[1 1 0 0]
'''

# Finds Most used words given a dataset
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import numpy as np
import random
import pickle
from collections import Counter

'''
Stemming removes ing and ed tokenize makes it an actual word
Lemmatizer checks the tense of the word
This is NLP if you want to learn more read 
NLP with Python by Steve Bird
'''

lemmatizer = WordNetLemmatizer()
# If you have a stronger PC you can increase this
hm_lines = 100000

def create_lexicon(pos, neg):
    lexicon = []
    for fi in [pos, neg]:
        with open(fi, 'r') as f:
            contents = f.readlines()
            for l in contents[:hm_lines]:
                all_words = word_tokenize(l)
                lexicon += list(all_words)
    lexicon = [lemmatizer.lemmatize(i) for i in lexicon]
    w_counts = Counter(lexicon)
    # w_counts = {'the': 52521, 'and': 25242}

    l2 = []
    for w in w_counts:
        if 1000 > w_counts[w] > 50:
            l2.append(w)
    print(len(l2))
    return l2

def sample_handling(sample, lexicon, classification):
    featureset = []
    '''
    [
    1 0 positive samples 
    0 1 is a negative sample
    [0 1 0 1 1 0]], [1 0]],
    [],
    ]
    '''

    with open(sample, 'r') as f:
        contents = f.readline()
        for l in contents[:hm_lines]:
            current_words = word_tokenize(l.lower())
            current_words = [lemmatizer.lemmatize(i) for i in current_words]
            features = np.zeros(len(lexicon))
            for word in current_words:
                if word.lower() in lexicon:
                    index_value = lexicon.index(word.lower())
                    features[index_value] += 1
            features = list(features)
            featureset.append([features, classification])
    return featureset

def create_feature_sets_and_labels(pos, neg, test_size=0.1):
    lexicon = create_lexicon(pos,neg)
    features = []
    features += sample_handling('pos.txt', lexicon, [1,0])
    features += sample_handling('neg.txt', lexicon, [0, 1])
    random.shuffle(features) # we need to do this for the neural network

    features = np.array(features)
    test_size = int(test_size*len(features))

    train_x = list(features[:, 0][:-test_size])
    train_y = list(features[:, 1][:-test_size])

    test_x = list(features[:, 0][:-test_size:])
    test_y = list(features[:, 1][:-test_size:])
    return train_x, train_y, test_x, test_y

if __name__ == '__main__':
    train_x, train_y, test_x, test_y = \
        create_feature_sets_and_labels('pos.txt', 'neg.txt')
    with open('sentiment_set.pickle', 'wb') as f:
        pickle.dump([train_x, train_y, test_x, test_y], f)