'''
@author: jcheung

Developed for Python 2. Automatically converted to Python 3; may result in bugs.
'''
import xml.etree.cElementTree as ET
import codecs
import re
import string
import math

import similarity_heuristic

from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords as sw
from nltk.wsd import lesk


stopWords = set(sw.words('english'))

tagDict = {
    'N': wn.NOUN,
    'V': wn.VERB,
    'R': wn.ADV,
    'J': wn.ADJ
}

OTHER_TAG = "OTHER"

true_word_senses_cache = {}


class WSDInstance:
    def __init__(self, my_id, lemma, context, context_pos, context_word_senses, index,  pos):
        self.id = my_id                  # id of the WSD instance
        self.lemma = lemma               # lemma of the word whose sense is to be resolved
        self.context = context           # lemma of all the words in the sentential context
        self.context_pos = context_pos   # part of speech for each context
        # list of word senses for each context.
        self.context_word_senses = context_word_senses
        self.index = index               # index of lemma within the context
        self.pos = pos                   # part of speech

    def __str__(self):
        '''
        For printing purposes.
        '''
        return f"{self.id} {self.lemma} {self.context} {self.context_pos} {self.index} {self.pos}"


def load_instances(f):
    '''
    Load two lists of cases to perform WSD on. The structure that is returned is a dict, where
    the keys are the ids, and the values are instances of WSDInstance.
    '''
    tree = ET.parse(f)
    root = tree.getroot()

    dev_instances = {}
    test_instances = {}

    for text in root:
        if text.attrib['id'].startswith('d001'):
            instances = dev_instances
        else:
            instances = test_instances
        for sentence in text:
            # construct sentence context
            context, context_pos, context_word_senses = preprocess_sentence(
                sentence)

            for i, el in enumerate(sentence):
                if el.tag == 'instance':
                    my_id = el.attrib['id']
                    lemma = to_ascii(el.attrib['lemma'])
                    pos = get_pos_tag(el.attrib['pos'])
                    instances[my_id] = WSDInstance(
                        my_id, lemma, context, context_pos, context_word_senses, i, pos)
    return dev_instances, test_instances


def load_key(f):
    '''
    Load the solutions as dicts.
    Key is the id
    Value is the list of correct sense keys. 
    '''
    dev_key = {}
    test_key = {}
    for line in open(f):
        if len(line) <= 1:
            continue
        doc, my_id, sense_key = line.strip().split(' ', 2)

        senses = sense_key.split()

        # Remove stopwords from key
        lemma = wn.lemma_from_key(senses[0]).name()
        if (lemma in stopWords):
            continue

        if doc == 'd001':
            dev_key[my_id] = senses
        else:
            test_key[my_id] = senses

    return dev_key, test_key


def to_ascii(s):
    # Remove all non-ascii characters; Decode to get string
    return codecs.encode(s, 'ascii', 'ignore').decode('ascii')


def to_synset(s):
    return wn.lemma_from_key(s).synset()


def get_pos_tag(tag):
    return tagDict[tag[0]] if (tag[0] in tagDict) else OTHER_TAG


def get_true_word_senses(key):

    global true_word_senses_cache

    cache_key = len(key)

    if (cache_key in true_word_senses_cache):
        return true_word_senses_cache[cache_key]

    # Retrieves all the true word sense from the key
    # It will be a list of list.
    true_word_senses = []
    for (k, true_values) in key.items():
        true_word_sense = []
        for true_value in true_values:
            true_word_sense.append(to_synset(true_value))
        true_word_senses.append(true_word_sense)

    true_word_senses_cache[cache_key] = true_word_senses
    return true_word_senses


def get_most_frequent_word_senses(key, instances):
    # Retrieves all the most frequent word sense from the instance
    # It will be a list.
    most_frequent_word_senses = []
    for (k, v) in key.items():
        instance_lemma = instances[k].lemma

        # Wordnet orders the synset by frequency
        most_frequent_word_senses.append(wn.synsets(instance_lemma)[0])

    return most_frequent_word_senses


def get_lesk_word_senses(key, instances):
    # Retrieves all the word sense from the instance from apply Lesk Algorithm
    # It will be a list.
    lesk_word_senses = []
    for (k, v) in key.items():
        instance_context_sentence = instances[k].context
        instance_lemma = instances[k].lemma
        instance_pos = instances[k].pos

        # Wordnet orders the synset by frequency
        word_sense = lesk(instance_context_sentence,
                          instance_lemma, instance_pos)

        lesk_word_senses.append(word_sense)

    return lesk_word_senses


def get_similarity_word_senses(key, instances, similarity_name):

    lch_word_senses = []

    for (k, v) in key.items():

        instance_lemma = instances[k].lemma
        instance_context = instances[k].context
        instance_context_pos = instances[k].context_pos
        instance_context_word_senses = instances[k].context_word_senses

        word_sense = similarity_heuristic.get_best_word_sense(
            instance_context, instance_context_pos, instance_context_word_senses,
            instance_lemma, similarity_name)

        lch_word_senses.append(word_sense)

    return lch_word_senses


def get_accuracy(true_values, pred_values):

    correct_values = 0
    total = len(true_values)

    if (total != len(pred_values)):
        print(
            f"Something is wrong... Found {total} in true and {len(pred_values)} in predicted.")

    for (true_value, pred_value) in zip(true_values, pred_values):
        if (pred_value is None):

            continue

        if (pred_value in true_value):
            correct_values += 1

    return (correct_values/total)


def get_most_frequent_word_sense_baseline(key, instances):
    # 1. Get True Word Sense
    true_word_senses = get_true_word_senses(key)
    # 2. Get Most Frequent Word Sense
    pred_word_senses = get_most_frequent_word_senses(key, instances)
    # 3. Calculate Accuracy
    most_frequent_sense_accuracy = get_accuracy(
        true_word_senses, pred_word_senses)

    return most_frequent_sense_accuracy


def get_lesk_word_sense_baseline(key, instances):
    # 1. Get True Word Sense
    true_word_senses = get_true_word_senses(key)
    # 2. Get Lesk Word Sense
    pred_word_senses = get_lesk_word_senses(key, instances)
    # 3. Calculate Accuracy
    lesk_word_sense_accuracy = get_accuracy(
        true_word_senses, pred_word_senses)
    return lesk_word_sense_accuracy


def get_similarity_sense_baseline(key, instances, similarity_name):
    # 1. Get True Word Sense
    true_word_senses = get_true_word_senses(key)
    # 2. Get lch word sense
    pred_word_senses = get_similarity_word_senses(
        key, instances, similarity_name)
    # 3. Calculate accuracy
    lch_word_sense_accuracy = get_accuracy(
        true_word_senses, pred_word_senses)

    return lch_word_sense_accuracy


def preprocess_sentence(sentence):

    context = []
    context_pos = []
    context_word_senses = []

    for el in sentence:
        lemma = el.attrib['lemma']

        if (lemma == "" or lemma in stopWords):
            continue

        pos = get_pos_tag(el.attrib['pos'])
        if (pos == OTHER_TAG):
            continue

        context.append(lemma)
        context_pos.append(pos)
        context_word_senses.append(wn.synsets(lemma, pos))

    return (context, context_pos, context_word_senses)


if __name__ == '__main__':
    data_f = 'multilingual-all-words.en.xml'
    key_f = 'wordnet.en.key'
    dev_instances, test_instances = load_instances(data_f)
    dev_key, test_key = load_key(key_f)

    # IMPORTANT: keys contain fewer entries than the instances; need to remove them
    dev_instances = {k: v for (k, v) in dev_instances.items() if k in dev_key}
    test_instances = {
        k: v for (k, v) in test_instances.items() if k in test_key}

    # Compare WSD with Most Frequent Sense Baseline (first word sense) and Key

    most_frequent_sense_accuracy = get_most_frequent_word_sense_baseline(
        test_key, test_instances)

    print(
        f"Most Frequent Sense Baseline Accuracy: {most_frequent_sense_accuracy}")

    # Use NLTK's implementation of Lesk's algorithm
    lesk_word_sense_accuracy = get_lesk_word_sense_baseline(
        test_key, test_instances)

    print(
        f"Lesk Word Sense Baseline Accuracy: {lesk_word_sense_accuracy}")

    best_heuristic = ""
    best_accuracy = 0

    for heuristic in similarity_heuristic.similarity_heuristic_dict:
        accuracy = get_similarity_sense_baseline(
            dev_key, dev_instances, heuristic)
        print(f"{heuristic} Accuracy: {accuracy}")
        if (accuracy >= best_accuracy):
            best_heuristic = heuristic

    best_heuristic_accuracy = get_similarity_sense_baseline(
        test_key, test_instances, best_heuristic)
    print(f"{best_heuristic} Accuracy: {best_heuristic_accuracy}")
