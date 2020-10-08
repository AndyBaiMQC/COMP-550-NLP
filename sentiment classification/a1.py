import warnings
warnings.filterwarnings('ignore')

import numpy as np
import codecs
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import CountVectorizer

import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
from nltk.corpus import stopwords, wordnet
from nltk import word_tokenize, pos_tag
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
stoplist = stopwords.words('english')

def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return None

def read_data():
    text, y = [], []
    # assume data files are in the same directory
    for line in codecs.open('data/rt-polarity.pos', encoding='Windows-1252'):
        line = line.strip()
        text.append(line.lower())
        y.append(1)
    for line in codecs.open('data/rt-polarity.neg', encoding='Windows-1252'):
        line = line.strip()
        text.append(line.lower())
        y.append(-1)
    y = np.array(y)
    return text, y

def feature_transformation(text, lemmatize_or_stem=True, use_stop_word=True, min_freq=5):
    if lemmatize_or_stem:  # to lemmatize
        wnl = WordNetLemmatizer()
        for i in range(len(text)):
            pos = pos_tag(word_tokenize(text[i]))
            sentence = []
            for w, tag in pos:
                tag = get_wordnet_pos(tag)
                if tag is not None:
                    sentence.append(wnl.lemmatize(w, tag))
                else:
                    sentence.append(w)
            text[i] = ' '.join(sentence)
    else:  # to stem
        ps = PorterStemmer()
        for i in range(len(text)):
            text[i] = ' '.join([ps.stem(w) for w in text[i].split()])
    if use_stop_word:  # filter out stop words
        for i in range(len(text)):
            text[i] = ' '.join([w for w in text[i].split() \
                if w not in stoplist])

    cnt_vectorizer = CountVectorizer(min_df=min_freq)  # remove infrequently occurring words
    x = cnt_vectorizer.fit_transform(text)
    return x

if __name__ == '__main__':
    text, y = read_data()
    min_freqs = [2, 3, 4, 5, 8, 10]

    # parameters
    alphas = [0.0, 0.5, 1.0, 2.0] # only naive bayes
    regularizations = ['l1', 'l2'] # svm, logistic regression
    Cs = [0.2, 0.5, 1.0, 2.0]
    min_child_weights = [1] # xgb
    gammas = [0.5]
    subsamples = [0.6, 0.8]
    lrs = [0.05, 0.1, 0.2]
    max_depths = [3, 4, 5]
    ns_estimators = [200, 400, 600, 800, 1000]


    # this section uses combination of hyperparameters with best results
    # to re-run grid search, comment out lines 95 to 204
    # and uncomment lines 207 to 332

    # logistic regression
    print('machine learning algorithm: logistic regression')
    for lemmatize_or_stem in [False, True]:
        for use_stop_word in [False, True]:
            for min_freq in min_freqs:
                x = feature_transformation(text, lemmatize_or_stem=lemmatize_or_stem, use_stop_word=use_stop_word, min_freq=min_freq)
                acc = 0.
                for train_index, test_index in StratifiedKFold(n_splits=5).split(x, y):
                    x_train, x_test = x[train_index,:], x[test_index,:]
                    y_train, y_test = y[train_index], y[test_index]

                    model = LogisticRegression()
                    model.fit(x_train, y_train)
                    z = model.predict(x_test)
                    acc += np.mean(z == y_test)
                acc /= 5.
                if lemmatize_or_stem:
                    print('  with lemmatizing', end=', ')
                else:
                    print('  with stemming', end=', ')
                if use_stop_word:
                    print('with using stop words', end=', ')
                else:
                    print('without using stop words', end=', ')
                print('with minimal words frequency = %d' % min_freq, end=', ')
                print('the average acc of 5-fold CV is %.4f' % acc)
    print()

    # naive bayes
    print('machine learning algorithm: naive bayes')
    for lemmatize_or_stem in [False, True]:
        for use_stop_word in [False, True]:
            for min_freq in min_freqs:
                x = feature_transformation(text, lemmatize_or_stem=lemmatize_or_stem, use_stop_word=use_stop_word, min_freq=min_freq)
                acc = 0.
                for train_index, test_index in StratifiedKFold(n_splits=5).split(x, y):
                    x_train, x_test = x[train_index,:], x[test_index,:]
                    y_train, y_test = y[train_index], y[test_index]

                    model = MultinomialNB(alpha=2.0)
                    model.fit(x_train, y_train)
                    z = model.predict(x_test)
                    acc += np.mean(z == y_test)
                acc /= 5.
                if lemmatize_or_stem:
                    print('  with lemmatizing', end=', ')
                else:
                    print('  with stemming', end=', ')
                if use_stop_word:
                    print('with using stop words', end=', ')
                else:
                    print('without using stop words', end=', ')
                print('with minimal words frequency = %d' % min_freq, end=', ')
                print('the average acc of 5-fold CV is %.4f' % acc)
    print()
    
    # svm with linear kernel
    print('machine learning algorithm: svm(linear kernel)')
    for lemmatize_or_stem in [False, True]:
        for use_stop_word in [False, True]:
            for min_freq in min_freqs:
                x = feature_transformation(text, lemmatize_or_stem=lemmatize_or_stem, use_stop_word=use_stop_word, min_freq=min_freq)
                acc = 0.
                for train_index, test_index in StratifiedKFold(n_splits=5).split(x, y):
                    x_train, x_test = x[train_index,:], x[test_index,:]
                    y_train, y_test = y[train_index], y[test_index]

                    model = LinearSVC(C=0.2)
                    model.fit(x_train, y_train)
                    z = model.predict(x_test)
                    acc += np.mean(z == y_test)
                acc /= 5.
                if lemmatize_or_stem:
                    print('  with lemmatizing', end=', ')
                else:
                    print('  with stemming', end=', ')
                if use_stop_word:
                    print('with using stop words', end=', ')
                else:
                    print('without using stop words', end=', ')
                print('with minimal words frequency = %d' % min_freq, end=', ')
                print('the average acc of 5-fold CV is %.4f' % acc)
    print()
    
    # extreme gradient boosting
    print('machine learning algorithm: xgb')
    for lemmatize_or_stem in [False, True]:
        for use_stop_word in [False, True]:
            for min_freq in min_freqs:
                x = feature_transformation(text, lemmatize_or_stem=lemmatize_or_stem, use_stop_word=use_stop_word, min_freq=min_freq)
                acc = 0.
                for train_index, test_index in StratifiedKFold(n_splits=5).split(x, y):
                    x_train, x_test = x[train_index,:], x[test_index,:]
                    y_train, y_test = y[train_index], y[test_index]

                    model = XGBClassifier(min_child_weights=1, gamma=0.5, subsamples=0.6, max_depth=5, learning_rate=0.2, n_estimators=1000)
                    model.fit(x_train, y_train)
                    z = model.predict(x_test)
                    acc += np.mean(z == y_test)
                acc /= 5.
                if lemmatize_or_stem:
                    print('  with lemmatizing', end=', ')
                else:
                    print('  with stemming', end=', ')
                if use_stop_word:
                    print('with using stop words', end=', ')
                else:
                    print('without using stop words', end=', ')
                print('with minimal words frequency = %d' % min_freq, end=', ')
                print('the average acc of 5-fold CV is %.4f' % acc)

    # this section is to re-run searches

    # # logistic regression
    # print('machine learning algorithm: logistic regression')
    # for r in regularizations:
    #     for c in Cs:
    #         print("regulatizations and strength: ", r,' ',c)
    #         for lemmatize_or_stem in [False, True]:
    #             for use_stop_word in [False, True]:
    #                 for min_freq in [2, 4, 5, 6, 8, 10]:
    #                     x = feature_transformation(text, lemmatize_or_stem=lemmatize_or_stem, use_stop_word=use_stop_word, min_freq=min_freq)
    #                     acc = 0.
    #                     for train_index, test_index in StratifiedKFold(n_splits=10).split(x, y):
    #                         x_train, x_test = x[train_index,:], x[test_index,:]
    #                         y_train, y_test = y[train_index], y[test_index]

    #                         model = LogisticRegression(penalty=r, C=c)
    #                         model.fit(x_train, y_train)
    #                         z = model.predict(x_test)
    #                         acc += np.mean(z == y_test)
    #                     acc /= 10.
    #                     if lemmatize_or_stem:
    #                         print('  with lemmatizing', end=', ')
    #                     else:
    #                         print('  with stemming', end=', ')
    #                     if use_stop_word:
    #                         print('with using stop words', end=', ')
    #                     else:
    #                         print('without using stop words', end=', ')
    #                     print('with minimal words frequency = %d' % min_freq, end=', ')
    #                     print('the average acc of 10-fold CV is %.4f' % acc)

    # # naive bayes
    # print('machine learning algorithm: naive bayes')
    # for a in alphas:
    #     print('alpha: ',a)
    #     for lemmatize_or_stem in [False, True]:
    #         for use_stop_word in [False, True]:
    #             for min_freq in [2, 4, 5, 6, 8]:
    #                 x = feature_transformation(text, lemmatize_or_stem=lemmatize_or_stem, use_stop_word=use_stop_word, min_freq=min_freq)
    #                 acc = 0.
    #                 for train_index, test_index in StratifiedKFold(n_splits=5).split(x, y):
    #                     x_train, x_test = x[train_index,:], x[test_index,:]
    #                     y_train, y_test = y[train_index], y[test_index]

    #                     model = MultinomialNB(alpha=a)
    #                     model.fit(x_train, y_train)
    #                     z = model.predict(x_test)
    #                     acc += np.mean(z == y_test)
    #                 acc /= 5.
    #                 if lemmatize_or_stem:
    #                     print('  with lemmatizing', end=', ')
    #                 else:
    #                     print('  with stemming', end=', ')
    #                 if use_stop_word:
    #                     print('with using stop words', end=', ')
    #                 else:
    #                     print('without using stop words', end=', ')
    #                 print('with minimal words frequency = %d' % min_freq, end=', ')
    #                 print('the average acc of 5-fold CV is %.4f' % acc)

    # # svm with linear kernel
    # print('machine learning algorithm: svm(linear kernel)')
    # for c in Cs:
    #     print('c: ',c)
    #     for lemmatize_or_stem in [False, True]:
    #         for use_stop_word in [False, True]:
    #             for min_freq in min_freqs:
    #                 x = feature_transformation(text, lemmatize_or_stem=lemmatize_or_stem, use_stop_word=use_stop_word, min_freq=min_freq)
    #                 acc = 0.
    #                 for train_index, test_index in StratifiedKFold(n_splits=5).split(x, y):
    #                     x_train, x_test = x[train_index,:], x[test_index,:]
    #                     y_train, y_test = y[train_index], y[test_index]

    #                     model = LinearSVC(C=c)
    #                     model.fit(x_train, y_train)
    #                     z = model.predict(x_test)
    #                     acc += np.mean(z == y_test)
    #                 acc /= 5.
    #                 if lemmatize_or_stem:
    #                     print('  with lemmatizing', end=', ')
    #                 else:
    #                     print('  with stemming', end=', ')
    #                 if use_stop_word:
    #                     print('with using stop words', end=', ')
    #                 else:
    #                     print('without using stop words', end=', ')
    #                 print('with minimal words frequency = %d' % min_freq, end=', ')
    #                 print('the average acc of 5-fold CV is %.4f' % acc)

    # # extreme gradient boosting
    # print('machine learning algorithm: xgb')
    # for mcw in min_child_weights:
    #     print("min_child_weights: ", mcw)
    #     for g in gammas:
    #         print("gamma: ", g)
    #         for s in subsamples:
    #             print("subsample: ", s)
    #             for lr in lrs:
    #                 print("learning rate: ", lr)
    #                 for md in max_depths:
    #                     print("max depth: ", md)
    #                     for n in ns_estimators:
    #                         print("# estimator: ", n)
    #                         for lemmatize_or_stem in [False, True]:
    #                             for use_stop_word in [False, True]:
    #                                 for min_freq in min_freqs:
    #                                     x = feature_transformation(text, lemmatize_or_stem=lemmatize_or_stem, use_stop_word=use_stop_word, min_freq=min_freq)
    #                                     acc = 0.
    #                                     for train_index, test_index in StratifiedKFold(n_splits=5).split(x, y):
    #                                         x_train, x_test = x[train_index,:], x[test_index,:]
    #                                         y_train, y_test = y[train_index], y[test_index]

    #                                         model = XGBClassifier(min_child_weights=mcw, gamma=g, subsamples=s, max_depth=md, learning_rate=lr, n_estimators=n)
    #                                         model.fit(x_train, y_train)
    #                                         z = model.predict(x_test)
    #                                         acc += np.mean(z == y_test)
    #                                     acc /= 5.
    #                                     if lemmatize_or_stem:
    #                                         print('  with lemmatizing', end=', ')
    #                                     else:
    #                                         print('  with stemming', end=', ')
    #                                     if use_stop_word:
    #                                         print('with using stop words', end=', ')
    #                                     else:
    #                                         print('without using stop words', end=', ')
    #                                     print('with minimal words frequency = %d' % min_freq, end=', ')
    #                                     print('the average acc of 5-fold CV is %.4f' % acc)
