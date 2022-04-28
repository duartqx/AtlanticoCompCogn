from show import *
from word2vec import *

from gensim.models import KeyedVectors
from nltk.tokenize import RegexpTokenizer
from pandas import read_csv
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

def get_data():
#    csv_data = 'https://raw.githubusercontent.com/dleebrown/' + \
#               'NLP_Techniques_Python/master/' + \
#               'socialmedia_relevant_cols_clean.csv'
    csv_data = 'socialmedia_relevant_cols_clean.csv'
    data = read_csv(csv_data)
    data.columns=['id', 'text', 'choose_one', 'class_label']
    #return stdize_text(q, 'text')
    return data

def standardize_text(df, text_field):
    to_replace = { "http\S+": "", "http": "", "@\S+": "", 
                   "[^A-Za-z0-9(),!?@\'\`\"\_\n]": " ", "@": "at" }
    df[text_field] = df[text_field].replace(to_replace)
    df[text_field] = df[text_field].str.lower()
    return df

def apply_tokenize(data, text_field):
    ''' Method to break the data[text_field] into tokens'''
    tokenizer = RegexpTokenizer(r'\w+')
    data['tokens'] = data[text_field].apply(tokenizer.tokenize)
    return data

def tt_split(list_corpus, list_labels):
    ''' Returns tuple of four items to be assigned to 
        x_train, x_test, y_train and y_test '''
    return train_test_split(list_corpus, list_labels, 
                            test_size=0.2, random_state=40)

def train_test(vectorizer, x_train, x_test):
    ''' Returns two item tuple to be assigned to train_counts and test_counts
    using CountVectorizer or TfidfVectorizer'''
    x_train_ct =  vectorizer.fit_transform(x_train)
    x_test_ct = vectorizer.transform(x_test)
    return x_train_ct, x_test_ct

def get_clf_n_predict(x_train_counts, x_test_counts, y_train):
    clf = LogisticRegression(C=30.0, class_weight='balanced', 
                             solver='newton-cg', multi_class='multinomial',
                             n_jobs=-1, random_state=40)
    clf = clf.fit(x_train_counts, y_train)
    return clf, clf.predict(x_test_counts)

def train_vect_n_plot(x_train, x_test, y_test, 
                      vectorizer=None,
                      lsa=False, metrics=False, 
                      conf=False, imp=True):
    if vectorizer:
        x_train_vec, x_test_vec = train_test(vectorizer, x_train, x_test)

    clf_vec, y_pred_vec = get_clf_n_predict(x_train_vec, x_test_vec, y_train)
    if lsa: plot_LSA(x_train_vec, y_train)
    if metrics: show_metrics(y_test, y_pred_vec)
    if conf: show_confusion(y_test, y_pred_vec)
    if imp: show_importance(vectorizer, clf_vec)

if __name__ == '__main__':

    #data = get_data()
    #data = standardize_text(data, 'text')
    data = apply_tokenize(get_data(), 'text')
    # Inspecionando novamente os dados
    #show_token_info(q, 'tokens', do_plot=True)
    t_corpus = data['text'].tolist()
    t_labels = data['class_label'].tolist()

    x_train, x_test, y_train, y_test = tt_split(t_corpus, t_labels)

    # BoW CountVectorizer
#    train_vect_n_plot(x_train, x_test, y_test, vectorizer=CountVectorizer(),
#            lsa=False, metrics=False, conf=False, imp=True)

    # BoW Tfidf
    train_vect_n_plot(x_train, x_test, y_test, vectorizer=TfidfVectorizer(),
            lsa=False, metrics=False, conf=False, imp=True)

    # Word2Vec
    # Training the relation matrix with news from google
    # The dataset can be downloaded from
    # https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit
    # It's a very large file (1.5gb) and it takes some minutes to decompress
    #word2vec_path = "GoogleNews-vectors-negative300.bin.gz"
    #word2vec = KeyedVectors.load_word2vec_format(word2vec_path, binary=True)

    #emb = get_word2vec_embeddings(word2vec, data)

    #x_train_w2v, x_test_w2v, y_train_w2v, y_test_w2v = tt_split(emb, t_labels)
    #clf_w2v, y_pred_w2v = get_clf_n_predict(x_train_w2v, x_test_w2v, y_train_w2v)
    #plot_LSA(emb, t_labels)
    #show_metrics(y_test_w2v, y_predicted_w2v)
    #show_confusion(y_test_w2v, y_predicted_w2v)
