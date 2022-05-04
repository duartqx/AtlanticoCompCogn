from utils.show import *
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

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
    clf = LogisticRegression(C=30.0, 
                             class_weight='balanced', 
                             solver='newton-cg', 
                             multi_class='multinomial',
                             n_jobs=-1, 
                             random_state=40
                             ).fit(x_train_counts, y_train)
    return clf, clf.predict(x_test_counts)

def train_vect_n_plot(
                        x_train, 
                        x_test, 
                        y_test, 
                        vectorizer=None,
                        imp=True, 
                        lsa=False, 
                        metr=False, 
                        conf=False
                     ):
    if vectorizer:
        x_train_vec, x_test_vec = train_test(vectorizer, x_train, x_test)

    clf_vec, y_pred_vec = get_clf_n_predict(x_train_vec, x_test_vec, y_train)

    if imp: show_importance(vectorizer, clf_vec)
    elif lsa: plot_LSA(x_train_vec, y_train)
    elif metrics: show_metrics(y_test, y_pred_vec)
    elif conf: show_confusion(y_test, y_pred_vec)
