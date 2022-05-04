from utils.preprocess_data import *
from utils.show import *
from utils.train_test import *
from utils.word2vec import *

from gensim.models import KeyedVectors
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

def main(what2plot: dict, tfidf=True, count=False) -> None:

    data = apply_tokenize(get_data(), 'text')
    # Inspecionando novamente os dados
    t_corpus = data['text'].tolist()
    t_labels = data['class_label'].tolist()

    x_train, x_test, y_train, y_test = tt_split(t_corpus, t_labels)
    
    if tfidf:
        # BoW TfidfVectorizer
        train_vect_n_plot(x_train, x_test, y_test, vect=CountVectorizer(), 
                          **what2plot)
    elif count:
        # BoW CountVectorizer
        train_vect_n_plot(x_train, x_test, y_test, vect=TfidfVectorizer(),
                          **what2plot)
    else:
        # Word2Vec
        # Training the relation matrix with news from google
        # The dataset can be downloaded from
        # https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit
        # It's a very large file (1.5gb) and it takes some minutes to decompress
        word2vec_path = "GoogleNews-vectors-negative300.bin.gz"
        word2vec = KeyedVectors.load_word2vec_format(word2vec_path, binary=True)
        
        emb = get_word2vec_embeddings(word2vec, data)
        
        x_train_w2v, x_test_w2v, y_train_w2v, y_test_w2v = tt_split(emb, t_labels)
        clf_w2v, y_pred_w2v = get_clf_n_predict(x_train_w2v, x_test_w2v, y_train_w2v)
        plot_LSA(emb, t_labels)
        show_metrics(y_test_w2v, y_predicted_w2v)
        show_confusion(y_test_w2v, y_predicted_w2v)


if __name__ == '__main__':

    main_kargs = dict()
    what2plot = dict(imp=True, lsa=False, metr=False, conf=False)

    main(what2plot, main_kargs)
