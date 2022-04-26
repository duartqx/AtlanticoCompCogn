from itertools import product
from matplotlib.colors import ListedColormap
from numpy import newaxis, arange
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.metrics import accuracy_score, f1_score, precision_score 
from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score, classification_report
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

# Show tokens info
def show_token_info(data, token_field, do_plot=False):
    all_words = [word for tokens in data[token_field] for word in tokens]
    sentence_lengths = [len(tokens) for tokens in data[token_field]]
    VOCAB = sorted(set(all_words))
    if do_plot:
        fig = plt.figure(figsize=(10, 10))
        plt.xlabel('Tamanho da sentença')
        plt.ylabel('Número de sentenças')
        plt.hist(sentence_lengths)
        plt.show()
    else:
        print(f'{len(all_words)} Quantidade total de palavras, com um',
              f'vocabulário de {len(VOCAB)}',
              f'\nTamanho máximo de uma sentença {max(sentence_lengths)}')

# Fucntion to plot LSA
def plot_LSA(test_data, test_labels, do_plot= True):
    lsa = TruncatedSVD(n_components=2)
    lsa.fit(test_data)
    lsa_scores = lsa.transform(test_data)
    color_mapper = {label:idx for idx, label in enumerate(set(test_labels))}
    color_column = [color_mapper[label] for label in test_labels]
    colors = ['orange', 'blue', 'blue']
    if do_plot:
        fig = plt.figure(figsize=(10,10))
        plt.scatter(lsa_scores[:,0], lsa_scores[:,1], s=8, alpha=0.8, 
                    c=test_labels, cmap=ListedColormap(colors))
        red_patch = mpatches.Patch(color='orange', label='Irrelevante')
        green_patch = mpatches.Patch(color='blue', label='Disaster')
        plt.legend(handles=[red_patch, green_patch], prop={'size':20})
        plt.show()

# Functions to plot Confusion Matrix
def plot_confusion_matrix(cm, classes=[], normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.winter):
    if not classes:
        classes = ['Irrelevant','Disaster','Unsure']
    fig = plt.figure(figsize=(9,9))
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, newaxis]
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=30)
    plt.colorbar()
    tick_marks = arange(len(classes))
    plt.xticks(tick_marks, classes, fontsize=20)
    plt.yticks(tick_marks, classes, fontsize=20)
    
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.

    for i, j in product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", 
                 color="white" if cm[i, j] < thresh else "black", 
                 fontsize=40)
        
    plt.tight_layout()
    plt.ylabel('True label', fontsize=30)
    plt.xlabel('Predicted label', fontsize=30)

    return plt

def show_confusion(y_test, y_predicted):
    cm = confusion_matrix(y_test,y_predicted)
    fig = plt.figure(figsize=(9,9))
    plot = plot_confusion_matrix(cm)
    print('Count Vectorizer confusion matrix')
    print(cm)
    plt.show()

# Functions to plot Importance
def define_subplot(num, pos, scores, words, title, subtitle, lbl='Importance'):
    plt.subplot(num)    
    plt.barh(pos, scores, align='center', alpha=0.5)    
    plt.title(title, fontsize=20)    
    plt.yticks(pos, words, fontsize=14)    
    plt.suptitle(subtitle, fontsize=16)    
    plt.xlabel(lbl, fontsize=20)
    return plt

def words_scores(pair):
    first, second = [], []
    for t in pair:
        first.append(t[0])
        second.append(t[1])
    return first, second

def get_pair(words, scores, r=False):
    pairs = [(a,b) for a,b in zip(words, scores)]
    pairs = sorted(pairs, key=lambda x: x[1], reverse=r) 
    return pairs

def get_most_important_features(vectorizer, model, n=5):
    index_to_word = {v:k for k,v in vectorizer.vocabulary_.items()}

    # Loop for each class
    classes = {}
    for class_index in range(model.coef_.shape[0]):
        word_importances = [(el, index_to_word[i]) 
                            for i,el in enumerate(model.coef_[class_index])]
        sorted_coeff = sorted(word_importances, key=lambda x:x[0],reverse=True)
        tops = sorted(sorted_coeff[:n], key = lambda x: x[0])
        bottom = sorted_coeff[-n:]
        classes[class_index] = {
            'tops':tops,
            'bottom':bottom
        }
    return classes

def plt_imp_words(top_scores, top_words, bottom_scores, bottom_words, name):
    y_pos = arange(len(top_words))

    top_pairs = get_pair(top_words, top_scores)
    bottom_pairs = get_pair(bottom_words, bottom_scores, r=True)

    top_words, top_scores = words_scores(top_pairs)
    bottom_words, bottom_scores = words_scores(bottom_pairs)

    fig = plt.figure(figsize=(10, 10))  

    define_subplot(121, y_pos, bottom_scores, bottom_words, 
                   'Irrelevant', 'Key words')
    define_subplot(122, y_pos, top_scores, top_words, 'Disaster', name)

    plt.subplots_adjust(wspace=0.8)
    plt.show()

def show_importance(cnt, clf):
    importance = get_most_important_features(cnt, clf, 10)

    top_scores, top_words = words_scores(importance[1]['tops'])
    bottom_scores, bottom_words = words_scores(importance[1]['bottom'])

    name = 'Most important words for relevance'
    plt_imp_words(top_scores, top_words, bottom_scores, bottom_words, name)

# Function to show Metrics
def show_metrics(y_test, y_predicted):
    # True positives + True negatives / total
    accuracy = accuracy_score(y_test, y_predicted)
    # True positives / (True positives + False positives)
    precision = precision_score(y_test, y_predicted, pos_label=None,
                                average='weighted')
    # True positives / (True positives + False negatives)
    recall = recall_score(y_test, y_predicted, pos_label=None, 
                          average='weighted')
    # Harmonic mean of precision and recall
    f1 = f1_score(y_test, y_predicted, pos_label=None, average='weighted')

    print(f'accuracy = {accuracy:.3f}, precision = {precision:.3f},',
          f'recall = {recall:.3f}, f1 = {f1:.3f}')
