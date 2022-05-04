from nltk.tokenize import RegexpTokenizer
from pandas import read_csv

def standardize_text(df, text_field):
    to_replace = { "http\S+": "", "http": "", "@\S+": "", 
                   "[^A-Za-z0-9(),!?@\'\`\"\_\n]": " ", "@": "at" }
    df[text_field] = df[text_field].replace(to_replace)
    df[text_field] = df[text_field].str.lower()
    return df

def get_data():
    ''' Source: 
        'https://raw.githubusercontent.com/dleebrown/NLP_Techniques_Python/'
        'master/socialmedia_relevant_cols_clean.csv'
    '''
    csv_data = 'socialmedia_relevant_cols_clean.csv'
    data = read_csv(csv_data)
    data.columns=['id', 'text', 'choose_one', 'class_label']
    #return stdize_text(q, 'text')
    return data

def apply_tokenize(data, text_field):
    ''' Method to break the data[text_field] into tokens'''
    tokenizer = RegexpTokenizer(r'\w+')
    data['tokens'] = data[text_field].apply(tokenizer.tokenize)
    return data
