from preprocess.clean_up import clean_up
from preprocess.lemmanizer import lemmanize
from preprocess.read_pdf import glob_pdfs, read_pdf

from typing import Any, Iterable
from itertools import chain
from math import log
from pandas import DataFrame

def flatten(docs: list[Any]) -> list[str]:
    ''' Takes a list of list of strings as an argument and returns a flattened
    list of strings to be used to calculate the document_frequency '''
    return list(chain(*docs))

def term_freq(token: str, docs: list[list[str]]) -> list[float]:
    ''' Takes a token as an argument and returns a list with the term_freq for
    each doc in docs
    TF = number of ocurrence of token in doc / len(doc)
    '''
    return [round(doc.count(token)/len(doc),8) for doc in docs]

def doc_freq(token: str, docs: list[str]) -> int:
    ''' Returns the a dicitonary with tokens as keys and their document
    frequencies as value 
    DF = flattened_docs.count(token)
    '''
    return docs.count(token)

def inverse_df(doc_freq: int, len_docs: int) -> float:
    ''' Returns a dictionary with each word that is a key in doc_frequencies as
    a key and their idf number as a value 
    IDF = log(len_docs / (DF + 1))
    '''
    return log(len_docs/(doc_freq + 1))

def tf_idf(df: DataFrame, idf: str='idf', tf: str='tf') -> list[list[float]]:
    return [ [round(tf * df[idf][i], 8) for tf in tfs]
              for i, tfs in enumerate(df[tf]) ]

def build_dataframe(lemmas: list[list[str]]) -> DataFrame:
    dataframe = DataFrame({'tokens': sorted(set(flatten(lemmas)))})
    to_be_applied = (
                    ('tf', 'tokens', term_freq, {'docs':lemmas}),
                    ('df', 'tokens', doc_freq, {'docs':flatten(lemmas)}),
                    ('idf', 'df', inverse_df, {'len_docs':len(lemmas)}),
                 )
    for new_field, field, func, kargs in to_be_applied:
        dataframe[new_field] = dataframe[field].apply(func=func, **kargs)
    dataframe['tf_idfs'] = tf_idf(dataframe)
    return dataframe

def main() -> None:

    to_sub: dict[str, str] = {
            # O conjunto chave e valor neste dicionário serão usados com re.sub
            # para limpar os textos de números romanos, -se, tração, vs e
            # transformando o nome de alguns locais em sigla para não se
            # separarem na hora que a string for separada em tokens
            r'I\.|I+ |IV| V |\-se|\—| vs|\d+': ' ',
            'Rio de Janeiro': 'RJ', 
            'São Paulo': 'SP', 
            'Reino Unido': 'UK',
            }

    pdfs: list[str] = glob_pdfs(directory='corpus')

    scraped_text: list[str] = [read_pdf(pdf) for pdf in pdfs]

    clean_texts: list[str] = [clean_up(text, to_sub) for text in scraped_text]

    lemmas: list[list[str]] = [lemmanize(t) for t in clean_texts]

    data_frame = build_dataframe(lemmas)

    print(data_frame)

if __name__ == '__main__':

    main()
