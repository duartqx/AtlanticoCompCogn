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

def doc_freq(token: str, docs: list[str]) -> dict[str, int]:
    ''' Returns the a dicitonary with tokens as keys and their document
    frequencies as value 
    DF = flattened_docs.count(token)
    '''
    return docs.count(token)

def inverse_df(len_docs: int, doc_freq: dict[str, int]) -> dict[str, int]:
    ''' Returns a dictionary with each word that is a key in doc_frequencies as
    a key and their idf number as a value 
    IDF = log(len_docs / (DF + 1))
    '''
    return {tkn: log(len_docs/(df + 1)) for tkn, df in doc_freq.items()}

def build_dataframe(lemmas: list[list[str]]) -> DataFrame:
    dataframe = DataFrame({'tokens': sorted(set(flatten(lemmas)))})
    func_field_data = (
                    (term_freq, 'tf', lemmas),
                    (doc_freq, 'df', flatten(lemmas)),
                 )
    for func, field, data in func_field_data:
        dataframe[field] = dataframe['tokens'].apply(func=func, docs=data)
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
