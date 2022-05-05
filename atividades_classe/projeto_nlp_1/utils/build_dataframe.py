from itertools import chain
from math import log
from pandas import DataFrame, options

# Avoids panda's DataFrame columns being hidden when printing them
options.display.width = None

class NLPDataFrame():

    def __init__(self, lemmas: list[list[str]]) -> None:
        ''' Contructs the dataframe with the lemmas as the first column, and
        the tf, df, idf and tf_idf of the respective lemma as columns
        '''
        self.lemmas = lemmas
        self.flat_lemmas = self._flatten(lemmas)
        self.df = DataFrame({'tokens': sorted(set(self.flat_lemmas))})
        self._calculate()

    def __str__(self) -> str:
        return self.df.__str__()

    def head(self, head_number: int=5) -> DataFrame:
        return self.df.head(head_number)

    def _calculate(self) -> None:

        field_funcs = (
                ('tf', 'tokens', self._term_freq),
                ('df', 'tokens', self._doc_freq),
                ('idf', 'df', self._idf),
                )
        for new_field, field, func in field_funcs:
            self.df[new_field] = self.df[field].apply(func=func)
        self.df['tf_idfs'] = self._tf_idf()

    def _flatten(self, lemmas: list[list[str]]) -> list[str]:
        ''' Takes a list of list of strings as an argument and returns a
        flattened list of strings to be used to calculate the
        document_frequency '''
        return list(chain(*lemmas))

    def _term_freq(self, token: str) -> list[float]:
        ''' Takes a token as an argument and returns a list with the term
        frequencies of that token 
        TF = number of ocurrence of token in doc / len(doc)
        '''
        return [round(doc.count(token)/len(doc),5) for doc in self.lemmas]

    def _doc_freq(self, token: str) -> int:
        ''' Returns the df of a token 
            DF = flattened_docs.count(token)
        '''
        return self.flat_lemmas.count(token)
    
    def _idf(self, doc_freq: int) -> float:
        ''' Returns the idf float value of a token
            IDF = log(len_docs / (DF + 1))
        '''
        return log(len(self.flat_lemmas)/(doc_freq + 1))
    
    def _tf_idf(self) -> list[list[float]]:
        ''' Receives a dataframe object and calculates the tf_idf for each row and
        returns as a list of list of floats, were each float is the tf_idf of the
        token for each document it was in
        '''
        return [ [round(tf * self.df['idf'][i], 5) for tf in tfs]
                  for i, tfs in enumerate(self.df['tf']) ]
