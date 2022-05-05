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
        self.lemmas: list[list[str]] = lemmas
        self.flat_lemmas: list[str] = self._flatten_lemmas()
        self.df = DataFrame({'tokens': sorted(set(self.flat_lemmas))})
        self._calculate()

    def __str__(self) -> str:
        return self.df.__str__()

    def head(self, head_number: int=5) -> DataFrame:
        ''' Returns the called result of self.df.head() '''
        return self.df.head(head_number)

    def _calculate(self) -> None:
        ''' Calculates the tf, df, idf and tf_idf and adds them as columns to
        self.df '''
        field_funcs = (
                ('tf', 'tokens', self._term_freq),
                ('df', 'tokens', self._doc_freq),
                ('idf', 'df', self._idf),
                )
        for new_field, field, func in field_funcs:
            self.df[new_field] = self.df[field].apply(func=func)
        self.df['tf_idfs'] = self._tf_idf()

    def _flatten_lemmas(self) -> list[str]:
        ''' Takes self.lemmas (list[list[str]]) and flats it to a single
        dimension list '''
        return list(chain(*self.lemmas))

    def _term_freq(self, token: str) -> list[float]:
        ''' Takes a token as an argument and returns a list with the term
        frequencies of that token 
        '''
        return [round(doc.count(token)/len(doc),5) for doc in self.lemmas]

    def _doc_freq(self, token: str) -> int:
        ''' Calculates the document frequency of token '''
        return self.flat_lemmas.count(token)
    
    def _idf(self, doc_freq: int) -> float:
        ''' Calculates the inverse document frequency '''
        return log(len(self.flat_lemmas)/(doc_freq + 1))
    
    def _tf_idf(self) -> list[list[float]]:
        ''' Calculates the tf_idf for each row in self.df and returns as a list
        of list of floats, were each float is the tf_idf of the token for each
        document it was in
        '''
        return [ [round(tf * self.df['idf'][i], 5) for tf in tfs]
                  for i, tfs in enumerate(self.df['tf']) ]
