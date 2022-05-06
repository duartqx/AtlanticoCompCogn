from .clean_up import flatten
from math import log
from pandas import DataFrame, options

# Avoids panda's DataFrame columns being hidden when printing them
options.display.width = None

class NLPDataFrame():

    def __init__(self, lemmas: list[list[str]], 
                 f_len: int= 6,log: bool=False) -> None:
        ''' Contructs a dataframe with the lemmas as the first
        column, and calculates the tf, tf_mean, df, idf,
        tf_idf and tf_idf_mean for each lemma and adds them as
        columns '''
        self.f_len = f_len
        self.docs_qntty: int = len(lemmas)
        self.lemmas = lemmas
        self.flat_lemmas: list[str] = flatten(lemmas)
        self.log = log

        self.df = DataFrame({'tokens': sorted(set(self.flat_lemmas))})
        
        self._calculate()

    def __repr__(self) -> str:
        ''' Returns the string representation of self.df '''
        return self.df.__repr__()

    def head(self, head_number: int=5) -> DataFrame:
        ''' Returns the called result of self.df.head() '''
        return self.df.head(head_number)

    def _calculate(self) -> None:
        ''' Calculates the tf, tf_mean, df, idf, tf_idf and tf_idf_mean and
        adds them as columns to self.df '''
        self.df['tf'] = self.df['tokens'].apply(self._term_freq)
        self.df['tf_mean'] = self.df['tf'].apply(self._mean)
        self.df['df'] = self.df['tokens'].apply(self._doc_freq)
        self.df['idf'] = self.df['df'].apply(self._idf)
        self.df['tf_idfs'] = self._tf_idf()
        self.df['tf_idf_mean'] = self.df['tf_idfs'].apply(self._mean)

    def _term_freq(self, token: str) -> list[float]:
        ''' Takes a token as an argument and returns a list with the term
        frequencies of that token '''
        return [round(d.count(token)/len(d), self.f_len) for d in self.lemmas]

    def _doc_freq(self, token: str) -> int:
        ''' Calculates the document frequency of token '''
        return self.flat_lemmas.count(token)
    
    def _idf(self, doc_freq: int) -> float:
        ''' Calculates the inverse document frequency '''
        if self.log:
            return log(self.docs_qntty/(doc_freq + 1))
        return round(self.docs_qntty/(doc_freq + 1), self.f_len)
    
    def _tf_idf(self) -> list[list[float]]:
        ''' Calculates the tf_idf for each row in self.df and returns as a list
        of list of floats, were each float is the tf_idf of the token for each
        document it was in '''
        return [ [round(tf * self.df['idf'][i], self.f_len) for tf in tfs]
                  for i, tfs in enumerate(self.df['tf']) ]

    def _mean(self, i: list[float]) -> float:
        ''' Returns the mean of tf or tf-idf '''
        return round(sum(i)/self.docs_qntty, self.f_len)
