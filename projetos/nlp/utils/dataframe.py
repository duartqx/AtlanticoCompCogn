from math import log
from pandas import DataFrame, Series, options # type: ignore
from typing import Any, Union

# Avoids panda's DataFrame columns being hidden when printing
options.display.width = None

class NLPDataFrame:

    def __init__(self, lemmas: list[list[str]], 
                 f_len: int= 6, idf_log: bool=False) -> None:
        ''' Contructs a dataframe with the lemmas as the first
        column, and calculates the tf, tf_mean, df, idf,
        tf_idf and tf_idf_mean for each lemma and adds them as
        columns 
        Args:
            lemmas (list[list[str]]): the list of lemmanized tokens
            f_len (int): the limit of float length when using round() on 
            _term_freq, _idf, _tf_idf and _mean
            idf_log (bool): if set to True then the idf will be calculated with
            log applied, if False log is not used (this avoids -0.0 problems)
        '''
        self.f_len = f_len
        self.docs_qntty: int = len(lemmas)
        self.lemmas = lemmas
        self.flat_lemmas: list[str] = self._flatten()
        self.idf_log = idf_log

        self.df = DataFrame({'tokens': sorted(set(self.flat_lemmas))})
        
        self._calculate()

    def __getitem__(self, key: str) -> Union[DataFrame, Series]:
        ''' Access elements from self.df when indexing on an instance of
        NLPDataFrame. Returns a new pandas.DataFrame or pandas.Series object 
        '''
        return self.df.__getitem__(key)

    def __getattr__(self, attr: str) -> Any:
        ''' Access methods and attributes of self.df when using dot notation on
        a instance of NLPDataFrame '''
        return self.df.__getattr__(attr)

    def __repr__(self) -> str:
        ''' Returns the string representation of self.df '''
        return self.df.__repr__()

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
        return sum(token in doc for doc in self.lemmas)
    
    def _idf(self, doc_freq: int) -> float:
        ''' Calculates the inverse document frequency '''
        if self.idf_log:
            return round(log(self.docs_qntty/(doc_freq + 1)), self.f_len)
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

    def _flatten(self) -> list[str]:
        ''' Takes self.lemmas (list[list[str]]) and flats it to a single
        dimension list '''
        flattened = []
        for t in self.lemmas:
            flattened += t
        return flattened
