from .clean_up import flatten
from math import log
from pandas import DataFrame, options

# Avoids panda's DataFrame columns being hidden when printing them
options.display.width = None

class NLPDataFrame():

    def __init__(self, docs_qntty:int, lemmas: list[list[str]]) -> None:
        ''' Contructs the dataframe with the lemmas as the first column, and
        the tf, df, idf and tf_idf of the respective lemma as columns
        '''

        self.docs_qntty = docs_qntty
        self.lemmas = lemmas
        self.flat_lemmas: list[str] = flatten(self.lemmas)

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
        self.df['tf'] = self.df['tokens'].apply(self._term_freq)
        self.df['df'] = self.df['tokens'].apply(self._doc_freq)
        self.df['idf'] = self.df['df'].apply(self._idf)
        self.df['tf_idfs'] = self._tf_idf()
        self.df['tf_idf_mean'] = self.df['tf_idfs'].apply(self._tf_idf_mean)

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
        #return log(self.docs_qntty/(doc_freq + 1))
        return self.docs_qntty/(doc_freq + 1)
    
    def _tf_idf(self) -> list[list[float]]:
        ''' Calculates the tf_idf for each row in self.df and returns as a list
        of list of floats, were each float is the tf_idf of the token for each
        document it was in
        '''
        return [ [round(tf * self.df['idf'][i], 5) for tf in tfs]
                  for i, tfs in enumerate(self.df['tf']) ]

    def _tf_idf_mean(self, tf_idf: list[float]) -> float:
        return sum(tf_idf)/self.docs_qntty
