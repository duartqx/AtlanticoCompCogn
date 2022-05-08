from utils.dataframe import NLPDataFrame
from utils.grab_lemmas import grab_lemmas
from utils.plot_nx import plot_nx

def main(to_csv: bool=False) -> None:
    ''' The main function of the project, first executes grab_lemmas that
    scrapes for the text, cleans and lemmanizes it and then builds a
    pandas.DataFrame 
    Arg:
        to_csv (bool): If set to True, saves the full dataframe to a csv file
    '''
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

    # Grabbing the lemmas
    l: list[list[str]] = grab_lemmas(to_sub=to_sub, 
                                     pdf_dir='corpus',
                                     stanza_dir='resources/stanza_models/',
                                     stop_words='resources/stopwords.txt')

    df = NLPDataFrame(l)
    # Builds the dataframe using the lemmas from grab_lemmas and saves them in
    # the 'tokens' column and use them as base to calculate the other columns.
    # columns: ['tokens','tf','tf_mean','df','idf','tf_idfs','tf_idf_mean']
    if to_csv:
        df.to_csv('dataframe.csv', index=False)

    plot_nx(df, norm=200000, spacing=0.38, iterations=80, seed=200, savefig=True)
    # Iteration < 50 the neighbors make a somewhat ring around some of the
    # keywords at the center (tested many times with iteration=8 or
    # iteration=17, but it gets harder to see to whom each of the neighbor word
    # links to

if __name__ == '__main__':

    main()
