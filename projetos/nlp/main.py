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
        # This set of key value pair will be used with re.sub to clean up the
        # text of I to V in roman numbers, -se, long dashes, vs and replacing
        # places names with just their initials, so that their names don't end
        # up separated in multiple tokens
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

    plot_nx(df, norm=100000, k=0.38, iterations=200, seed=100, savefig=True)
    # Iteration < 50 the neighbors make a somewhat ring around some of the
    # keywords at the center (tested many times with iteration=8 or
    # iteration=17, but it gets harder to see to who each of the neighbor word
    # links to, while iterations > 50 things get a little better to read and
    # became the prefered numbers (specially 80, 100 and 200)

if __name__ == '__main__':

    main()
