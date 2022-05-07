from utils.dataframe import DataFrame, NLPDataFrame
from utils.grab_lemmas import grab_lemmas

def get_neighbors(flat_lemmas: list[str], to_find: str) -> list[str]:
    ''' Returns a list of neighbors of to_find in tokens '''
    flat_lemmas_len: int = len(flat_lemmas)
    neighbors: list[str] = []
    start: int = 0
    while True:
        try:
            index: int = flat_lemmas.index(to_find, start)
            # Only tries to append index+1 if it's smaller than
            # flat_lemmas_len-1
            if index + 1 < flat_lemmas_len-1:
                neighbors.append(flat_lemmas[index+1])
            neighbors.append(flat_lemmas[index-1])
            start = (index + 1) 
        except ValueError:
            break
    return neighbors

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

    lemmas = grab_lemmas(to_sub=to_sub, directory='corpus')

    df = NLPDataFrame(lemmas)
    # Builds the dataframe using the lemmas from grab_lemmas and saves them in
    # the 'tokens' column and use them as base to calculate the other columns.
    # columns: ['tokens','tf','tf_mean','df','idf','tf_idfs','tf_idf_mean']
    if to_csv:
        df.to_csv('dataframe.csv', index=False)

    df_slice = df[['tokens', 'tf_mean', 'tf_idf_mean']]
    five_largest_tf_idf = df_slice.nlargest(5, 'tf_idf_mean')

    fl_list = five_largest_tf_idf['tokens'].tolist()
    # ['cirurgia', 'paciente', 'pequeno', 'dado', 'realizar']

    # print(five_largest_tf_idf)
    neighbors: dict[str, list[int]] = {t: get_neighbors(df.flat_lemmas, t) 
                                       for t in fl_list}

    print(neighbors)

    # Check on lemmas the closest words to fl_list
    # TODO
    # How to find the closes words in the df after getting then from the
    # previous step
    # print(df[df['tokens'].str.contains('cirurgia', regex=False)])

if __name__ == '__main__':

    main()
