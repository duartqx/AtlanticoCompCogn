from utils.clean_up import clean_up, flatten
from utils.dataframe import NLPDataFrame
from utils.lemmanizer import lemmanize
from utils.read_pdf import glob_pdfs, read_pdf

def main(to_csv: bool=False) -> None:

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

    # Mounts the list of pdfs it can find on the corpus folder
    pdfs: list[str] = glob_pdfs(directory='corpus')
    # Reads all pdfs and scrapes it's text
    scraped_text: list[str] = [read_pdf(pdf) for pdf in pdfs]
    # Cleans up the text with re.sub and str.translate to remove stopwords,
    # punctuation and everything in to_sub
    clean_texts: list[str] = [clean_up(text, to_sub) for text in scraped_text]
    # Lemmanizes the text with the module stanza
    lemmas: list[list[str]] = [lemmanize(t) for t in clean_texts]

    df = NLPDataFrame(lemmas)
    # Builds the dataframe with the keys:
    # ['tokens','tf','tf_mean','df','idf','tf_idfs','tf_idf_mean']

    if to_csv:
        df.to_csv('dataframe.csv')

    df_slice = df[['tokens', 'tf_mean', 'tf_idf_mean']]
    five_largest_tf_idf = df_slice.nlargest(5, 'tf_idf_mean')

    fl_list = five_largest_tf_idf['tokens'].tolist()
    # ['cirurgia', 'paciente', 'pequeno', 'dado', 'realizar']

    print(five_largest_tf_idf)

    # Check on lemmas the closest words to fl_list
    # TODO
    # How to find the closes words in the df after getting then from the
    # previous step
    print(df[df['tokens'].str.contains('cirurgia', regex=False)])

if __name__ == '__main__':

    main()
