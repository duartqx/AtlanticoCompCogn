from .clean_up import clean_up
from .lemmanizer import lemmanize
from .read_pdf import glob_pdfs, read_pdf

def grab_lemmas() -> list[list[str]]:
    ''' globs for the pdfs, scrapes the text out of them, cleans up and
    lemmanizes all strings '''
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

    return lemmas
