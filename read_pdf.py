from glob import glob
from string import punctuation
from typing import TypeAlias
import pdfplumber
import re
import stanza
import json

StanzaDoc: TypeAlias = stanza.models.common.doc.Document
StanzaPipeline: TypeAlias = stanza.pipeline.core.Pipeline

def glob_pdfs(directory: str) -> list[str]:
    '''
    glob_pdf returns a string with all the pdf files glob.glob could find
    inside directory
    Arg
        directory (str): the name of the folder where you have  all the pdfs
        you want to scrape the text from, if any pdfs that you don't want to
        scrape are also inside it will grab as well, so make sure only save the
        ones you want on directory
    '''

    if not directory.endswith('/'): 
        # In case the directory string doesn't ends with a slash, adds one
        directory += '/'

    pdfs: list[str] = glob(directory + '*.pdf')

    if len(pdfs) == 0:
        raise LookupError('directory (str) must contain all'
        ' the pdf files you want to grab the text from.')
    return pdfs

def read_pdf(pdf: str, directory: str='_data') -> str:
    '''
    Reads all pdf files inside directory using pdfplumber and globing all pdf
    with glob
    And returns the text content that it could grab from all pdfs as a single
    concatenated string and the number of pdf documents it found to be used to
    find the IDF later (IDF = log(qtd de documentos / (DF + 1)))
    Arg
        directory (str): the name of the directory that you want to glob for
        pdf files
    '''
    extracted_text: str = ''
    #pdfs: list[str] = glob_pdfs(directory)

    with pdfplumber.open(pdf) as fh:
        # Reads pdf text content with pdfplumber
        for i in range(len(fh.pages)):
            # concatenates all text to 'extracted_text'
            extracted_text += fh.pages[i].extract_text()

    return extracted_text

def clean_up(to_clean: str, to_sub: dict[str, str]={},
             stopwords: str='resources/stopwords.txt') -> str:
    '''
    Cleans up the text a little with re.sub and str.translate
    Args
        to_clean (str): the text that you want to tokenize
        to_sub (dict[str, str]): a dictionary to be used with re.sub that maps
        patterns with the string to be replaced with
        stopwords (str): the path location to the file containing the stopwords
    '''

    with open(stopwords) as fh:
        stop_words: set[str] ={word.strip() for word in fh.readlines()}

    if to_sub: 
    # If the dictionary with regexes was passed, loops all keys and do re.sub
        for key in to_sub:
            to_clean = re.sub(key, to_sub[key], to_clean)

    # Removing all punctuations
    to_clean = to_clean.translate(str.maketrans('', '', punctuation))
    # lowering the case so that it matches stopwords
    to_clean = to_clean.lower()

    return ' '.join(w for w in to_clean.split() if w not in stop_words)


def lemmanize(to_lemanize: str, 
              models_dir: str='resources/stanza_models/',) -> list[str]:
    '''
    Returns a list of lemmanized words
    Args:
        sntnc_tokens (list[list[str]]): a list of list of pretokenized words
        that will be lemmanized by stanza.Pipeline
        models_dir (str): the path location to the stanza_models that you want
        to apply

    '''
    nlp: StanzaPipeline = stanza.Pipeline(lang='pt', 
                          processors='tokenize,lemma', 
                          dir=models_dir, verbose=False,)
    doc: StanzaDoc = nlp(to_lemanize)
    return [word.lemma for sent in doc.sentences for word in sent.words]


if __name__ == '__main__':

    pdfs: list[str] = glob_pdfs('_data')

    extracted_texts: list[str] = [read_pdf(pdf) for pdf in pdfs]

    #extracted_text, number_of_documents = read_pdf()

    to_sub = {
            # Essas duas primeiras chaves são para remover todas os números de
            # notas de rodapé, /100.000, porcentagens, pontuações e números
            # romanos
            r'I+ |IV| V |\-se|\—| vs': ' ',
            r'I\.': '.',
            r'100\.000|\d+': ' ', 
            # 'Rio de Janeiro': 'RJ', 
            # 'São Paulo': 'SP', 
            # 'Reino Unido': 'UK',
            'Rio de Janeiro': ' ', 
            'São Paulo': ' ', 
            'Reino Unido': ' ',
            r'\/| - ':'  ',
            }

    cleaned: list[str] = [clean_up(ext, to_sub) for ext in extracted_texts]
    lemmas: list[list[str]] = [lemmanize(t) for t in cleaned]
    print(lemmas)

#    with open('tokens.json', 'w') as f:
#        json.dump(lemmas, f)
