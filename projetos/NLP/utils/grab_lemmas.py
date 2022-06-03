from .read_pdf import glob_pdfs, read_pdf
from os import path
from re import sub
from string import punctuation
from typing import TypeAlias
import stanza # type: ignore

StanzaPipeline: TypeAlias = stanza.pipeline.core.Pipeline
StanzaDoc: TypeAlias = stanza.models.common.doc.Document

def lemmanize(to_lemanize: str, models_dir: str) -> list[str]:
    '''
    Returns a list of lemmanized words
    Args:
        sntnc_tokens (str): a string with pre_cleaned words
        that will be lemmanized by stanza.Pipeline
        models_dir (str): the path location to the stanza_models that you want
        to apply

    '''
    nlp: StanzaPipeline = stanza.Pipeline(lang='pt', 
                                          processors='tokenize,lemma', 
                                          dir=models_dir, 
                                          verbose=False)
    doc: StanzaDoc = nlp(to_lemanize)
    return [word.lemma for sent in doc.sentences for word in sent.words]

def clean_up(to_clean: str, stopwords: str, to_sub: dict[str, str]={}) -> str:
    '''
    Cleans up the text a little with re.sub and str.translate
    Args
        to_clean (str): the text that you want to tokenize
        to_sub (dict[str, str]): a dictionary to be used with re.sub that maps
        patterns with the string to be replaced with
        stopwords (str): the path location to the file containing the stopwords
    '''

    with open(stopwords) as fh:
        stop_words: set[str] = {word.strip() for word in fh.readlines()}

    if to_sub: 
        for key, value in to_sub.items():
            to_clean = sub(key, value, to_clean)

    # Removing all punctuations and lowering the case
    to_clean = to_clean.translate(str.maketrans(' ', ' ', punctuation)).lower()

    return ' '.join(w.strip() for w in to_clean.split() 
                    if w.strip() not in stop_words)

def grab_lemmas(to_sub: dict[str, str], 
                pdf_dir: str, 
                stanza_dir: str,
                stop_words: str) -> list[list[str]]:
    ''' globs for the pdfs, scrapes the text out of them, cleans up and
    lemmanizes all strings 
    Args:
        to_sub (dict[str, str]): a dictionary with regex as keys and the string
        to replace the regex with
        directory (str): the name of the folder with the pdf files
    '''
    # Mounts the list of pdfs it can find on the corpus folder
    pdfs: list[str] = glob_pdfs(directory=pdf_dir)
    # Reads all pdfs and scrapes it's text
    scraped_text: list[str] = [read_pdf(pdf) for pdf in pdfs]
    # Cleans up the text with re.sub and str.translate to remove stopwords,
    # punctuation and everything in to_sub
    clean_texts: list[str] = [clean_up(text, stop_words, to_sub) 
                              for text in scraped_text]
    # Lemmanizes the text with the module stanza
    lemmas: list[list[str]] = [lemmanize(t, stanza_dir) for t in clean_texts]

    return lemmas
