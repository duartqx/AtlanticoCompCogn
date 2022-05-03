from glob import glob
import pdfplumber
import re
import stanza

def glob_pdfs(directory: str) -> list[str]:

    if not directory.endswith('/'): 
        # In case the directory string doesn't ends with a slash, adds one
        directory = directory + '/'

    pdfs = glob(directory + '*.pdf')

    if len(pdfs) == 0:
        raise LookupError('directory (str) must contain all'
        ' the pdf files you want to grab the text from.')
    return pdfs

def read_pdf(directory: str='input') -> tuple[str, int]:
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
    pdfs: list[str] = glob_pdfs(directory)

    for pdf in pdfs:
        # Reads pdf text content with pdfplumber and cats to extracted_text
        with pdfplumber.open(pdf) as fh:
            for i in range(len(fh.pages)):
                extracted_text += fh.pages[i].extract_text()

    return extracted_text, len(pdfs)

def pretokenize(to_tokenize: str, 
             to_sub: dict[str, str]={},
             stopwords: str='resources/stopwords.txt') -> list[list[str]]:
    '''

    '''
    stop_words = [word.strip() for word in open(stopwords).readlines()]

    if to_sub: 
    # If the dictionary with regexes was passed, loops all keys and do re.sub
        for key in to_sub:
            to_tokenize = re.sub(key, to_sub[key], to_tokenize)

    sntnces: list[list[str]] = [[t.lower() for t in sentence.split()
                                 if t.lower() not in stop_words]
                                 for sentence in to_tokenize.split('.')
                                 if len(sentence)>1 
                                 # This if avoids empty lists in the result
                               ]
    return sntnces

def lemmanize(sntnc_tokens: list[list[str]], 
              models_dir='resources/stanza_models/') -> list[str]:
    '''

    '''
    nlp = stanza.Pipeline(lang='pt', processors='tokenize,lemma', 
                          dir=models_dir,
                          tokenize_pretokenized=True)
    doc = nlp(sntnc_tokens)
    return [word.lemma for sent in doc.sentences for word in sent.words]


if __name__ == '__main__':

    extracted_text, number_of_documents = read_pdf()

    to_sub = {
            # Essas duas primeiras chaves são para remover todas os números de
            # notas de rodapé, /100.000, porcentagens, pontuações e números
            # romanos, o ' I ' com espaços é importante para não remover I do
            # início de palavras
            r'I+ |IV| V |\-se': ' ',
            r'I\.': '.',
            r'\(\d.+\)|\/100\.000|\d+%|\d+': ' ', 
            r'\,|\(|\)|\—|\:|\;': ' ',
            # Caso bem específico dos input que recebemos, por isso escolhemos
            # deixar este dicionário externo à função e requerir um dicionário
            # como argumento para clean_up
            'Rio de Janeiro': 'RJ', 
            'São Paulo': 'SP', 
            'Reino Unido': 'UK',
            r'\/| - ':'  ',
            }

    sentences_tokenized = pretokenize(extracted_text, to_sub)

    lemmas = lemmanize(sentences_tokenized)

    print(lemmas)
