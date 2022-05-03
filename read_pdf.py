from glob import glob
import pdfplumber
import re

def read_pdf(directory: str) -> (str, int):
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
    extracted_text = ''

    if not directory.endswith('/'): 
        # In case the directory string doesn't ends with a slash, adds one
        directory = directory + '/'

    pdfs = glob(directory + '*.pdf')

    if len(pdfs) == 0:
        raise LookupError('directory (str) must contain all'
        ' the pdf files you want to grab the text from.')

    for pdf in pdfs:
        with pdfplumber.open(pdf) as fh:
            for i in range(len(fh.pages)):
                extracted_text += fh.pages[i].extract_text()
    return extracted_text, len(pdfs)

def clean_up(to_tokenize: str, to_sub: dict[str, str]) -> list[str]:
    '''

    '''
    stop_words = [word.strip() for word in open('stopwords.txt').readlines()]

    if to_sub:
        for key in to_sub:
            to_tokenize = re.sub(key, to_sub[key], to_tokenize)

    tokens = [word.lower() 
              for word in to_tokenize.split() 
              if word.lower() not in stop_words]
    return tokens


if __name__ == '__main__':

    extracted_text, number_of_documents = read_pdf('_data')

    to_sub = {
            # Essas duas primeiras chaves são para remover todas os números de
            # notas de rodapé, /100.000, porcentagens, pontuações e números
            # romanos, o ' I ' com espaços é importante para não remover I do
            # início de palavras
            r'\(\d.+\)|\/100\.000|\d+%|\d+': ' ', 
            r'\,|\.|\(|\)|\—|\:|\;': ' ',
            r' I |II|III|IV| V |\-se': ' ',
            # Caso bem específico dos input que recebemos, por isso escolhemos
            # deixar este dicionário externo à função e requerir um dicionário
            # como argumento para clean_up
            'Rio de Janeiro': 'RJ', 
            'São Paulo': 'SP', 
            'Reino Unido': 'UK',
            r'\/| - ':'  ',
            }

    print(clean_up(extracted_text, to_sub))
