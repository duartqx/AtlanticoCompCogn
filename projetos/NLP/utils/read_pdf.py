from glob import glob
from os import path
import pdfplumber # type: ignore

class PDFNotFoundError(LookupError): pass

def glob_pdfs(directory: str) -> list[str]:
    '''
    glob_pdf returns a string with all the pdf files glob.glob could find
    inside directory
    Arg
        directory (str): the name of the folder where you have all the pdfs you
        want to scrape the text from, if any pdfs that you don't want to scrape
        are also inside the directory it will grab it as well, so make sure
        only save the ones you want on directory
    '''

    pdfs: list[str] = glob(path.join(directory, '*.pdf'))

    if len(pdfs) == 0:
        raise PDFNotFoundError('directory (str) must contain all'
        ' the pdf files you want to grab the text from.')
    return pdfs

def read_pdf(pdf: str) -> str:
    '''
    Reads pdf file using pdfplumber and returns the text content that it could
    grab from all pages as a single concatenated string
    Arg
        pdf (str): the pdf path you want the function to read
    '''
    extracted_text: str = ''

    with pdfplumber.open(pdf) as fh:
        for page in fh.pages:
            extracted_text += page.extract_text()

    return extracted_text
