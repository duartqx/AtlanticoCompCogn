from glob import glob
import pdfplumber

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

    if not directory.endswith('/'): 
        # In case the directory string doesn't ends with a slash, adds one
        directory += '/'

    pdfs: list[str] = glob(directory + '*.pdf')

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
        # Reads pdf text content with pdfplumber
        for i in range(len(fh.pages)):
            # concatenates all text to 'extracted_text'
            extracted_text += fh.pages[i].extract_text()

    return extracted_text
