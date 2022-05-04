from preprocess.clean_up import clean_up
from preprocess.lemmanizer import lemmanize
from preprocess.read_pdf import glob_pdfs, read_pdf

def main() -> None:

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

    pdfs: list[str] = glob_pdfs(directory='corpus')

    extracted_texts: list[str] = [read_pdf(pdf) for pdf in pdfs]

    clean_texts: list[str] = [clean_up(ext, to_sub) for ext in extracted_texts]

    lemmas: list[list[str]] = [lemmanize(t) for t in clean_texts]
    print(lemmas)


if __name__ == '__main__':

    main()
