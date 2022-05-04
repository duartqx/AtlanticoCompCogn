from preprocess.clean_up import clean_up
from preprocess.lemmanizer import lemmanize
from preprocess.read_pdf import glob_pdfs, read_pdf

def main() -> None:

    to_sub = {
            # O conjunto chave e valor neste dicionário serão usados com re.sub
            # para limpar os textos de números romanos, -se, tração, vs e
            # transformando o nome de alguns locais em sigla para não se
            # separarem na hora que a string for separada em tokens
            r'I\.|I+ |IV| V |\-se|\—| vs|\d+': ' ',
            'Rio de Janeiro': 'RJ', 
            'São Paulo': 'SP', 
            'Reino Unido': 'UK',
            }

    pdfs: list[str] = glob_pdfs(directory='corpus')

    extracted_texts: list[str] = [read_pdf(pdf) for pdf in pdfs]

    clean_texts: list[str] = [clean_up(text, to_sub) for text in extracted_texts]

    lemmas: list[list[str]] = [lemmanize(t) for t in clean_texts]

    print(lemmas)


if __name__ == '__main__':

    main()
