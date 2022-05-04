from string import punctuation
from re import sub

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
            to_clean = sub(key, to_sub[key], to_clean)

    # Removing all punctuations
    to_clean = to_clean.translate(str.maketrans('', '', punctuation))
    # lowering the case so that it matches stopwords
    to_clean = to_clean.lower()

    return ' '.join(w for w in to_clean.split() if w not in stop_words)
