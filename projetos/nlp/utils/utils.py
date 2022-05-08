from itertools import chain

def flatten(lemmas: list[list[str]]) -> list[str]:
    ''' Takes self.lemmas (list[list[str]]) and flats it to a single
    dimension list '''
    return list(chain(*lemmas))
