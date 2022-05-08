from typing import TypeAlias
from os import path
import stanza

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
