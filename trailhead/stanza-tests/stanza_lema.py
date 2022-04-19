import stanza

nlp = stanza.Pipeline(lang='en', 
                      processors='tokenize,mwt,pos,lemma', 
                      dir='../../venv/models')

doc = nlp('Barack Obama was born in Hawaii.')
print(*[f'word: {word.text+" "}\tlemma: {word.lemma}' 
        for sent in doc.sentences for word in sent.words], sep='\n')
