import stanza

try:
    nlp = stanza.Pipeline(lang='en', 
                          processors='tokenize,mwt,pos,lemma', 
                          dir='../../venv/models')
except stanza.pipeline.core.ResourcesFileNotFoundError:
    print("Lembre-se de alterar o keyword argument dir='../../venv/models'" 
          "para a pasta onde seu modelo está salvo")
    exit()

doc = nlp('Barack Obama was born in Hawaii.')
print(*[f'word: {word.text+" "}\tlemma: {word.lemma}' 
        for sent in doc.sentences for word in sent.words], sep='\n')
