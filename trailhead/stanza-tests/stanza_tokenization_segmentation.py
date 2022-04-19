import stanza

# It's necessary to first download the default model ('en' lang)
# stanza.download('en', model_dir='../../venv/models')

#print([sentence.text for sentence in doc.sentences])

def print_pipeline(sntnc, directory, no_split=False, pretok=False):
    # the dir= keyword argument is necessary so that stanza.Pipeline can know
    # where I've downloaded the 'en' model
    nlp = stanza.Pipeline(lang='en',  processors='tokenize', 
                          dir=directory, tokenize_no_ssplit=no_split,
                          tokenize_pretokenized=pretok)
    doc = nlp(sntnc)
    for i, sentence in enumerate(doc.sentences, start=1):
        print(f'\n===== Sentence {i} tokens =====')
        print(*[f'id: {token.id}\ttext: {token.text}' 
                for token in sentence.tokens], sep='\n')

if __name__ == '__main__':

    directory = '../../venv/models'
    #sntnc = 'This is a test sentence for stanza. This is another sentence.'
    #print_pipeline(sntnc, directory)

    # sntnc = 'This is a sentence.\n\nThis is a second. This is a third.'
    # print_pipeline(sntnc, directory, no_split=True)
    # no_split doesn't separates the second and third sentence

    # sntnc = 'This is token.ization done my way!\nSentence split, too!'
    # print_pipeline(sntnc, directory, pretok=True)

    sntnc_list = [['This', 'is', 'token.ization', 'done', 'my', 'way!'], 
                  ['Sentence', 'split,', 'too!']]
    print_pipeline(sntnc_list, directory, pretok=True)
    #print_pipeline(sntnc_list, directory, pretok=False)
    # If pretok=False and no_split=False raises AssertionError: If neither
    # 'pretokenized' or 'no_ssplit' option is enabled, the inp ut to the
    # TokenizerProcessor must be a string or a Document object.
    # This is probably something added in an update after the tutorial was done
