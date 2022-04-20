import nltk

download_dir = './nltk_data'

modules = [
        'averaged_perceptron_tagger',
        'omw-1.4', # Had a Traceback without this one
        'punkt',
        'stopwords',
        'twitter_samples',
        'wordnet',
        ]

for module in modules:
    nltk.download(module, download_dir=download_dir)
