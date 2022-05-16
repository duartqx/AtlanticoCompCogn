# Import Counter and word_tokenize
from collections import Counter
from nltk.tokenize import word_tokenize
from nltk.data import path
#from nltk import download
path.append('src/nltk_data');

#download('punkt', download_dir='src/nltk_data')

from src.nlp_utils import get_sample_article


article = get_sample_article()

# Tokenize the article: tokens
tokens = word_tokenize(article)

# Convert the tokens into lowercase: lower_tokens
lower_tokens = [t.lower() for t in tokens]

# Create a Counter with the lowercase tokens: bow_simple
bow_simple = Counter(lower_tokens)

# Print the 10 most common tokens
print(bow_simple.most_common(10))

#[(',', 151), ('the', 150), ('.', 89), ('of', 81), ("''", 69), ('to', 63),
#('a', 60), ('``', 47), ('in', 44), ('and', 41)]
