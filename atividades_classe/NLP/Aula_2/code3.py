# Import Dictionary
from nltk.data import path
path.append('src/nltk_data');
from gensim.corpora.dictionary import Dictionary
from src.nlp_utils import get_pre_process_wiki_articles

# Create a Dictionary from the articles: dictionary
articles = get_pre_process_wiki_articles()
dictionary = Dictionary(articles)

# Select the id for "computer": computer_id
computer_id = dictionary.token2id.get('computer')

# Use computer_id with the dictionary to print the word
print('the word', 'computer', 'has index', computer_id, 'in dictionary')

# Create a MmCorpus: corpus
corpus = [dictionary.doc2bow(article) for article in articles]

# Print the first 10 word ids with their frequency counts from the fifth document
print(corpus[5][:10])

#the word computer has index 104 in dictionary
#[(0, 33), (12, 9), (23, 1), (29, 4), (31, 1), (42, 1), (73, 1), (87, 19),
#(102, 1), (104, 7), (107, 1)]
