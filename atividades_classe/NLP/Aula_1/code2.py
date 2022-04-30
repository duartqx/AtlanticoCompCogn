# Import necessary modules
from nltk.tokenize import sent_tokenize , word_tokenize
from nlp_utils import get_sample_Santo_Graal

# Split scene_one into sentences: sentences
scene_one = get_sample_Santo_Graal()
sentences = sent_tokenize(scene_one)

# Use word_tokenize to tokenize the fourth sentence: tokenized_sent
tokenized_sent = word_tokenize(sentences[3])
#print(tokenized_sent)

# Make a set of unique tokens in the entire scene: unique_tokens
unique_tokens = set(word_tokenize(scene_one))

# Print the unique tokens result
print(unique_tokens, len(unique_tokens))
# Output: {'do', 'our', 'or', 'What', 'You', 'go', 'could', 'point', 'yeah',
# '.', 'lord', 'simple', 'climes', '...', 'empty', 'and', 'coconut', 'needs',
# 'Oh', 'knights', 'from', 'second', 'It', 'Pendragon', 'covered', 'may',
# 'where', 'of', 'at', 'husk', '2', 'are', 'coconuts', 'European', '1',
# 'order', 'That', 'England', 'winter', "'s", "'", 'times', 'who', 'So',
# 'found', 'defeator', 'right', 'got', 'will', 'have', 'carry', 'swallows',
# 'Yes', 'course', ']', 'they', 'under', 'forty-three', 'trusty', 'We',
# 'suggesting', 'Court', 'he', 'Whoa', 'martin', 'but', 'other', 'dorsal',
# 'Where', 'velocity', 'you', 'warmer', 'every', 'fly', 'Halt', 'Mercea', 'it',
# 'wings', 'search', 'Britons', 'temperate', 'I', 'grip', "'ve", 'Pull',
# 'sovereign', 'speak', 'Am', 'to', 'kingdom', 'SOLDIER', 'son', 'wind',
# 'Will', 'an', 'tropical', 'But', 'bird', 'sun', 'King', 'agree', "n't",
# 'Supposing', 'must', 'me', 'question', 'together', 'since', 'SCENE', 'Found',
# 'migrate', 'the', 'bangin', 'join', 'They', 'KING', 'ridden', 'snows', 'A',
# 'non-migratory', 'two', 'Well', '[', 'African', 'five', 'Who', 'Patsy',
# 'get', 'strand', 'in', 'back', '!', 'court', 'just', 'one', 'Are', 'does',
# 'ask', 'them', 'south', 'grips', 'by', 'tell', 'servant', 'through',
# 'minute', 'The', 'interested', 'carrying', "'m", 'that', 'Please', 'matter',
# 'mean', 'breadth', 'weight', 'line', 'plover', 'halves', 'feathers', 'these',
# 'ARTHUR', 'your', 'Wait', 'castle', 'horse', 'creeper', 'on', 'Ridden',
# 'length', 'my', 'zone', 'its', 'ounce', 'air-speed', 'be', 'swallow', 'here',
# 'house', 'bring', 'Arthur', 'use', 'am', 'with', 'maintain', 'not',
# 'guiding', 'wants', '?', 'Uther', 'beat', 'carried', 'a', 'ratios', 'then',
# '#', 'maybe', "'em", 'Listen', 'why', 'this', 'land', 'In', 'Camelot',
# 'master', 'pound', "'d", 'using', '--', 'yet', ',', 'Not', 'strangers',
# 'all', 'seek', 'is', 'there', 'goes', 'Saxons', 'anyway', 'No', 'held',
# 'clop', "'re", 'if', ':'}
