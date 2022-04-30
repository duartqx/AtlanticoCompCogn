from nlp_utils import get_sample_Santo_Graal
from nltk.tokenize import regexp_tokenize
import matplotlib.pyplot as plt
import re

# Split the script into lines: lines
holy_grail = get_sample_Santo_Graal()
lines = holy_grail.split('\n')

# Replace all script lines for speaker
pattern = "[A-Z]{2,}(\s)?(#\d)?([A-Z]{2,})?: "
lines = [re.sub(pattern, '', l.strip()) for l in lines]

# Tokenize each line: tokenized_lines
#tokenized_lines = [regexp_tokenize(s,r'\w+') for s in lines]
# r'\w+' divides words with apostrophe like "they'd" "it's"
# while r"[a-zA-Z']+" that has the apostrophe inside the brackets keep words
# that has it as just one
tokenized_lines = [regexp_tokenize(s,r"[a-zA-Z']+") for s in lines]
print(tokenized_lines)

# Make a frequency list of lengths: line_num_words
line_num_words = [len(t_line) for t_line in tokenized_lines]

# Plot a histogram of the line lengths
plt.hist(line_num_words)

# Show the plot
plt.show()
