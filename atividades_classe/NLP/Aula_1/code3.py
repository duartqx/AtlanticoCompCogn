import re
from nltk.tokenize import sent_tokenize
from nlp_utils import get_sample_Santo_Graal

scene_one =  get_sample_Santo_Graal()
sentences =  sent_tokenize(scene_one)
print(sentences[3])
# Search for the first occurrence of "coconuts" in scene_one: match
match = re.search(r'coconuts', scene_one)

# Print the start and end indexes of match
print(match.start(), match.end())
print(match)

# Write a regular expression to search for anything in square brackets: pattern1
pattern1 = r"\[.*?\]" # Finds only the first
#pattern1 = r"\[.*\]" # Finds both

# Use re.search to find the first text in square brackets
print(re.search(pattern1, scene_one))

# Find the script notation at the beginning of the fourth sentence and print it
pattern2 = r"[a-zA-Z]+\:"
print(re.search(pattern2, sentences[3]))
print(re.findall(pattern2, scene_one))
