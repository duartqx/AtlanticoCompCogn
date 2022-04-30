import re
# Write a pattern to match sentence endings: sentence_endings
my_string = "Let's write RegEx!  Won't that be fun?  " \
            "I sure think so.  Can you find 4 sentences? " \
            "Or perhaps, all 19 words?"

sentence_endings = r'[?.!]\s*'

# Split my_string on sentence endings and print the result
print(re.split(sentence_endings, my_string), '\n')

# Find all capitalized words in my_string and print the result
capitalized_words = r'[A-Z]\w+'
print(re.findall(capitalized_words, my_string), '\n')

# Split my_string on spaces and print the result
spaces = r'\s+'
all_words = re.split(spaces, my_string)

# This is getting the digits as words, giving a total of len == 21
print(all_words, len(all_words), '\n')

only_words = re.findall(r"[a-zA-Z']+", my_string)
print(only_words, len(only_words), '\n')
# This way, len == 19

# Find all digits in my_string and print the result
digits = r'\d+'
print(re.findall(digits, my_string))
