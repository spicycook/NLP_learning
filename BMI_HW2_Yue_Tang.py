## author: Yue Tang (Goizueta Business School)
## email: yue.tang@emory.edu
## date: Sep 3, 2023

'''
#### STATEMENT ####
The first question, (a), (b), (c), and (d) are completed with the assistance of ChatGPT (https://chat.openai.com).
'''
##################################################################################

'''
HW2: Q1
1. Document statistics generation.
During the session, we worked with news documents. The structures and contents of documents from distinct domains may vary significantly. For this part of the homework, we are going to work with scientific publications. The ’science_text’ folder (inside the inclass_task_solutions folder) contains several scientific publication texts (with some noise, which is typical for web-scraped data).

a) For scientificpub1: (i) what is the length of the document in (i) sentences and (ii) words. How many word types are there?
b) What is the average length of an unprocessed token in the file.
c) What are the 10 most frequent terms in the document before and after preprocessing?
d) What are the 10 most frequent terms in the document that start with the letter ‘p’ (case insensitive). 
e) What type of noise is present in the file? Does it affect the statistics we generated?

'''
import nltk
#nltk.download('punkt')
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.probability import FreqDist
from nltk.corpus import stopwords
from nltk.stem import *
import string
stemmer = PorterStemmer()
# Define the path to the file
file_path = '/Users/yue/Desktop/USA/Emory Study/2023Fall/2023Fall_BMI550_1_NLP/HW/science_text/scientificpub1'

# Open and read the file
with open(file_path, 'r', encoding='utf-8') as file:
    text1 = file.read()

##################################################################################
# Q1 (a)
# Tokenize the text into sentences and words
sentences = sent_tokenize(text1)
words = word_tokenize(text1)

# Calculate the length of the document in sentences and words
num_sentences = len(sentences)
num_words = len(words)

# Calculate the number of word types (unique words)
word_types = set(words)
num_word_types = len(word_types)

# Print the results
print(f"Number of sentences: {num_sentences}") # 387
print(f"Number of words: {num_words}") # 10236
print(f"Number of word types: {num_word_types}") # 1710

##################################################################################
# Q1 (b) # unprocessed = without removing stopwords and stemming
# Tokenize the text into words
words = word_tokenize(text1)

# Calculate the total length of all tokens
total_token_length = sum(len(word) for word in words)

# Calculate the average length of a token
average_token_length = total_token_length / len(words)

# Print the average token length
print(f"Average token length: {average_token_length:.2f} characters")

##################################################################################
# Q1 (c) 
# the 10 most frequent terms in the document before preprocessing?
freq_dist = FreqDist(words)
freq_10_before = freq_dist.most_common(10)
print('The 10 most frequent terms in the document before preprocessing are: \n')
for word, frequency in freq_10_before:
    print(f"{word}: {frequency}")

# the 10 most frequent terms in the document after preprocessing?
# lowering cases
words = [word.lower() for word in words]
# remove punctuation
words = [word for word in words if word not in string.punctuation]
# remove stopwords
stop_english = set(stopwords.words('english'))
words = [word for word in words if word not in stop_english]
freq_dist = FreqDist(words)
freq_10_after = freq_dist.most_common(10)
print('The 10 most frequent terms in the document after preprocessing are: \n')
for word, frequency in freq_10_after:
    print(f"{word}: {frequency}")

##################################################################################
# Q1 (d) 
# the 10 most frequent terms in the document that start with the letter ‘p’ (case insensitive). 
# Filter words that start with 'p' (case insensitive)
p_words = [word for word in words if word.startswith('p')]

# Create a frequency distribution of 'p' words
freq_dist = FreqDist(p_words)

# Find the 10 most common 'p' words
most_common_p_words = freq_dist.most_common(10)

# Print the 10 most common 'p' words and their frequencies
print('The 10 most frequent terms in the document that start with the letter ‘p’ are: \n')
for word, frequency in most_common_p_words:
    print(f"{word}: {frequency}")

##################################################################################
# Q1 (e) 
q1e = "The noise includes: (1) lower or upper cases, (2) stopwords, (3) punctuation, (4) newline symbol, (5) word variations (e.g., post, posts, posted, and posting), (6) possibly typos (I didn't check this in current file, but normally typos occur). \n These types of noise may not influence answers to (a) and (b), but if some typos occur a lot, this will definitely affect results in (c) and (d). For example, if some consistently type 'post' as 'pist', then stemming won't work to such typos, and our statistics may change."
print(q1e)


##################################################################################

'''
HW2: Q2
2. Corpus statistics generation.
A corpus is a set of texts that are useful for some task (typically annotated). Let’s consider the collection of files in the ‘science_text’ folder as a small corpus. Analyzing these documents may give us an idea of the domain of text.
a) What is the average document length for the whole corpus (i) in sentences, (ii) in words?
b) Which document in the folder has the highest lexical diversity?

'''

##################################################################################
# Q2 (a) 
# Read the three text files
file_path1 = '/Users/yue/Desktop/USA/Emory Study/2023Fall/2023Fall_BMI550_1_NLP/HW/science_text/scientificpub1'
file_path2 = '/Users/yue/Desktop/USA/Emory Study/2023Fall/2023Fall_BMI550_1_NLP/HW/science_text/scientificpub2'
file_path3 = '/Users/yue/Desktop/USA/Emory Study/2023Fall/2023Fall_BMI550_1_NLP/HW/science_text/scientificpub3'

# Define a function to do the task
# The function's output: length in sentences, length in words, number of word types/ unique words, aka lexical diversity.
def HW2Q2(f_path):
    # read file
    with open(f_path, 'r', encoding='utf-8') as file:
        txt1 = file.read()
    # basic preprocessing
    # Tokenize the text into sentences and words
    sentences = sent_tokenize(txt1)
    words = word_tokenize(txt1)
    # Calculate the length of the document in sentences
    num_sentences = len(sentences) # *** value to return
    # Calculate the number of words and word types (unique words)
    # Remove punctuation, lower cases, and stopwords (stopwords may account for word counts, but not domain identification)
    words = [word.lower() for word in words]
    words = [word for word in words if word not in string.punctuation]
    num_words = len(words) # *** value to return
    words = [word for word in words if word not in stop_english] # stopwords removed
    # We also do stemming to refine the lexical diversity
    words = [stemmer.stem(word) for word in words]
    word_types = set(words)
    num_word_types = len(word_types) # *** value to return
    # Return the results
    return {'num_sentences': num_sentences, 'num_words': num_words, 'lexical_diversity': num_word_types}

res1 = HW2Q2(file_path1)
res2 = HW2Q2(file_path2)
res3 = HW2Q2(file_path3)
print('Stats for file1: \n'+str(res1))
print('Stats for file2: \n'+str(res2))
print('Stats for file3: \n'+str(res3))
print('Average document length in sentences: \n'+str(sum([res1['num_sentences'], res2['num_sentences'], res3['num_sentences']])/3))
print('Average document length in words: \n'+str(sum([res1['num_words'], res2['num_words'], res3['num_words']])/3))

max_diversity = max([res1['lexical_diversity'], res2['lexical_diversity'], res3['lexical_diversity']])
if(res1['lexical_diversity']==max_diversity):
    t = 'scientificpub1'
elif(res2['lexical_diversity']==max_diversity):
    t = 'scientificpub2'
else:
    t = 'scientificpub3'

print('In the folder, ' + t + ' has the highest lexical diversity!')



