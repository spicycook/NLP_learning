'''
BMI 550: Applied BioNLP

Assignment 1

@author Yue Tang
Date: 09/20/2023
'''
import string
import re
import nltk
import itertools
import os
from fuzzywuzzy import fuzz
import pandas as pd
import numpy as np
pd.options.mode.chained_assignment = None  # default='warn'
import sys
sys.path.append('/Users/yue/Desktop/USA/Emory Study/2023Fall/2023Fall_BMI550_1_NLP/Assignments/Assignment1/')
from assignment1_functions import *

####################
'''
Workflow 1-3: get augmented Lexicon list, we just call function from assignment1_functions.py
'''
symps_dict = symptoms_dict()
len(symps_dict)
symps_dict = {key: value for key, value in symps_dict.items() if len(key.split()) <=4}
symps_dict = {key.lower(): value for key, value in symps_dict.items()}
len(symps_dict)
# print out our current lexicon
# for k,v in symps_dict.items():
#     print (k,'\t',v)
####################
####################

count = []
for key, value in symps_dict.items():
    # Split the string into words using whitespace as the delimiter
    words = key.split()
    # Count the number of words
    word_count = len(words)
    # Print the result
    count.append(word_count)

print('Stats of symptoms dict (mean and max): ', np.average(count), max(count))

'''
Workflow 4: with Lexicon ready, next step is grab 
Here, I use the regular expressions method in hw4_solutions.py file provided directly
'''
from collections import defaultdict
intest = '/Users/yue/Desktop/USA/Emory Study/2023Fall/2023Fall_BMI550_1_NLP/Assignments/Assignment1/Assignment1GoldStandardSet.xlsx'
test = pd.read_excel(intest)
test = test.dropna(how='any')
test_dict = defaultdict(list)
for i in range(0,test.shape[0]):
    ID = test.iloc[i]['ID']
    text = test.iloc[i]['TEXT']
    text = re.sub(r'(\.\n| \n)', ' ', text)
    matched_symps = grab_symp(text, symps_dict)
    # print(matched_symps)
    check_neg = neg_check(matched_symps)
    # print(check_neg)
    test_dict[ID] = check_neg
    print('Labelled', ID, '.', i+1, 'in', test.shape[0], 'subscribers.')
print(test_dict)
####################
####################
'''
Workflow 5: evaluation 
'''
# locate the gold standard 
infile = '/Users/yue/Desktop/USA/Emory Study/2023Fall/2023Fall_BMI550_1_NLP/Assignments/Assignment1/Assignment1GoldStandardSet.xlsx'
results = gold_compare(test_dict, infile)
print('Recall: ', results['Recall'])
print('Precision: ', results['Precision'])
print('F1-Score: ', results['F1-Score'])
print(results)
####################
####################
'''
Workflow 6: label the unlabelled 
'''
unlabel = '/Users/yue/Desktop/USA/Emory Study/2023Fall/2023Fall_BMI550_1_NLP/Assignments/Assignment1/UnlabeledSet.xlsx'
test = pd.read_excel(unlabel, usecols=['ID', 'TEXT'])
test = test.dropna(how='any')
test_dict = defaultdict(list)
for i in range(0,test.shape[0]):
    ID = test.iloc[i]['ID']
    text = test.iloc[i]['TEXT']
    text = re.sub(r'(\.\n| \n)', ' ', text)
    matched_symps = grab_symp(text, symps_dict)
    # print(matched_symps)
    check_neg = neg_check(matched_symps)
    # print(check_neg)
    test_dict[ID] = check_neg
    print('Labelled', ID, '.', i, 'in', test.shape[0]-1, 'subscribers.')
print(test_dict)
