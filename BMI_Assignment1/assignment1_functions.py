import Levenshtein
from fuzzywuzzy import fuzz
import string
import re
import nltk
import itertools
import os
import pandas as pd
import numpy as np
pd.options.mode.chained_assignment = None  # default='warn'
from nltk.tokenize import sent_tokenize
from collections import defaultdict



def symptoms_dict():
    '''
    Workflow 1: import classmates' annots and combine them into one, keeping three columns: 'Symptom Expressions', 'Symptom CUIs', 'Negation Flag'
    '''
    # there are issues, since there are annotated files that used '$$' rather than '$$$'. splitting will be difficult, since different columns wont be able to pair
    # I skip this step for this moment
    # I will just directly delete the rows where the number of '$$$' are not equal among: 'Symptom Expressions', 'Symptom CUIs', 'Negation Flag'
    import os
    import pandas as pd
    pd.options.mode.chained_assignment = None  # default='warn'
    # Define the directory path
    directory_path = '/Users/yue/Desktop/USA/Emory Study/2023Fall/2023Fall_BMI550_1_NLP/Assignments/Assignment1/annots'
    # Initialize an empty list to store DataFrames
    dfs = pd.DataFrame(columns=['Symptom Expressions', 'Symptom CUIs', 'Negation Flag'])
    # Initialize an empty DataFrame to store mistakes
    mistake = pd.DataFrame(columns=['Symptom Expressions', 'Symptom CUIs', 'Negation Flag', 'File Number'])
    # Loop through files s1.xlsx to s15.xlsx (excluding s5.xlsx)
    for i in range(1, 16):
        if i != 5:
            filename = f's{i}.xlsx'
            file_path = os.path.join(directory_path, filename)
            # Read the Excel file and select the desired columns
            df = pd.read_excel(file_path, usecols=['Symptom Expressions', 'Symptom CUIs', 'Negation Flag'])
            # Count the number of '$$$' occurrences in each column
            # counts = df.apply(lambda col: col.str.count(r'\$\$\$'))
            counts = df.apply(lambda col: col.str.count(r'\${3}(?!\$)'))
            # Check for rows where counts are not equal
            non_equal_rows = ~((counts['Symptom Expressions'] == counts['Symptom CUIs']) & (counts['Symptom Expressions'] == counts['Negation Flag']))
            # Filter rows where counts are not equal and add the 'File Number' column
            df_mistake = df[non_equal_rows]
            df_mistake['File Number'] = i
            # Append the filtered DataFrame to the list and the 'mistake' DataFrame
            dfs = pd.concat([dfs, df[~non_equal_rows]], ignore_index=True)
            mistake = pd.concat([mistake, df_mistake], ignore_index=True)
    # Remove duplicates from the combined DataFrame
    combined_df = dfs.drop_duplicates().reset_index(drop=True)
    # print(combined_df)
    # # Print the 'mistake' DataFrame
    # print("Mistake DataFrame:")
    # print(mistake)
    # Make the combined_df be our additional Lexicon
    # Initialize empty lists for each column
    symptom_expressions_list = []
    symptom_cuis_list = []
    negation_flag_list = []
    # Split the rows in each column by "$$$" and append to respective lists
    for index, row in combined_df.iterrows():
        symptom_expressions = row['Symptom Expressions'].split('$$$')
        symptom_cuis = row['Symptom CUIs'].split('$$$')
        negation_flags = row['Negation Flag'].split('$$$')
        # Append to the respective lists
        symptom_expressions_list.extend(symptom_expressions)
        symptom_cuis_list.extend(symptom_cuis)
        negation_flag_list.extend(negation_flags)
    # Create the lex_annot DataFrame
    annot = pd.DataFrame({
        'Symptom Expressions': symptom_expressions_list,
        'Symptom CUIs': symptom_cuis_list,
        'Negation Flag': negation_flag_list
    })
    # Remove rows where all columns are empty
    annot = annot.replace('', np.nan)
    annot = annot.dropna(how='any')
    # Create a boolean mask to identify rows containing "$" in any of the specified columns
    dollar_mask = annot[['Symptom Expressions', 'Symptom CUIs', 'Negation Flag']].apply(lambda row: row.astype(str).str.contains(r'\$').any(), axis=1)
    # Create a boolean mask to identify rows containing "S" in the 'Negation Flag' column
    s_mask = annot['Negation Flag'].str.contains('S')
    # Combine the masks using logical OR to identify rows that meet any of the conditions
    combined_mask = dollar_mask | s_mask
    # Use the combined mask to filter rows that meet any of the conditions
    annot = annot[~combined_mask]
    # annot.to_csv('/Users/yue/Desktop/USA/Emory Study/2023Fall/2023Fall_BMI550_1_NLP/Assignments/Assignment1/annot.csv', index=False)
    # annot = annot.drop(annot[(annot['Symptom Expressions'].str.strip() == '') &
    #                         (annot['Symptom CUIs'].str.strip() == '') &
    #                         (annot['Negation Flag'].str.strip() == '') &
    #                         (annot['Symptom Expressions'].str.len() < 2) &
    #                         (annot['Symptom CUIs'].str.len() < 2) &
    #                         (annot['Negation Flag'].str.len() <1) & 
    #                         (annot['Symptom Expressions'].str.strip() == ' ') &
    #                         (annot['Symptom CUIs'].str.strip() == ' ') &
    #                         (annot['Negation Flag'].str.strip() == ' ') ].index)
    # # Print the resulting lex_annot DataFrame
    annot = annot.drop_duplicates().reset_index(drop=True)
    annot['Negation Flag'] = annot['Negation Flag'].astype(int)
    #print("annot DataFrame:")
    #print(annot)
    # Create a Lexicon from Annotation: lex_annot
    # Filter rows where 'Negation Flag' is equal to 0
    filtered_annot = annot[annot['Negation Flag'] == 0]
    # Create the lex_annot dictionary using 'Symptom Expressions' as keys and 'Symptom CUIs' as values
    lex_annot = dict(zip(filtered_annot['Symptom Expressions'], filtered_annot['Symptom CUIs']))
    # count1 = []
    # for key, value in lex_annot.items():
    #     # Split the string into words using whitespace as the delimiter
    #     words = key.split()
    #     # Count the number of words
    #     word_count = len(words)
    #     # Print the result
    #     count1.append(word_count)
    # Print the lex_annot dictionary
    # print("lex_annot Dictionary:")
    # print(lex_annot)
    '''
    Workflow 2: import COVID-Twitter-Symptom-Lexicon.txt
    '''
    # Initialize an empty dictionary
    lex_std = {} # std means standard
    # Open and read the file
    with open('/Users/yue/Desktop/USA/Emory Study/2023Fall/2023Fall_BMI550_1_NLP/Assignments/Assignment1/COVID-Twitter-Symptom-Lexicon.txt', 'r') as file:
        for line in file:
            # Split the line into columns using tab ('\t') as the separator
            columns = line.strip().split('\t')        
            # Check if the line has at least 3 columns
            if len(columns) >= 3:
                # Use the 3rd column as the key and the 2nd column as the value
                lex_std[columns[2]] = columns[1]
    # Print the dictionary
    # print(lex_std)
    # count = []
    # for key, value in lex_std.items():
    #     # Split the string into words using whitespace as the delimiter
    #     words = key.split()
    #     # Count the number of words
    #     word_count = len(words)
    #     # Print the result
    #     count.append(word_count)
    # Filter the dictionary to keep only elements with fewer words in keys
    # xxx = {key: value for key, value in lex_annot.items() if len(key.split()) < (max(count) )}
    '''
    Workflow 3: combine lex_std and lex_annot and remove duplicatesl; make a customized negation Lexicon, lex_neg
    '''
    # pos means positive, we didnt do negation to this dict
    lex_pos = {key: value for d in (lex_std, lex_annot) for key, value in d.items()}
    return lex_pos
    #print(len(lex_annot)+len(lex_std)-len(lex_pos)) # there are only 102 variants removed, which means in our annotation, we included more variants, which i expect to improve the performance of the system.
    # Create the customized negation lexicon, lex_neg
    # neg_annot = annot[annot['Negation Flag'] == 1]
    # neg_annot['Combined'] = neg_annot['Symptom CUIs'].astype(str) + '-' + neg_annot['Negation Flag'].astype(str)
    # # Create the lex_annot dictionary using 'Symptom Expressions' as keys and 'Symptom CUIs' as values
    # lex_neg = dict(zip(neg_annot['Symptom Expressions'], neg_annot['Combined']))
    # # Print the lex_annot dictionary
    # print("lex_neg Dictionary:")
    # print(lex_neg)


def run_sliding_window_through_text(words, window_size):
    """
    Generate a window sliding through a sequence of words
    """
    word_iterator = iter(words) # creates an object which can be iterated one element at a time
    word_window = tuple(itertools.islice(word_iterator, window_size)) #islice() makes an iterator that returns selected elements from the the word_iterator
    yield word_window
    #now to move the window forward, one word at a time
    for w in word_iterator:
        word_window = word_window[1:] + (w,)
        yield word_window

def match_dict_similarity(text, expressions):
    '''
    :param text:
    :param expressions:
    :return:
    '''
    threshold = 0.7
    # Through some tests, I notice that it is not necessary to set the threshold too low even though the string may be very long
    # For example
    # Levenshtein.ratio('bviewbvuierueornfvoernvoerv','wevnwfnelkwnfcbewivbeuigdfgbjkmkmllkjuygfdewasdcvewklfnvkenvlkenvkelnviueruerihvre')
    # The two strings have a Lev ratio of 0.312, even though the two strings look quite irrelevant to each other
    # Levenshtein.ratio('oooooooooooooooooooooooooooooooooooooooooooooo','aaaaaaa'
    # ratio 0 occurs mostly because the two strings do not have common components at all
    '''
    This implicates that as long as two expressions have at least one common element, their Lev ratio will not be 0.
    In practice, two irrelevant strings usually have many common elements.
    Thus, we do not want to use two low threshold, even for long expressions.
    '''
    max_similarity_obtained = -1
    best_match = ''
    #go through each expression
    for exp in expressions: # we can stem exp, and text
        #create the window size equal to the number of words in the expression in the lexicon
        size_of_window = len(exp.split())
        tokenized_text = list(nltk.word_tokenize(text))
        for window in run_sliding_window_through_text(tokenized_text, size_of_window):
            # print(window, '################')
            window_string = ' '.join(window)
            similarity_score = Levenshtein.ratio(window_string, exp)
            if similarity_score >= threshold:
                #print (similarity_score,'\t', exp,'\t', window_string) # no need to print this out while doing the assignment
                if similarity_score>max_similarity_obtained:
                    max_similarity_obtained = similarity_score
                    best_match = window_string
    return [best_match,max_similarity_obtained]

def negs():
    # Negation triggers
    negations = []
    infile = open('/Users/yue/Desktop/USA/Emory Study/2023Fall/2023Fall_BMI550_1_NLP/neg_trigs.txt')
    for line in infile:
        negations.append(str.strip(line))
    return negations

# grab symptoms from texts
def grab_symp(text, symptom_dict):
    sentences = sent_tokenize(text)
    # print (sentences)
    # print ('----')
    matched_tuples = []
    #go through each sentence
    for s in sentences:
        #go through each symptom expression in the dictionary
        for symptom in symptom_dict.keys():
            #find all matches
            for match in re.finditer(r'\b'+symptom+r'\b',s):
                #Note: uncomment below to see the output
                #print (s, symptom_dict[k],match.group(), match.start(), match.end())
                match_tuple = (s,symptom_dict[symptom],match.group(),match.start(),match.end())
                matched_tuples.append(match_tuple) 
                # ('sorry if these words seem angry.', 'C0022107', 'angry', 26, 31)
    return matched_tuples



def in_scope(neg_end, text,symptom_expression):
    '''
    Function to check if a symptom occurs within the scope of a negation based on some
    pre-defined rules.
    :param neg_end: the end index of the negation expression
    :param text:
    :param symptom_expression:
    :return:
    '''
    negations = negs()
    negated = False
    text_following_negation = text[neg_end:]
    tokenized_text_following_negation = list(nltk.word_tokenize(text_following_negation))
    # this is the maximum scope of the negation, unless there is a '.' or another negation
    three_terms_following_negation = ' '.join(tokenized_text_following_negation[:min(len(tokenized_text_following_negation),3)])
    #Note: in the above we have to make sure that the text actually contains 3 words after the negation
    #that's why we are using the min function -- it will be the minimum or 3 or whatever number of terms are occurring after
    #the negation. Uncomment the print function to see these texts.
    # print (three_terms_following_negation)
    match_object = re.search(symptom_expression,three_terms_following_negation)
    if match_object:
        period_check = re.search('\.',three_terms_following_negation)
        next_negation = 1000 #starting with a very large number
        #searching for more negations that may be occurring
        for neg in negations:
            # a little simplified search..
            if re.search(neg,text_following_negation):
                index = text_following_negation.find(neg)
                if index<next_negation:
                    next_negation = index # what is next_negation for? 'I have no fever but cough', first neg is 'no fever', the next neg is 'but cough'
        if period_check:
            #if the period occurs after the symptom expression
            if period_check.start() > match_object.start() and next_negation > match_object.start():
                negated = True
        else:
            negated = True
    return negated

def neg_check(matched_tuples):
    # Define neg_check function
    symptoms = []
    negations = negs()
    #now to check if a concept is negated or not
    for mt in matched_tuples:
        is_negated = False
        #Note: I broke down the code into simpler chunks for easier understanding..
        text = mt[0]
        cui = mt[1]
        expression = mt[2]
        start = mt[3]
        end = mt[4]
        #go through each negation expression
        for neg in negations:
        #check if the negation matches anything in the text of the tuple
            for match in re.finditer(r'\b'+neg+r'\b', text):
            #if there is a negation in the sentence, we need to check
            #if the symptom expression also falls under this expression
            #it's perhaps best to pass this functionality to a function.
            # See the in_scope() function
                is_negated = in_scope(match.end(),text,expression)
                if is_negated:
                # print (text,'\t',cui+'-1')
                    symptoms.append(cui+'-1')
                    break
        if not is_negated:
            #print (text,'\t',cui+'-0')
            symptoms.append(cui+'-0')
    return symptoms

def load_labels(f_path):
    '''
    Loads the labels

    :param f_path:
    :return:
    '''
    labeled_df = pd.read_excel(f_path)
    labeled_dict = defaultdict(list)
    for index,row in labeled_df.iterrows():
        id_ = row['ID']
        if not pd.isna(row['Symptom CUIs']) and not pd.isna(row['Negation Flag']):
            cuis = row['Symptom CUIs'].split('$$$')[1:-1]
            neg_flags = row['Negation Flag'].split('$$$')[1:-1]
            for cui,neg_flag in zip(cuis,neg_flags):
                labeled_dict[id_].append(cui + '-' + str(neg_flag))
    return labeled_dict


def gold_compare(submission_dict, infile):
    gold_standard_dict = load_labels(infile)
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    for k,v in gold_standard_dict.items():
        for c in v:
            try:
                if c in submission_dict[k]:
                    tp+=1
                else:
                    fn+=1
                    # print('{}\t{}\tfn'.format(k, c))
            except KeyError:#if the key is not found in the submission file, each is considered
                            #to be a false negative..
                fn+=1
                # print('{}\t{}\tfn'.format(k, c))
        for c2 in submission_dict[k]:
            if not c2 in gold_standard_dict[k]:
                fp+=1
                # print('{}\t{}\tfp'.format(k, c2))
    print('True Positives:',tp, 'False Positives: ', fp, 'False Negatives:', fn)
    recall = tp/(tp+fn)
    precision = tp/(tp+fp)
    f1 = (2*recall*precision)/(recall+precision)
    print('Recall: ',recall,'\nPrecision:',precision,'\nF1-Score:',f1)
    #print('{}\t{}\t{}'.format(precision, recall, f1))
    return {'Recall':recall,
            'Precision': precision,
            'F1-Score':f1}

