'''
BMI 550: Applied BioNLP

Assignment 2

@author Yue Tang
Date: 10/23/2023
'''

import nltk
from collections import defaultdict
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from nltk.stem.porter import *
from nltk.corpus import stopwords
import numpy as np
import warnings
warnings.filterwarnings("ignore")

st = stopwords.words('english')
stemmer = PorterStemmer()

def loadwordclusters():
    word_clusters = {}
    infile = open('/Users/yue/Desktop/USA/Emory Study/2023Fall/2023Fall_BMI550_1_NLP/Assignments/Assignment2/clusters_ty.txt')
    for line in infile:
        items = str.strip(line).split()
        class_ = items[0]
        term = items[1]
        word_clusters[term] = class_
    return word_clusters

def getclusterfeatures(sent):
    sent = sent.lower()
    terms = nltk.word_tokenize(sent)
    cluster_string = ''
    for t in terms:
        if t in word_clusters.keys():
                cluster_string += 'clust_' + word_clusters[t] + '_clust '
    return str.strip(cluster_string)

def loadDataAsDataFrame(f_path):
    df = pd.read_csv(f_path)
    return df

def preprocess_text(raw_text):
    words = [stemmer.stem(w) for w in raw_text.lower().split()]
    return (" ".join(words))


def getClusters(texts):
    clusters = []
    for tr in texts:
        clusters.append(getclusterfeatures(tr))
    return(clusters)

def search_hyperparam_space(params, classifier, x_train, y_train, x_dev, y_dev, search_strategy = 'grid'):
    if search_strategy == 'grid':
        grid = GridSearchCV(estimator=classifier,
                            param_grid=params,
                            refit=True,
                            cv=5,
                            return_train_score=False,
                            scoring='f1_micro',
                                )
        grid.fit(x_train, y_train)
        # estimator = grid.best_estimator_
        #cvs = cross_val_score(estimator, x_train, y_train, cv=5)
        results = pd.DataFrame(grid.cv_results_)
        return results, grid


train = loadDataAsDataFrame('/Users/yue/Desktop/USA/Emory Study/2023Fall/2023Fall_BMI550_1_NLP/Assignments/Assignment2/fallreports_2023-9-21_train.csv') # 299 rows: all_data[:299]
test = loadDataAsDataFrame('/Users/yue/Desktop/USA/Emory Study/2023Fall/2023Fall_BMI550_1_NLP/Assignments/Assignment2/fallreports_2023-9-21_test.csv') # 71 rows: all_data[299:]

all_data = pd.concat([train, test], ignore_index=True) # 370 rows

# np.shape(data); np.shape(final_test)

# Remove rows with 'fog_q_class' being NaN
all_data = all_data.dropna(subset=['fog_q_class'])

# Replace any 'fall_description' being nan by empty string ''
all_data['fall_description'].fillna('', inplace=True)
all_data['fall_study_day'].fillna(99999, inplace=True)

train, test = all_data[:299], all_data[299:]


##################################################################################################
#################################Get additional features #########################################
##################################################################################################
# Get additional features (including non-text factors)print(df.columns)
non_text = ['gender', 'race', 'ethnicity', 'education', 'age_at_enrollment', 'pd_duration', # pd: Parkinson's Disease
            'num_falls_6_mo', 'previous_falls', 'mds_updrs_iii_total_video', 'mds_updrs_iii_binary', 'mds_updrs_iii_hy_video', 'abc_total', 'moca_total',
            'minibestest_total', 'fall_study_day', 'location_binary', 'fall_class', 'fog_yn', 'fall_total', 'fall_rate']
# print(all_data[non_text].info())
#  #   Column                     Non-Null Count  Dtype  
# ---  ------                     --------------  -----  
#  0   gender                     299 non-null    object *
#  1   race                       299 non-null    object *
#  2   ethnicity                  299 non-null    object *
#  3   education                  299 non-null    object *
#  4   age_at_enrollment          299 non-null    int64  
#  5   pd_duration                299 non-null    float64
#  6   num_falls_6_mo             279 non-null    object *
#  7   previous_falls             299 non-null    object *
#  8   mds_updrs_iii_total_video  299 non-null    int64  
#  9   mds_updrs_iii_binary       299 non-null    object *
#  10  mds_updrs_iii_hy_video     218 non-null    object *
#  11  abc_total                  299 non-null    float64
#  12  moca_total                 299 non-null    int64  
#  13  minibestest_total          299 non-null    int64  
#  14  fall_study_day             292 non-null    float64
#  15  location_binary            295 non-null    object *
#  16  fall_class                 297 non-null    object *
#  17  fog_yn                     295 non-null    object *
#  18  fall_total                 299 non-null    int64  
#  19  fall_rate                  299 non-null    float64

# Convert string factors to ono-hot
non_text_cat = ['gender', 'race', 'ethnicity', 'education', 'num_falls_6_mo', 'previous_falls', 'mds_updrs_iii_binary', 'mds_updrs_iii_hy_video',
            'location_binary', 'fall_class', 'fog_yn']
non_text_num = list(set(non_text).difference(set(non_text_cat)))
# Get categorical one-hot
non_text_one_hot = pd.get_dummies(all_data[non_text_cat], drop_first=True)
# Print the resulting DataFrame
# print(non_text_one_hot)

# Combine non-text categorical with non-text number
non_text_all = pd.concat([non_text_one_hot, all_data[non_text_num]], axis=1)
non_text_all = non_text_all.fillna(-99999)

non_text_TRAIN = non_text_all[:299]
non_text_TEST = non_text_all[299:]
# print(non_text_all)


##################################################################################################
################################# Get texts variable #############################################
##################################################################################################
# Extract 'fall_description' (text column) and 'fall_q_class' (label)
# Note that we now only look at texts from training set
texts = train['fall_description']; classes = train['fog_q_class']
test_texts = test['fall_description']; test_classes = test['fog_q_class'] 

# Initialize word_clusters
word_clusters = loadwordclusters()

# Get n-grams (n = 1,2,3) feature
texts_preprocessed = []
TEST_texts_preprocessed = []
#PROGRAMMING TIP: c++ style coding here can help when doing feature engineering.. see below
for tr in texts:
    texts_preprocessed.append(preprocess_text(tr))
for tst in test_texts:
    TEST_texts_preprocessed.append(preprocess_text(tst))
#VECTORIZE
mx_ftr_cls = 50 # max features
mx_ftr_gram = 3000 # max features
clustervectorizer = CountVectorizer(ngram_range=(1,3), max_features=mx_ftr_cls)
# underscore text # this is for cluster generation
texts_preprocessed_ = [' '.join(['_'.join([word, next_word]) for word, next_word in zip(sentence.split(), sentence.split()[1:])]) for sentence in texts_preprocessed]
TEST_texts_preprocessed_ = [' '.join(['_'.join([word, next_word]) for word, next_word in zip(sentence.split(), sentence.split()[1:])]) for sentence in TEST_texts_preprocessed]



from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, f1_score, classification_report





##################################################################################################
################################### I. Tune NB Model ############################################# Naive Bayes
##################################################################################################
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
# def formatData(non_text_DATA, texts_preprocessed, texts_preprocessed_, train_index, vectorizer):
#     texts_preprocessed_train = [texts_preprocessed[i] for i in train_index]
#     texts_preprocessed_train_ = [texts_preprocessed_[i] for i in train_index]
#     # Feature 1: n-grams
#     # VECTORIZE
#     # vectorizer = CountVectorizer(ngram_range=(1, 3), analyzer="word", tokenizer=None, stop_words='english', preprocessor=None,
#     #                                 max_features=mx_ftr)
#     training_gram = vectorizer.transform(texts_preprocessed_train).toarray()
#     # Feature 2: clusters
#     clusters_train = []; clusters_train_ = [];
#     #PROGRAMMING TIP: c++ style coding here can help when doing feature engineering.. see below
#     clusters_train = getClusters(texts_preprocessed_train); clusters_train_ = getClusters(texts_preprocessed_train_)
#     #VECTORIZE
#     cluster_train = clustervectorizer.fit_transform(clusters_train).toarray(); cluster_train_ = clustervectorizer.fit_transform(clusters_train_).toarray()
#     # Combine all features together
#     training_data = np.concatenate((training_gram, cluster_train, cluster_train_, non_text_DATA.iloc[train_index]), axis=1)
#     return(training_data)


skf = StratifiedKFold(n_splits=10)
skf.get_n_splits(texts_preprocessed,classes)

naive_bayes = MultinomialNB()
for train_index, test_index in skf.split(texts_preprocessed,classes):
    texts_preprocessed_train = [texts_preprocessed[i] for i in train_index]
    texts_preprocessed_train_ = [texts_preprocessed_[i] for i in train_index]
    texts_preprocessed_dev = [texts_preprocessed[i] for i in test_index]
    texts_preprocessed_dev_ = [texts_preprocessed_[i] for i in test_index]
    # Feature 1: n-grams
    # VECTORIZE
    vectorizer = CountVectorizer(ngram_range=(1, 3), analyzer="word", tokenizer=None, stop_words='english', preprocessor=None,
                                    max_features=mx_ftr_gram)
    vectorizer.fit(texts_preprocessed_train)
    training_gram = vectorizer.transform(texts_preprocessed_train).toarray()
    test_gram = vectorizer.transform(texts_preprocessed_dev).toarray()
    # Feature 2: clusters
    clusters_train = []; clusters_train_ = [];
    clusters_dev = []; clusters_dev_ = [];
    #PROGRAMMING TIP: c++ style coding here can help when doing feature engineering.. see below
    clusters_train = getClusters(texts_preprocessed_train); clusters_dev = getClusters(texts_preprocessed_dev); 
    clusters_train_ = getClusters(texts_preprocessed_train_); clusters_dev_ = getClusters(texts_preprocessed_dev_)
    #VECTORIZE
    cluster_train = clustervectorizer.fit_transform(clusters_train).toarray(); cluster_dev = clustervectorizer.transform(clusters_dev).toarray();
    cluster_train_ = clustervectorizer.fit_transform(clusters_train_).toarray(); cluster_dev_ = clustervectorizer.transform(clusters_dev_).toarray()
    # Combine all features together
    training_data = np.concatenate((training_gram, cluster_train, cluster_train_, non_text_TRAIN.iloc[train_index]), axis=1)
    test_data = np.concatenate((test_gram,cluster_dev, cluster_dev_, non_text_TRAIN.iloc[test_index]),axis=1)
    ttp_train, ttp_test = classes[train_index], classes[test_index]
    # training_data = formatData(non_text_TRAIN, texts_preprocessed, texts_preprocessed_, train_index, vectorizer)
    # test_data = formatData(non_text_TRAIN, texts_preprocessed, texts_preprocessed_, test_index, vectorizer)

    grid_params = {
        'alpha': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],  # Values of alpha to test
    }
    results, grid = search_hyperparam_space(grid_params, naive_bayes, training_data,ttp_train,test_data,ttp_test)
    print("Best parameters set found on development set:")
    print(grid.best_params_)
    print("Grid scores on development set:")
    means = grid.cv_results_['mean_test_score']
    stds = grid.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, grid.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
                % (mean, std * 2, params))
    # TRAIN THE MODEL
    nb_classifier = grid.best_estimator_.fit(training_data, ttp_train)
    predictions = nb_classifier.predict(test_data)    
    accuracy, f1_micro, f1_macro = accuracy_score(ttp_test, predictions), f1_score(ttp_test, predictions, average='micro'), f1_score(ttp_test, predictions, average='macro')
    # Print the results
    print(f'Accuracy: {accuracy:.2f}')
    print(f'Micro-averaged F1 Score: {f1_micro:.2f}')
    print(f'Macro-averaged F1 Score: {f1_macro:.2f}')

####################################
###### Best parameter is 0.1 #######
####################################

# Process test data
def prepare_data():
    # Feature 1: n-grams
    # VECTORIZE
    vectorizer = CountVectorizer(ngram_range=(1, 3), analyzer="word", tokenizer=None, stop_words='english', preprocessor=None,
                                    max_features=mx_ftr_gram)
    vectorizer.fit(texts_preprocessed)
    training_gram = vectorizer.transform(texts_preprocessed).toarray()
    test_gram = vectorizer.transform(TEST_texts_preprocessed).toarray()
    # Feature 2: clusters
    clusters_train = []; clusters_train_ = [];
    clusters_dev = []; clusters_dev_ = [];
    #PROGRAMMING TIP: c++ style coding here can help when doing feature engineering.. see below
    clusters_train = getClusters(texts_preprocessed); clusters_dev = getClusters(TEST_texts_preprocessed); 
    clusters_train_ = getClusters(texts_preprocessed_); clusters_dev_ = getClusters(TEST_texts_preprocessed_)
    #VECTORIZE
    cluster_train = clustervectorizer.fit_transform(clusters_train).toarray(); cluster_dev = clustervectorizer.transform(clusters_dev).toarray();
    cluster_train_ = clustervectorizer.fit_transform(clusters_train_).toarray(); cluster_dev_ = clustervectorizer.transform(clusters_dev_).toarray()
    # Combine all features together
    train_X = np.concatenate((training_gram, cluster_train, cluster_train_, non_text_TRAIN), axis=1)
    test_X = np.concatenate((test_gram,cluster_dev, cluster_dev_, non_text_TEST), axis=1)
    return train_X, test_X

train_X, test_X = prepare_data()

nb_classifier = MultinomialNB(alpha=0.1)
nb_classifier = nb_classifier.fit(train_X, classes)
predictions = nb_classifier.predict(test_X)    
accuracy, f1_micro, f1_macro = accuracy_score(test_classes, predictions), f1_score(test_classes, predictions, average='micro'), f1_score(test_classes, predictions, average='macro')
# Print the results
print(f'Accuracy: {accuracy:.2f}')
print(f'Micro-averaged F1 Score: {f1_micro:.2f}')
print(f'Macro-averaged F1 Score: {f1_macro:.2f}')
# >>> print(f'Accuracy: {accuracy:.2f}')
# Accuracy: 0.56
# >>> print(f'Micro-averaged F1 Score: {f1_micro:.2f}')
# Micro-averaged F1 Score: 0.56
# >>> print(f'Macro-averaged F1 Score: {f1_macro:.2f}')
# Macro-averaged F1 Score: 0.45
# >>> 



##################################################################################################
################################### II. Tune RF Model ############################################ Random Forest
##################################################################################################
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
# skf = StratifiedKFold(n_splits=10)
# skf.get_n_splits(texts_preprocessed,classes)

rf_classifier = RandomForestClassifier()
for train_index, test_index in skf.split(texts_preprocessed,classes):
    texts_preprocessed_train = [texts_preprocessed[i] for i in train_index]
    texts_preprocessed_train_ = [texts_preprocessed_[i] for i in train_index]
    texts_preprocessed_dev = [texts_preprocessed[i] for i in test_index]
    texts_preprocessed_dev_ = [texts_preprocessed_[i] for i in test_index]
    # Feature 1: n-grams
    # VECTORIZE
    vectorizer = CountVectorizer(ngram_range=(1, 3), analyzer="word", tokenizer=None, stop_words='english', preprocessor=None,
                                    max_features=mx_ftr_gram)
    vectorizer.fit(texts_preprocessed_train)
    training_gram = vectorizer.transform(texts_preprocessed_train).toarray()
    test_gram = vectorizer.transform(texts_preprocessed_dev).toarray()
    # Feature 2: clusters
    clusters_train = []; clusters_train_ = [];
    clusters_dev = []; clusters_dev_ = [];
    #PROGRAMMING TIP: c++ style coding here can help when doing feature engineering.. see below
    clusters_train = getClusters(texts_preprocessed_train); clusters_dev = getClusters(texts_preprocessed_dev); 
    clusters_train_ = getClusters(texts_preprocessed_train_); clusters_dev_ = getClusters(texts_preprocessed_dev_)
    #VECTORIZE
    cluster_train = clustervectorizer.fit_transform(clusters_train).toarray(); cluster_dev = clustervectorizer.transform(clusters_dev).toarray();
    cluster_train_ = clustervectorizer.fit_transform(clusters_train_).toarray(); cluster_dev_ = clustervectorizer.transform(clusters_dev_).toarray()
    # Combine all features together
    training_data = np.concatenate((training_gram, cluster_train, cluster_train_, non_text_TRAIN.iloc[train_index]), axis=1)
    test_data = np.concatenate((test_gram,cluster_dev, cluster_dev_, non_text_TRAIN.iloc[test_index]),axis=1)
    ttp_train, ttp_test = classes[train_index], classes[test_index]

    grid_params = {
        'n_estimators': [50, 100, 200],  # Number of trees in the forest
        'max_depth': [None],  # Maximum depth of the trees
        'min_samples_split': [2, 3, 4],  # Minimum samples required to split a node
        'min_samples_leaf': [2, 3, 4]  # Minimum samples required at each leaf node
    }
    results, grid = search_hyperparam_space(grid_params, rf_classifier, training_data,ttp_train,test_data,ttp_test)
    print("Best parameters set found on development set:")
    print(grid.best_params_)
    print("Grid scores on development set:")
    means = grid.cv_results_['mean_test_score']
    stds = grid.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, grid.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
                % (mean, std * 2, params))
    # TRAIN THE MODEL
    rf_classifier = grid.best_estimator_.fit(training_data, ttp_train)
    predictions = rf_classifier.predict(test_data)    
    accuracy, f1_micro, f1_macro = accuracy_score(ttp_test, predictions), f1_score(ttp_test, predictions, average='micro'), f1_score(ttp_test, predictions, average='macro')
    # Print the results
    print(f'Accuracy: {accuracy:.2f}')
    print(f'Micro-averaged F1 Score: {f1_micro:.2f}')
    print(f'Macro-averaged F1 Score: {f1_macro:.2f}')


#########################################################################################################################
###### Best parameter is {'max_depth': None, 'min_samples_leaf': 2, 'min_samples_split': 4, 'n_estimators': 100} ########
#########################################################################################################################

rf_classifier = RandomForestClassifier(max_depth = None, min_samples_leaf = 2, min_samples_split = 4, n_estimators = 100)
rf_classifier = rf_classifier.fit(train_X, classes)
predictions = rf_classifier.predict(test_X) 
# print (f1_score(test_classes, predictions, average='micro'))
accuracy, f1_micro, f1_macro = accuracy_score(test_classes, predictions), f1_score(test_classes, predictions, average='micro'), f1_score(test_classes, predictions, average='macro')
# Print the results
print(f'Accuracy: {accuracy:.2f}')
print(f'Micro-averaged F1 Score: {f1_micro:.2f}')
print(f'Macro-averaged F1 Score: {f1_macro:.2f}')
# >>> print(f'Accuracy: {accuracy:.2f}')
# Accuracy: 0.94
# >>> print(f'Micro-averaged F1 Score: {f1_micro:.2f}')
# Micro-averaged F1 Score: 0.94
# >>> print(f'Macro-averaged F1 Score: {f1_macro:.2f}')
# Macro-averaged F1 Score: 0.94



###################################################################################################
################################### III. Tune KNN Model ########################################### K-Nearest Neighbors
###################################################################################################
from sklearn.neighbors import KNeighborsClassifier
# skf = StratifiedKFold(n_splits=10)
# skf.get_n_splits(texts_preprocessed,classes)

knn_classifier = KNeighborsClassifier()
for train_index, test_index in skf.split(texts_preprocessed,classes):
    texts_preprocessed_train = [texts_preprocessed[i] for i in train_index]
    texts_preprocessed_train_ = [texts_preprocessed_[i] for i in train_index]
    texts_preprocessed_dev = [texts_preprocessed[i] for i in test_index]
    texts_preprocessed_dev_ = [texts_preprocessed_[i] for i in test_index]
    # Feature 1: n-grams
    # VECTORIZE
    vectorizer = CountVectorizer(ngram_range=(1, 3), analyzer="word", tokenizer=None, stop_words='english', preprocessor=None,
                                    max_features=mx_ftr_gram)
    vectorizer.fit(texts_preprocessed_train)
    training_gram = vectorizer.transform(texts_preprocessed_train).toarray()
    test_gram = vectorizer.transform(texts_preprocessed_dev).toarray()
    # Feature 2: clusters
    clusters_train = []; clusters_train_ = [];
    clusters_dev = []; clusters_dev_ = [];
    #PROGRAMMING TIP: c++ style coding here can help when doing feature engineering.. see below
    clusters_train = getClusters(texts_preprocessed_train); clusters_dev = getClusters(texts_preprocessed_dev); 
    clusters_train_ = getClusters(texts_preprocessed_train_); clusters_dev_ = getClusters(texts_preprocessed_dev_)
    #VECTORIZE
    cluster_train = clustervectorizer.fit_transform(clusters_train).toarray(); cluster_dev = clustervectorizer.transform(clusters_dev).toarray();
    cluster_train_ = clustervectorizer.fit_transform(clusters_train_).toarray(); cluster_dev_ = clustervectorizer.transform(clusters_dev_).toarray()
    # Combine all features together
    training_data = np.concatenate((training_gram, cluster_train, cluster_train_, non_text_TRAIN.iloc[train_index]), axis=1)
    test_data = np.concatenate((test_gram,cluster_dev, cluster_dev_, non_text_TRAIN.iloc[test_index]),axis=1)
    ttp_train, ttp_test = classes[train_index], classes[test_index]

    grid_params = {
        'n_neighbors': [3, 5, 7, 9],  # Number of neighbors to consider
        'weights': ['uniform', 'distance'],  # Weighting of neighbors
        'p': [1, 2]  # Power parameter for the Minkowski distance metric (1 for Manhattan, 2 for Euclidean)
    }
    results, grid = search_hyperparam_space(grid_params, knn_classifier, training_data,ttp_train,test_data,ttp_test)
    print("Best parameters set found on development set:")
    print(grid.best_params_)
    print("Grid scores on development set:")
    means = grid.cv_results_['mean_test_score']
    stds = grid.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, grid.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
                % (mean, std * 2, params))
    # TRAIN THE MODEL
    knn_classifier = grid.best_estimator_.fit(training_data, ttp_train)
    predictions = knn_classifier.predict(test_data)    
    accuracy, f1_micro, f1_macro = accuracy_score(ttp_test, predictions), f1_score(ttp_test, predictions, average='micro'), f1_score(ttp_test, predictions, average='macro')
    # Print the results
    print(f'Accuracy: {accuracy:.2f}')
    print(f'Micro-averaged F1 Score: {f1_micro:.2f}')
    print(f'Macro-averaged F1 Score: {f1_macro:.2f}')


###################################################################################
###### Best parameter is {'n_neighbors': 3, 'p': 1, 'weights': 'distance'} ########
###################################################################################

knn_classifier = KNeighborsClassifier(n_neighbors = 3, p = 1, weights = 'distance')
knn_classifier = knn_classifier.fit(train_X, classes)
predictions = knn_classifier.predict(test_X)    
accuracy, f1_micro, f1_macro = accuracy_score(test_classes, predictions), f1_score(test_classes, predictions, average='micro'), f1_score(test_classes, predictions, average='macro')
# Print the results
print(f'Accuracy: {accuracy:.2f}')
print(f'Micro-averaged F1 Score: {f1_micro:.2f}')
print(f'Macro-averaged F1 Score: {f1_macro:.2f}')
# >>> print(f'Accuracy: {accuracy:.2f}')
# Accuracy: 0.93
# >>> print(f'Micro-averaged F1 Score: {f1_micro:.2f}')
# Micro-averaged F1 Score: 0.93
# >>> print(f'Macro-averaged F1 Score: {f1_macro:.2f}')
# Macro-averaged F1 Score: 0.93


###################################################################################################
################################### IV. Tune LR Model ############################################# Logistic Regression
###################################################################################################
from sklearn.linear_model import LogisticRegression
lr_classifier = LogisticRegression()
for train_index, test_index in skf.split(texts_preprocessed,classes):
    texts_preprocessed_train = [texts_preprocessed[i] for i in train_index]
    texts_preprocessed_train_ = [texts_preprocessed_[i] for i in train_index]
    texts_preprocessed_dev = [texts_preprocessed[i] for i in test_index]
    texts_preprocessed_dev_ = [texts_preprocessed_[i] for i in test_index]
    # Feature 1: n-grams
    # VECTORIZE
    vectorizer = CountVectorizer(ngram_range=(1, 3), analyzer="word", tokenizer=None, stop_words='english', preprocessor=None,
                                    max_features=mx_ftr_gram)
    vectorizer.fit(texts_preprocessed_train)
    training_gram = vectorizer.transform(texts_preprocessed_train).toarray()
    test_gram = vectorizer.transform(texts_preprocessed_dev).toarray()
    # Feature 2: clusters
    clusters_train = []; clusters_train_ = [];
    clusters_dev = []; clusters_dev_ = [];
    #PROGRAMMING TIP: c++ style coding here can help when doing feature engineering.. see below
    clusters_train = getClusters(texts_preprocessed_train); clusters_dev = getClusters(texts_preprocessed_dev); 
    clusters_train_ = getClusters(texts_preprocessed_train_); clusters_dev_ = getClusters(texts_preprocessed_dev_)
    #VECTORIZE
    cluster_train = clustervectorizer.fit_transform(clusters_train).toarray(); cluster_dev = clustervectorizer.transform(clusters_dev).toarray();
    cluster_train_ = clustervectorizer.fit_transform(clusters_train_).toarray(); cluster_dev_ = clustervectorizer.transform(clusters_dev_).toarray()
    # Combine all features together
    training_data = np.concatenate((training_gram, cluster_train, cluster_train_, non_text_TRAIN.iloc[train_index]), axis=1)
    test_data = np.concatenate((test_gram,cluster_dev, cluster_dev_, non_text_TRAIN.iloc[test_index]),axis=1)
    ttp_train, ttp_test = classes[train_index], classes[test_index]

    grid_params = {
        'penalty': ['l1', 'l2'],  # Regularization type (L1 or L2)
        'C': [0.001, 0.01, 0.1, 1.0, 10.0],  # Inverse of regularization strength (smaller values for stronger regularization)
        'solver': ['liblinear']  # Solver algorithm
    }
    results, grid = search_hyperparam_space(grid_params, lr_classifier, training_data,ttp_train,test_data,ttp_test)
    print("Best parameters set found on development set:")
    print(grid.best_params_)
    print("Grid scores on development set:")
    means = grid.cv_results_['mean_test_score']
    stds = grid.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, grid.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
                % (mean, std * 2, params))
    # TRAIN THE MODEL
    lr_classifier = grid.best_estimator_.fit(training_data, ttp_train)
    predictions = lr_classifier.predict(test_data)    
    accuracy, f1_micro, f1_macro = accuracy_score(ttp_test, predictions), f1_score(ttp_test, predictions, average='micro'), f1_score(ttp_test, predictions, average='macro')
    # Print the results
    print(f'Accuracy: {accuracy:.2f}')
    print(f'Micro-averaged F1 Score: {f1_micro:.2f}')
    print(f'Macro-averaged F1 Score: {f1_macro:.2f}')


###################################################################################
###### Best parameter is {'C': 1.0, 'penalty': 'l1', 'solver': 'liblinear'} #######
###################################################################################

lr_classifier = LogisticRegression(C = 1.0, penalty = 'l1', solver = 'liblinear')
lr_classifier = lr_classifier.fit(train_X, classes)
predictions = lr_classifier.predict(test_X)    
accuracy, f1_micro, f1_macro = accuracy_score(test_classes, predictions), f1_score(test_classes, predictions, average='micro'), f1_score(test_classes, predictions, average='macro')
# Print the results
print(f'Accuracy: {accuracy:.2f}')
print(f'Micro-averaged F1 Score: {f1_micro:.2f}')
print(f'Macro-averaged F1 Score: {f1_macro:.2f}')
# >>> print(f'Accuracy: {accuracy:.2f}')
# Accuracy: 0.92
# >>> print(f'Micro-averaged F1 Score: {f1_micro:.2f}')
# Micro-averaged F1 Score: 0.92
# >>> print(f'Macro-averaged F1 Score: {f1_macro:.2f}')
# Macro-averaged F1 Score: 0.92




###################################################################################################
################################### V. Tune XGB Model ############################################# XGBoost
###################################################################################################
import xgboost as xgb
xgb_classifier = xgb.XGBClassifier(subsample = 0.8, objective = "reg:logistic", 
                                   max_depth = 6, booster = 'gbtree', 
                                   eval_metric = 'auc')
for train_index, test_index in skf.split(texts_preprocessed,classes):
    texts_preprocessed_train = [texts_preprocessed[i] for i in train_index]
    texts_preprocessed_train_ = [texts_preprocessed_[i] for i in train_index]
    texts_preprocessed_dev = [texts_preprocessed[i] for i in test_index]
    texts_preprocessed_dev_ = [texts_preprocessed_[i] for i in test_index]
    # Feature 1: n-grams
    # VECTORIZE
    vectorizer = CountVectorizer(ngram_range=(1, 3), analyzer="word", tokenizer=None, stop_words='english', preprocessor=None,
                                    max_features=mx_ftr_gram)
    vectorizer.fit(texts_preprocessed_train)
    training_gram = vectorizer.transform(texts_preprocessed_train).toarray()
    test_gram = vectorizer.transform(texts_preprocessed_dev).toarray()
    # Feature 2: clusters
    clusters_train = []; clusters_train_ = [];
    clusters_dev = []; clusters_dev_ = [];
    #PROGRAMMING TIP: c++ style coding here can help when doing feature engineering.. see below
    clusters_train = getClusters(texts_preprocessed_train); clusters_dev = getClusters(texts_preprocessed_dev); 
    clusters_train_ = getClusters(texts_preprocessed_train_); clusters_dev_ = getClusters(texts_preprocessed_dev_)
    #VECTORIZE
    cluster_train = clustervectorizer.fit_transform(clusters_train).toarray(); cluster_dev = clustervectorizer.transform(clusters_dev).toarray();
    cluster_train_ = clustervectorizer.fit_transform(clusters_train_).toarray(); cluster_dev_ = clustervectorizer.transform(clusters_dev_).toarray()
    # Combine all features together
    training_data = np.concatenate((training_gram, cluster_train, cluster_train_, non_text_TRAIN.iloc[train_index]), axis=1)
    test_data = np.concatenate((test_gram,cluster_dev, cluster_dev_, non_text_TRAIN.iloc[test_index]),axis=1)
    ttp_train, ttp_test = classes[train_index], classes[test_index]

    grid_params = {
        'learning_rate': [0.1, 0.2],  # Learning rate
        'n_estimators': [50, 100, 200],  # Number of boosting rounds
        'reg_lambda': [0.01, 0.1],
        'alpha': [0.01, 0.1]
        }

    results, grid = search_hyperparam_space(grid_params, xgb_classifier, training_data,ttp_train,test_data,ttp_test)
    print("Best parameters set found on development set:")
    print(grid.best_params_)
    print("Grid scores on development set:")
    means = grid.cv_results_['mean_test_score']
    stds = grid.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, grid.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
                % (mean, std * 2, params))
    # TRAIN THE MODEL
    xgb_classifier = grid.best_estimator_.fit(training_data, ttp_train)
    predictions = xgb_classifier.predict(test_data)    
    accuracy, f1_micro, f1_macro = accuracy_score(ttp_test, predictions), f1_score(ttp_test, predictions, average='micro'), f1_score(ttp_test, predictions, average='macro')
    # Print the results
    print(f'Accuracy: {accuracy:.2f}')
    print(f'Micro-averaged F1 Score: {f1_micro:.2f}')
    print(f'Macro-averaged F1 Score: {f1_macro:.2f}')



###########################################################################################################
###### Best parameter is {'alpha': 0.01, 'lambda': 0.01, 'learning_rate': 0.1, 'n_estimators': 100} #######
###########################################################################################################

xgb_classifier = xgb.XGBClassifier(subsample = 0.8, objective = "reg:logistic", 
                                   max_depth = 6, booster = 'gbtree', 
                                   eval_metric = 'auc',
                                   alpha = 0.01, reg_lambda = 0.01, learning_rate = 0.1, n_estimators = 100)
xgb_classifier = xgb_classifier.fit(train_X, classes)
predictions = xgb_classifier.predict(test_X)    
accuracy, f1_micro, f1_macro = accuracy_score(test_classes, predictions), f1_score(test_classes, predictions, average='micro'), f1_score(test_classes, predictions, average='macro')
# Print the results
print(f'Accuracy: {accuracy:.2f}')
print(f'Micro-averaged F1 Score: {f1_micro:.2f}')
print(f'Macro-averaged F1 Score: {f1_macro:.2f}')
# >>> print(f'Accuracy: {accuracy:.2f}')
# Accuracy: 0.97
# >>> print(f'Micro-averaged F1 Score: {f1_micro:.2f}')
# Micro-averaged F1 Score: 0.97
# >>> print(f'Macro-averaged F1 Score: {f1_macro:.2f}')
# Macro-averaged F1 Score: 0.97



###################################################################################################
################################### VI. Tune SVM Model ############################################ Support Vector Machine
###################################################################################################
from sklearn.svm import SVC
svm_classifier = SVC(kernel = 'linear')
for train_index, test_index in skf.split(texts_preprocessed,classes):
    texts_preprocessed_train = [texts_preprocessed[i] for i in train_index]
    texts_preprocessed_train_ = [texts_preprocessed_[i] for i in train_index]
    texts_preprocessed_dev = [texts_preprocessed[i] for i in test_index]
    texts_preprocessed_dev_ = [texts_preprocessed_[i] for i in test_index]
    # Feature 1: n-grams
    # VECTORIZE
    vectorizer = CountVectorizer(ngram_range=(1, 3), analyzer="word", tokenizer=None, stop_words='english', preprocessor=None,
                                    max_features=mx_ftr_gram)
    vectorizer.fit(texts_preprocessed_train)
    training_gram = vectorizer.transform(texts_preprocessed_train).toarray()
    test_gram = vectorizer.transform(texts_preprocessed_dev).toarray()
    # Feature 2: clusters
    clusters_train = []; clusters_train_ = [];
    clusters_dev = []; clusters_dev_ = [];
    #PROGRAMMING TIP: c++ style coding here can help when doing feature engineering.. see below
    clusters_train = getClusters(texts_preprocessed_train); clusters_dev = getClusters(texts_preprocessed_dev); 
    clusters_train_ = getClusters(texts_preprocessed_train_); clusters_dev_ = getClusters(texts_preprocessed_dev_)
    #VECTORIZE
    cluster_train = clustervectorizer.fit_transform(clusters_train).toarray(); cluster_dev = clustervectorizer.transform(clusters_dev).toarray();
    cluster_train_ = clustervectorizer.fit_transform(clusters_train_).toarray(); cluster_dev_ = clustervectorizer.transform(clusters_dev_).toarray()
    # Combine all features together
    training_data = np.concatenate((training_gram, cluster_train, cluster_train_, non_text_TRAIN.iloc[train_index]), axis=1)
    test_data = np.concatenate((test_gram,cluster_dev, cluster_dev_, non_text_TRAIN.iloc[test_index]),axis=1)
    ttp_train, ttp_test = classes[train_index], classes[test_index]

    grid_params = {
        'C': [0.1, 1, 10],  # Regularization parameter
        'gamma': [0.1, 1, 10],  # Kernel coefficient
        }

    results, grid = search_hyperparam_space(grid_params, svm_classifier, training_data,ttp_train,test_data,ttp_test)
    print("Best parameters set found on development set:")
    print(grid.best_params_)
    print("Grid scores on development set:")
    means = grid.cv_results_['mean_test_score']
    stds = grid.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, grid.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
                % (mean, std * 2, params))
    # TRAIN THE MODEL
    svm_classifier = grid.best_estimator_.fit(training_data, ttp_train)
    predictions = svm_classifier.predict(test_data)    
    accuracy, f1_micro, f1_macro = accuracy_score(ttp_test, predictions), f1_score(ttp_test, predictions, average='micro'), f1_score(ttp_test, predictions, average='macro')
    # Print the results
    print(f'Accuracy: {accuracy:.2f}')
    print(f'Micro-averaged F1 Score: {f1_micro:.2f}')
    print(f'Macro-averaged F1 Score: {f1_macro:.2f}')



#########################################################
###### Best parameter is {'C': 0.1, 'gamma': 0.1} #######
#########################################################

svm_classifier = SVC(kernel = 'linear', C = 0.1, gamma = 0.1)
svm_classifier = svm_classifier.fit(train_X, classes)
predictions = svm_classifier.predict(test_X)    
accuracy, f1_micro, f1_macro = accuracy_score(test_classes, predictions), f1_score(test_classes, predictions, average='micro'), f1_score(test_classes, predictions, average='macro')
# Print the results
print(f'Accuracy: {accuracy:.2f}')
print(f'Micro-averaged F1 Score: {f1_micro:.2f}')
print(f'Macro-averaged F1 Score: {f1_macro:.2f}')
# >>> print(f'Accuracy: {accuracy:.2f}')
# Accuracy: 0.92
# >>> print(f'Micro-averaged F1 Score: {f1_micro:.2f}')
# Micro-averaged F1 Score: 0.92
# >>> print(f'Macro-averaged F1 Score: {f1_macro:.2f}')
# Macro-averaged F1 Score: 0.92









###################################################################################################
######################### As we show above, XGBoost model performs the best #######################
###################################################################################################

###################################################################################################
############################### Training set size vs. performance graph ###########################
###################################################################################################

from collections import defaultdict
import random
pfmc = defaultdict(list)
pfmc2 = defaultdict(list)
total_rows = 299
all_indices = list(range(total_rows))

for i in range(0,29,1):
    k = 19 + i*10
    xgb_classifier = xgb.XGBClassifier(subsample = 0.8, objective = "reg:logistic", 
                                    max_depth = 6, booster = 'gbtree', 
                                    eval_metric = 'auc',
                                    alpha = 0.01, reg_lambda = 0.01, learning_rate = 0.1, n_estimators = 100)
    idx = random.sample(all_indices, k)
    xgb_classifier = xgb_classifier.fit(train_X[idx], classes[idx])
    predictions = xgb_classifier.predict(test_X)    
    xgb_classifier2 = xgb_classifier.fit(train_X[:k], classes[:k])
    predictions2 = xgb_classifier.predict(test_X)    
    pfmc[str(k)] = f1_score(test_classes, predictions, average='micro')
    pfmc2[str(k)] = f1_score(test_classes, predictions2, average='micro')

# print(pfmc)

import matplotlib.pyplot as plt
x = [int(key) for key in pfmc.keys()]
y = list(pfmc.values())
x2 = [int(key) for key in pfmc2.keys()]
y2 = list(pfmc2.values())
# Create a line plot
plt.plot(x, y, marker='o', linestyle='-', color = 'green', label = 'Random sample')
plt.plot(x2, y2, marker='o', linestyle='-', color = 'red', label = 'Sample starting from 1st row')
# Add labels and title
plt.xlabel('Sample size of training set')
plt.ylabel('Performance (micro-averaged F1 score on test data)')
plt.title('Classifier Performance Varies with Size of Training Set')
plt.legend()
# Show the plot
plt.show()

###################################################################################################
##### From the figure, we see that generally learning improves with training data size ############
##### We need to annotate at least 150 to get a micro F1 score of 0.85; at least 50   ############
##### to obtain a micro F1 score of 0.80 ##########################################################
###################################################################################################





###################################################################################################
######################################### VII. Ablation study #####################################
###################################################################################################

# For simplicity, here we only plot 'micro-averaged F1 score' on test data

## define colnames
non_text0 = ['gender', 'race', 'ethnicity', 'education', 'age_at_enrollment', 'pd_duration', # pd: Parkinson's Disease
            'num_falls_6_mo', 'previous_falls', 'mds_updrs_iii_total_video', 'mds_updrs_iii_binary', 'mds_updrs_iii_hy_video', 'abc_total', 'moca_total',
            'minibestest_total', 'fall_study_day', 'location_binary', 'fall_class', 'fog_yn', 'fall_total', 'fall_rate']
# Convert string factors to ono-hot
non_text_cat0 = ['gender', 'race', 'ethnicity', 'education', 'num_falls_6_mo', 'previous_falls', 'mds_updrs_iii_binary', 'mds_updrs_iii_hy_video',
            'location_binary', 'fall_class', 'fog_yn']
non_text_num0 = list(set(non_text0).difference(set(non_text_cat0)))
text_var0 = ['cluster', 'ngram']

## define a function 
def prepare_data_ablt(non_text_cat0, non_text_num0, text_var0):
    # Feature 1: n-grams
    # VECTORIZE
    if len(text_var0)==2:
        vectorizer = CountVectorizer(ngram_range=(1, 3), analyzer="word", tokenizer=None, stop_words='english', preprocessor=None,
                                        max_features=mx_ftr_gram)
        vectorizer.fit(texts_preprocessed)
        training_gram = vectorizer.transform(texts_preprocessed).toarray()
        test_gram = vectorizer.transform(TEST_texts_preprocessed).toarray()
        # Feature 2: clusters
        clusters_train = []; clusters_train_ = [];
        clusters_dev = []; clusters_dev_ = [];
        #PROGRAMMING TIP: c++ style coding here can help when doing feature engineering.. see below
        clusters_train = getClusters(texts_preprocessed); clusters_dev = getClusters(TEST_texts_preprocessed); 
        clusters_train_ = getClusters(texts_preprocessed_); clusters_dev_ = getClusters(TEST_texts_preprocessed_)
        #VECTORIZE
        cluster_train = clustervectorizer.fit_transform(clusters_train).toarray(); cluster_dev = clustervectorizer.transform(clusters_dev).toarray();
        cluster_train_ = clustervectorizer.fit_transform(clusters_train_).toarray(); cluster_dev_ = clustervectorizer.transform(clusters_dev_).toarray()
        train_text = np.concatenate((training_gram, cluster_train, cluster_train_), axis=1)
        test_text = np.concatenate((test_gram, cluster_dev, cluster_dev_), axis=1)
    elif len(text_var0)==1:
        if text_var0[0]=='cluster':
            # Feature 2: clusters
            clusters_train = []; clusters_train_ = [];
            clusters_dev = []; clusters_dev_ = [];
            #PROGRAMMING TIP: c++ style coding here can help when doing feature engineering.. see below
            clusters_train = getClusters(texts_preprocessed); clusters_dev = getClusters(TEST_texts_preprocessed); 
            clusters_train_ = getClusters(texts_preprocessed_); clusters_dev_ = getClusters(TEST_texts_preprocessed_)
            #VECTORIZE
            cluster_train = clustervectorizer.fit_transform(clusters_train).toarray(); cluster_dev = clustervectorizer.transform(clusters_dev).toarray();
            cluster_train_ = clustervectorizer.fit_transform(clusters_train_).toarray(); cluster_dev_ = clustervectorizer.transform(clusters_dev_).toarray()
            train_text = np.concatenate((cluster_train, cluster_train_), axis=1)
            test_text = np.concatenate((cluster_dev, cluster_dev_), axis=1)
        if text_var0[0]=='ngram':
            vectorizer = CountVectorizer(ngram_range=(1, 3), analyzer="word", tokenizer=None, stop_words='english', preprocessor=None,
                                            max_features=mx_ftr_gram)
            vectorizer.fit(texts_preprocessed)
            train_text = vectorizer.transform(texts_preprocessed).toarray()
            test_text = vectorizer.transform(TEST_texts_preprocessed).toarray()
    else:
        train_text = pd.DataFrame(data=[1] * 299, columns=['text'])
        test_text = pd.DataFrame(data=[1] * 71, columns=['text'])
    # check non-text colnames
    if non_text_cat0:
        non_text_one_hot0 = pd.get_dummies(all_data[non_text_cat0], drop_first=True)
    # combine non-text data
    if non_text_cat0:
        non_text_all0 = pd.concat([non_text_one_hot0, all_data[non_text_num0]], axis=1)
        non_text_all0 = non_text_all.fillna(-99999)
    elif non_text_num0:
        non_text_all0 = all_data[non_text_num0]
        non_text_all0 = all_data[non_text_num0]
    else:
        non_text_all0 = pd.DataFrame(data=[1] * 370, columns=['non_text'])
    # return final dataset
    non_text_TRAIN = non_text_all0[:299]
    non_text_TEST = non_text_all0[299:]    
    # Combine all features together
    train_X = np.concatenate((non_text_TRAIN, train_text), axis=1)
    test_X = np.concatenate((non_text_TEST, test_text), axis=1)
    return train_X, test_X

dropped_elements = []
pfmc_ablt = defaultdict(list)
# Continue the loop until any of the lists is not empty
while non_text_cat0 or non_text_num0 or text_var0:
    # Check the lists in a specific order and drop an element from the first non-empty list
    if non_text_num0:
        dropped_element = non_text_num0.pop(0)
    elif non_text_cat0:
        dropped_element = non_text_cat0.pop(0)
    else:
        dropped_element = text_var0.pop(0)
    # Document the dropped element
    dropped_elements.append(dropped_element)
    xgb_classifier = xgb.XGBClassifier(subsample = 0.8, objective = "reg:logistic", 
                                max_depth = 6, booster = 'gbtree', 
                                eval_metric = 'auc',
                                alpha = 0.01, reg_lambda = 0.01, learning_rate = 0.1, n_estimators = 100)
    train_X0, test_X0 = prepare_data_ablt(non_text_cat0, non_text_num0, text_var0)
    xgb_classifier = xgb_classifier.fit(train_X0, classes)
    predictions = xgb_classifier.predict(test_X0)    
    print(dropped_element,f1_score(test_classes, predictions, average='micro'))
    pfmc_ablt[dropped_element] = f1_score(test_classes, predictions, average='micro')


# for key in pfmc_ablt.keys():
#     print(key)

name_ablt = ['mini', 'pdd', 'abc', 'moc', 'fsd', 'ftt', 'age', 'mdsV', 'fr', 'gen', 'rac', 'eth', 'edu', 'f6m', 'fler', 'mdsBi', 'HY', 'loc', 'fcl', 'fog', 'clst', 'ngrm']
notes = dict(zip(name_ablt, pfmc_ablt.keys()))
pfmc_ablt = dict(zip(name_ablt, pfmc_ablt.values()))
notes_df = pd.DataFrame({'Full_name': notes},index=notes.keys())
notes_df['Annotation'] = notes_df.index + "  :  " + notes_df['Full_name']

all_notes = '\n'.join(notes_df['Annotation'])

# plot
import matplotlib.pyplot as plt
x = [str(key) for key in pfmc_ablt.keys()]
y = list(pfmc_ablt.values())
# Create a line plot
plt.plot(x, y, marker='o', linestyle='-', color = 'red', label = 'XGB Performance after Feature Removed')
# Add labels and title
plt.xlabel('Feature Removed')
plt.ylabel('Performance (Micro-Averaged F1 Score on Test Data)')
plt.title('XGBoost Performance Varies with Cumulative Removed Features')
plt.legend()
# Annotate the notes at the left bottom
left_margin = 0.1
top_margin = max(y) - 0.15  # Adjust the margins as needed
line_spacing = 0.01  # Adjust the spacing between lines
# plt.text(left_margin, top_margin + line_spacing, "Annotation for abbrs", fontsize=7, fontweight = 'bold')# Show the plot
text_box = plt.text(left_margin , top_margin - line_spacing*20, all_notes, fontsize=5,
                    bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'))
plt.show()



###################################################################################################
############################ VIII. Only Two Features: clusters & n-grams ##########################
###################################################################################################

xgb_classifier = xgb.XGBClassifier(subsample = 0.8, objective = "reg:logistic", 
                                   max_depth = 6, booster = 'gbtree', 
                                   eval_metric = 'auc',
                                   alpha = 0.01, reg_lambda = 0.01, learning_rate = 0.1, n_estimators = 100)

############################ 2 texts variables ##########################
text_var = ['cluster', 'ngram']
## define a function 
train_X1, test_X1 = prepare_data_ablt([], [], text_var)
xgb_classifier = xgb_classifier.fit(train_X1, classes)
predictions = xgb_classifier.predict(test_X1)
accuracy, f1_micro, f1_macro = accuracy_score(test_classes, predictions), f1_score(test_classes, predictions, average='micro'), f1_score(test_classes, predictions, average='macro')
# Print the results
print(f'Accuracy: {accuracy:.2f}')
print(f'Micro-averaged F1 Score: {f1_micro:.2f}')
print(f'Macro-averaged F1 Score: {f1_macro:.2f}')
# >>> print(f'Accuracy: {accuracy:.2f}')
# Accuracy: 0.97
# >>> print(f'Micro-averaged F1 Score: {f1_micro:.2f}')
# Micro-averaged F1 Score: 0.97
# >>> print(f'Macro-averaged F1 Score: {f1_macro:.2f}')
# Macro-averaged F1 Score: 0.97

############################ 1 texts variables: n-gram ##########################
text_var = ['ngram']
## define a function 
train_X1, test_X1 = prepare_data_ablt([], [], text_var)
xgb_classifier = xgb_classifier.fit(train_X1, classes)
predictions = xgb_classifier.predict(test_X1)
accuracy, f1_micro, f1_macro = accuracy_score(test_classes, predictions), f1_score(test_classes, predictions, average='micro'), f1_score(test_classes, predictions, average='macro')
# Print the results
print(f'Accuracy: {accuracy:.2f}')
print(f'Micro-averaged F1 Score: {f1_micro:.2f}')
print(f'Macro-averaged F1 Score: {f1_macro:.2f}')
# >>> print(f'Accuracy: {accuracy:.2f}')
# Accuracy: 0.63
# >>> print(f'Micro-averaged F1 Score: {f1_micro:.2f}')
# Micro-averaged F1 Score: 0.63
# >>> print(f'Macro-averaged F1 Score: {f1_macro:.2f}')
# Macro-averaged F1 Score: 0.63

############################ 1 texts variables: clusters ##########################
text_var = ['cluster']
## define a function 
train_X1, test_X1 = prepare_data_ablt([], [], text_var)
xgb_classifier = xgb_classifier.fit(train_X1, classes)
predictions = xgb_classifier.predict(test_X1)
accuracy, f1_micro, f1_macro = accuracy_score(test_classes, predictions), f1_score(test_classes, predictions, average='micro'), f1_score(test_classes, predictions, average='macro')
# Print the results
print(f'Accuracy: {accuracy:.2f}')
print(f'Micro-averaged F1 Score: {f1_micro:.2f}')
print(f'Macro-averaged F1 Score: {f1_macro:.2f}')
# >>> print(f'Accuracy: {accuracy:.2f}')
# Accuracy: 0.65
# >>> print(f'Micro-averaged F1 Score: {f1_micro:.2f}')
# Micro-averaged F1 Score: 0.65
# >>> print(f'Macro-averaged F1 Score: {f1_macro:.2f}')
# Macro-averaged F1 Score: 0.64
