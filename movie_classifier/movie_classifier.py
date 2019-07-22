import pandas as pd
import json
import nltk
import re
nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
import csv
import argparse

from sklearn.preprocessing import MultiLabelBinarizer


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

# Binary Relevance
from sklearn.multiclass import OneVsRestClassifier

# Performance metric
from sklearn.metrics import f1_score

from sklearn.externals import joblib


class Preprocess_Data:

    def __init__(self, dataset_file):
        self.movies = pd.read_csv(dataset_file, low_memory=False)
        print(self.movies.shape)

    def clean_genres(self,dataset):
        genre_new = []
        mylist = eval(dataset)
        for x in mylist:
            genre_new.append(x['name'])
        return genre_new

    def Get_Req_Columns_DataFrame(self):
        #Selecting 3 columns which are required for processing while eliminating the rest
        self.movies = self.movies.filter(['genres','original_title','overview'])


    def Genre_Filtering(self):
        # remove samples with 0 genre tags as they are of no use
        self.movies = self.movies[~(self.movies['genres'].str.len() == 0)]
        # At same time we even get all genre tags in a list
        all_genres = sum(self.movies['genres'],[])
        print(set(all_genres))
        print(len(set(all_genres)))

    # function for text cleaning
    def clean_text_overview(self,text):
        # print(type(text))
        text = str(text)
        # remove backslash-apostrophe
        text = re.sub("\'", "", text)
        # remove everything except alphabets
        text = re.sub("[^a-zA-Z]", " ", text)
        # remove whitespaces
        text = ' '.join(text.split())
        # convert text to lowercase
        text = text.lower()
        # Let’s remove the stopwords
        no_stopword_text = [w for w in text.split() if not w in stop_words]
        return ' '.join(no_stopword_text)

    def create_training_dataset(self):
        #print('Old Genres==>'+str(self.movies['genres'][0]))
        self.movies['genres'] = self.movies['genres'].apply(lambda x: self.clean_genres(x))
        #print(self.movies['genres'][0])
        #print('Cleaned Genres==>' + str(self.movies['genres'][0]))
        self.Get_Req_Columns_DataFrame()
        self.Genre_Filtering()
        #We need, clean data for training Let’s apply the clean_text_overview function on the movie overview by using the apply-lambda duo:
        self.movies['overview'] = self.movies['overview'].apply(lambda x: self.clean_text_overview(x))
        print(self.movies.shape)


class Training_Data:

    def __init__(self, df_sampl,dataset_split):
        self.movies_new = df_sampl
        print('Inside Training')
        print(self.movies_new.shape)
        self.dataset_split = dataset_split

    # Converting Text to Features

    """
    we will treat this multi-label classification problem as a Binary Relevance problem. 
    Hence, we will now one hot encode the target variable, i.e., genre by using sklearn’s MultiLabelBinarizer( ). 
    Since there are 32 unique genre tags, there are going to be 32 new target variables.
    """

    def Text_to_Features(self):
        multilabel_binarizer = MultiLabelBinarizer()
        multilabel_binarizer.fit(self.movies_new['genres'])

        # transform target variable
        y = multilabel_binarizer.transform(self.movies_new['genres'])
        joblib.dump(multilabel_binarizer, 'data/multilabel_binarizer.pkl')
        return y

    def Training_Initialization(self):
        print('Initiated Training Just now.....')
        y = self.Text_to_Features()
        # split dataset into training and validation set
        xtrain, xval, ytrain, yval = train_test_split(self.movies_new['overview'], y, test_size=self.dataset_split, random_state=9)

        # I will be using TF-IDF, to extract features from the cleaned version of the movie overview data

        # I am using top 15000 most frequent words in my data for feature set

        #tfidf_vectorizer = TfidfVectorizer(max_df=0.6, max_features=25000)
        tfidf_vectorizer = TfidfVectorizer()
        # create TF-IDF features
        xtrain_tfidf = tfidf_vectorizer.fit_transform(xtrain)
        xval_tfidf = tfidf_vectorizer.transform(xval)
        joblib.dump(tfidf_vectorizer, 'data/tf_idf_vectorizer_model.pkl')
        # Remember, we will have to build a model for every one-hot encoded target variable. Since we have 32 target variables, we will have to fit 32 different models with the same set of predictors (TF-IDF features).

        # We will use sk-learn’s OneVsRestClassifier class to solve this problem as a Binary Relevance or one-vs-all problem:

        lr = LogisticRegression()
        clf = OneVsRestClassifier(lr)
        # fit model on train data
        clf.fit(xtrain_tfidf, ytrain)

        # Save the model as a pickle in a file
        joblib.dump(clf, 'data/genre_predictor_model.pkl')
        print('Training Completed')

        # make predictions for validation set
        y_pred = clf.predict(xval_tfidf)

        # To evaluate our model’s overall performance, we need to take into consideration all the predictions and the entire target variable of the validation set:

        # evaluate performance
        print('Our trained model scored an F1-score of:')
        print(f1_score(yval, y_pred, average="micro"))
        f1 = f1_score(yval, y_pred, average="micro")
        return f1


class Model_Testing(Preprocess_Data):

    def __init__(self,title,description ):
        print('Inside Test Funct')
        self.title = title
        self.description = description
        data = [[title,description]]
        self.df = pd.DataFrame(data, columns=['original_title','overview'])
        print(self.df.shape)
        # Load the model from the file
        self.genre_prediction = joblib.load('data/genre_predictor_model.pkl')

    def pred(self):
        # We clean overview text, by using clean_text_overview method that is been inherited from Parent Class PreProcess_Data
        self.df['overview'] = self.df['overview'].apply(lambda x: self.clean_text_overview(x))
        print(self.df)
        #tfidf_vectorizer = TfidfVectorizer()
        self.tfidf_vectorizer = joblib.load('data/tf_idf_vectorizer_model.pkl')
        xval_tfidf = self.tfidf_vectorizer.transform(self.df['overview'])
        #print(xval_tfidf)
        y_pred = self.genre_prediction.predict(xval_tfidf)
        print(y_pred)
        self.multilabel_binarizer = joblib.load('data/multilabel_binarizer.pkl')
        # Luckily, sk-learn comes to our rescue once again. We will use the inverse_transform( ) function along with the MultiLabelBinarizer( ) object to convert the predicted arrays into movie genre tags:
        print(self.multilabel_binarizer.inverse_transform(y_pred))
        op = self.multilabel_binarizer.inverse_transform(y_pred)
        print(len(op))
        if(len(op) == 1):
            for gen in op:
                print(len(gen))
                if(len(gen) == 0):
                    return "['Drama']"
                else:
                    return op
        else:
            return op


