# Python Command Line Based Movie Genre Classifier

Movie_Classifier is a Python command line based tool which takes a movies TITLE and a short DESCRIPTION as input and using it identifies the genres that a movie can be categorized into.

For example, if one has to identify the genre of movie 'hollywood chainsaw hookers'
with description as follows:'private eye hired worried mother find missing runaway daughter samantha private dick jack chandler searches whereabouts misfortune encountering evil cult worships egyptian god methods human sacrifice using chainsaws choice appeasing deity chandler learns samantha revenge store master bevy blood thirsty chainsaw wielding hookers'

our application would identify it as 'Horror'
Giving us an output in form of dictionary as follows:
{ Title: 'hollywood chainsaw hookers'
  Description: 'private eye hired worried mother find missing runaway daughter samantha private dick jack chandler searches whereabouts misfortune encountering evil cult worships egyptian god methods human sacrifice using chainsaws choice appeasing deity chandler learns samantha revenge store master bevy blood thirsty chainsaw wielding hookers'
  Genre: 'Horror' }
  
# Whats under the hood of the application?
We are using trained Machine Learning algorithm to make these smart guess. We are training our Model based on Movie-lens dataset and then using this trained model to make prediction on new movie plots and titles.


#prerequisite: The system running CLI should have python installed on it.

#Installation:

1. Clone my repository 'git clone  https://github.com/Piyush1416/movie_classifier.git'

2. Install conda environment in the system

3. Navigate to directory where you cloned the github project.

4. In the directory there would be file called install.sh in conda terminal run 'install.sh'
    (install.sh is shell script which does the heavy lifting for us by using setup.py and installing all necessary packages)
	
5. Once that is done you are good to go, in terminal type:
   'python movie_classifier --title "movie name" -description "short description of movie"'
 
6. That would give us a Dictionary as output with movie Genre detected.


#Programmming Laguage, Libraries and Algorithm

1. I have used Python as programming language.

2. Scikit-learn is used as major package for Machine Learning Algorithms along with few other pre-processing libraries like nltk for text cleaning and Pandas for dataframe processing.

3. we will treat this multi-label classification problem as a Binary Relevance problem. So we have to one hot encode the target variable, i.e., genre by using sklearn’s MultiLabelBinarizer( ). Since there are 32 unique genre tags, there are going to be 32 new target variables.

we will have to build a model for every one-hot encoded target variable. Since we have 32 target variables, we will have to fit 32 different models with the same set of predictors (TF-IDF features). So that is going to be computationaly expensive and time consuming so we better start with an algorithm that is quick.
So I have selected, Logistic Regression as my first preference, followed by decision tree and SVM.
We will use sk-learn’s OneVsRestClassifier class to solve this problem as a Binary Relevance or one-vs-all problem:


# How to reproduce the work, training:

Current model been used is trained using movieLens Dataset on Logistic Regression algorithm with train-test split of 80-20%.
However, we can retrain the model by using following commands:

'python movie_classifier --split 0.2 --classifier "LogisticRegression" --dataset "data/movies_metadata.csv"'

of which dataset field is optional....if you wish to stick with already available Movie-lens dataset and try experimenting with different algorithms and train-test split size.

At end of training we get a F1 score for our model, the current model gives an F1 score of 0.46 which needs to be improved...the closer your models F1 score to 1.0 the better your model is going to be.


# Tests carried out & Robust Input check

I have conducted unit testing,based on different criterias:

1. How well is Text_Preprocessing been done

2. Selecting Model which scores higher F1-Score than current model after training

3. If empty strings for TITLE & DESCRIPTION are given as inputs.


#Comments are Present in code