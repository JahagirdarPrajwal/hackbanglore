pip install kaggle

# configure the path
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json

# api to fetch the dataset from kaggle
!kaggle datasets download -d kazanova/sentiment140

# extracting the compressed dataset

from zipfile import ZipFile # reads zipfile 
zdataset = '/content/sentiment140.zip'

with ZipFile(dataset,'r') as zip:
  zip.extractall()
  print('Done')


import numpy as np
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer # used to feed the data from vector to numerical
from sklearn.model_selection import train_test_split # used to split the data into training and test
from sklearn.linear_model import LogisticRegression # training data
from sklearn.metrics import accuracy_score #accuracy calculator

import nltk
nltk.download('stopwords') # stopwords are those which doesnt have influential meaning to the data

# printing the stopwords in english
print(stopwords.words('english'))

#laoding the data from csv file
twitter_data = pd.read_csv('/content/training.1600000.processed.noemoticon.csv',encoding='ISO-8859-1')

# chechking the number of rows and column
twitter_data.shape # 16lakhs reach as in python it starts from 0 ( 16 laksh rows and 6 column)

# printing the first 5 rows
twitter_data.head()

# naming the columns and reading the dataset again as its reading the heading of the table as a individual column

column_names = ['target','id','date','flag','user','text'] # data given on the websitez
twitter_data = pd.read_csv('/content/training.1600000.processed.noemoticon.csv', names=column_names,encoding='ISO-8859-1')

# chechking the number of rows and column
twitter_data.shape # the first tweet was read as column name

# printing the first 5 rows
twitter_data.head()

# chechking and counting of missing values
twitter_data.isnull().sum() # isnull helps for missing values

#CHECHKING   THE DISTRIBUTION OF TARGET COLUMN
twitter_data['target'].value_counts() # data if 0 means negative and data if 4 means positive data has equal distribution so its working otherwise would have to make it equal to make it work

twitter_data.replace({'target':{4:1}}, inplace=True)

#CHECHKING   THE DISTRIBUTION OF TARGET COLUMN
twitter_data['target'].value_counts() # converted it from 4 to 1

port_stem = PorterStemmer()

def stemming(content):

  steamed_content = re.sub ('[^a-zA-Z]',' ',content) # removing everything which is not letter
  steamed_content = steamed_content.lower() # converting uppercase to lowercase
  steamed_content = steamed_content.split() # spliting the words
  steamed_content = [port_stem.stem(word) for word in steamed_content if not word in stopwords.words('english')] # check for stopwords
  steamed_content = ' '.join(steamed_content)

  return steamed_content

twitter_data['stemmed_contet'] = twitter_data['text'].apply(stemming)

twitter_data.head()

print(twitter_data['stemmed_contet'])

print(twitter_data['target']) # we dont need id data flag for checking if its negative or positive

# separting the data and label 
X = twitter_data['stemmed_contet'].values # all the tweets are stored here 
Y = twitter_data['target'].values # all the target values are stored here 
print(X) # printing variables seperately 
print(Y) # printing variables seperately
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2, stratify=Y, random_state=2) # X_train contains all the training data and labels in y_train y test contains all the negative and positive for all the xtest  
# spliting the x and y test size 0.2 = 20 percent data will go for test purpose and rest for training  
#  stratify will give the almost equal proportion of training data and random state 2 will replicate 

print(X.shape, X_train.shape, X_test.shape)

print(X_train)

print(X_test)

# converting the text data to numerical data 

vectorizer = TfidfVectorizer() # used for convert text to numerical for model to unerstand 
# will check the no of times a word has occured and get the importance required 
# if a word was occured many times in a positive term then it will be termed as positive 

X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

print(X_train) # starting 0 means words in the first tweet and similarly 1 means tweet in the second tweet 

print(X_test)

model = LogisticRegression(max_iter=1000) # max no of times model will go through the data 
model.fit(X_train, Y_train) # x train is training data y train is target for training data 
# feeding the xtrain and y train to the model 

# Accuracy score of the train data 
X_train_prediction = model.predict(X_train) # predits for 0 and 1 
training_data_accuracy = accuracy_score(Y_train, X_train_prediction) # accuracy score is the function 

print('Accuracy score of the data : ', training_data_accuracy)
# Accuracy score of the test data 
X_test_prediction = model.predict(X_test) # predits for 0 and 1 
test_data_accuracy = accuracy_score(Y_test, X_test_prediction) # accuracy score is the function 

print('Accuracy score of the data : ', test_data_accuracy) # as the accuracy percentage for both test and train data are close we can say the model worked properly

import pickle

filename = 'trained_model.sav'
pickle.dump(model, open(filename, 'wb')) # dumping all the saved trained model 
# loading the saved model 
loaded_model = pickle.load(open('/content/trained_model.sav', 'rb')) # rb is reading the file in binary format 
X_new = X_test[350]
print(Y_test[350])

prediction = model.predict(X_new) # saves the new prediction 
print(prediction) # prints the new value is 0 or 1 \

if prediction[0] == 0:
  print('The tweet is negative')
else:
  print('The tweet is positive')

  # [1] is a list 

  X_new = X_test[100]
print(Y_test[100])

prediction = model.predict(X_new) # saves the new prediction 
print(prediction) # prints the new value is 0 or 1 \

if prediction[0] == 0:
  print('The tweet is negative')
else:
  print('The tweet is positive')

  # [1] is a list 