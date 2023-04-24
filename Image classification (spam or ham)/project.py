import numpy as np 
import pandas as pd 
#for visualisation
import seaborn as sns
import matplotlib.pyplot as plt

#for NLP 
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import one_hot
#for classification with neural network
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Embedding
from tensorflow.keras.models import Sequential

#for machine learning classification
from sklearn.cluster import KMeans
from sklearn.model_selection import KFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import BernoulliNB

from sklearn.model_selection import KFold, cross_val_score
##telecharger les Stops Words
nltk.download('stopwords')
from sklearn.metrics import accuracy_score
from tensorflow.keras.optimizers import Adam
##uploader le fichier qui contient dataset
# uploaded=files.upload()


# data=pd.read_csv('spamm.csv',encoding='iso-8859-1')

data=pd.read_csv('C:/Users/yassine\Desktop/piton/spamm.csv',encoding='iso-8859-1')
d=data.head()
print(d)
data.columns


data=data.drop(columns=["Unnamed: 2","Unnamed: 3","Unnamed: 4"])


data=data.rename(
{
    "isspam":"Category",
    "v2":"Message"
},
    axis=1
)

data.head()


data.isnull().sum()


data.info()
data["Message Length"]=data["Message"].apply(len)
fig=plt.figure(figsize=(12,8))
sns.histplot(
    x=data["Message Length"],
    hue=data["Category"]
)
plt.title("ham & spam messege length comparision")
plt.show()
data.describe(include="all")
data["Category"].value_counts()


sns.countplot(
    data=data,
    x="Category"
)
plt.title("ham vs spam")
plt.show()
ham_count=data["Category"].value_counts()[0]
spam_count=data["Category"].value_counts()[1]

total_count=data.shape[0]
print("Ham contains:{:.2f}% of total data.".format(ham_count/total_count*100))
print("Spam contains:{:.2f}% of total data.".format(spam_count/total_count*100))
#compute the length of majority & minority class
minority_len=len(data[data["Category"]=="spam"])
majority_len=len(data[data["Category"]=="ham"])

#store the indices of majority and minority class
minority_indices=data[data["Category"]=="spam"].index
majority_indices=data[data["Category"]=="ham"].index

#generate new majority indices from the total majority_indices
#with size equal to minority class length so we obtain equivalent number of indices leng
random_majority_indices=np.random.choice(
    majority_indices,
    size=minority_len,
    replace=False
)

random_majority_indices
#concatenate the two indices to obtain indices of new dataframe
undersampled_indices=np.concatenate([minority_indices,random_majority_indices])

#create df using new indices
df=data.loc[undersampled_indices]

#shuffle the sample
df=df.sample(frac=1)

#reset the index as its all mixed
df=df.reset_index()

#drop the older index
df=df.drop(
    columns=["index"],
)
df.head()
dd=df
df.head()
##############################""
sns.countplot(
    data=df,
    x="Category"
)
plt.title("ham vs spam")
plt.show()

#############################
df["Label"]=df["Category"].map(
    {
        "ham":0,
        "spam":1
    }
)
#########################################################################


stemmer=PorterStemmer()
##Tokenizationand stemming
#declare empty list to store tokenized message
corpus=[]

#iterate through the df["Message"]
for message in df["Message"]:
    
    #replace every special characters, numbers etc.. with whitespace of message
    #It will help retain only letter/alphabets
    message=re.sub("[^a-zA-Z]"," ",message)
    
    #convert every letters to its lowercase
    message=message.lower()
    
    # #split the word into individual word list
    message=message.split()
    # # message
    # message=ngrams(message,2)
    #perform stemming using PorterStemmer for all non-english-stopwords
    message=[stemmer.stem(words)
            for words in message
             if words not in set(stopwords.words("english"))
            ]
    #join the word lists with the whitespace
    message=" ".join(message)
    
    # append the message in corpus list
    corpus.append(message)

###

vocab_size=10000

oneHot_doc=[one_hot(words,n=vocab_size)
           for words in corpus
           ]

df["Message Length"].describe()

fig=plt.figure(figsize=(12,8))
sns.kdeplot(
    x=df["Message Length"],
    hue=df["Category"]
)
plt.title("ham & spam messege length comparision")
plt.show()

sentence_len=200
# sentence_len=4647
embedded_doc=pad_sequences(
    oneHot_doc,
    maxlen=sentence_len,
    padding="pre"
)


extract_features=pd.DataFrame(
    data=embedded_doc
)
target=df["Label"]


df_final=pd.concat([extract_features,target],axis=1)


df_final.head()
embedded_doc
###########SPLIT DATASET
from sklearn.model_selection import train_test_split
X=df_final.drop("Label",axis=1)
y=df_final["Label"]

X_trainval,X_test,y_trainval,y_test=train_test_split(
    X,
    y,
    random_state=42,
    test_size=0.20
)


X_train,X_val,y_train,y_val=train_test_split(
    X_trainval,
    y_trainval,
    random_state=42,
    test_size=0.20
)
##############################################
## CHOISIR TON ALGORITHM DE CLASSIFICATION  ##
##############################################
model=Sequential()

feature_num=100
model.add(
    Embedding(
        input_dim=vocab_size,
        output_dim=feature_num,
        input_length=sentence_len
    )
)
model.add(
    LSTM(
    units=128
    )
)
model.add(
    Dense(
        units=1,
        activation="sigmoid"
    )
)
model.compile(
    optimizer=Adam(
    learning_rate=0.001
    ),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.fit(
    X_train,
    y_train,
    validation_data=(
        X_val,
        y_val
    ),
    epochs=10
)
# y_pred=model.predict(X_test)

y_pred=model.predict(X_test)
y_pred=(y_pred>0.5)
score=accuracy_score(y_test,y_pred)
print("Test Score:{:.2f}%".format(score*100))

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)
fig=plt.figure(figsize=(12,8))
sns.heatmap(
    cm,
    annot=True,
)
plt.title("Confusion Matrix")
cm
#knn model 
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import KFold, cross_val_score
# Create k-Fold cross-validation
kf = KFold(n_splits=10, shuffle=True, random_state=1)

model1=KNeighborsClassifier(n_neighbors=2)

model1.fit(
    X_trainval,
    y_trainval)
# Conduct k-fold cross-validation
cv_results = cross_val_score(model1,
X, # Feature matrix
y, # Target vector
cv=kf, # Cross-validation technique
scoring="accuracy", # Loss function
n_jobs=-1) # Use all CPU scores
# Calculate mean
cv_results.mean()
# model1.fit(X_train,y_train)
y_predi=model1.predict(X_test)
y_predi=(y_predi>0.5)
score1=accuracy_score(y_test,y_predi)
print("Test Score:{:.2f}%".format(score1*100))

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_predi)
fig=plt.figure(figsize=(12,8))
sns.heatmap(
    cm,
    annot=True,
)
plt.title("Confusion Matrix")
cm
#logistic regression 
from sklearn.linear_model import LogisticRegression

from sklearn.preprocessing import StandardScaler
# Load data with only two classes

features = X
target = y
# Standardize features
scaler = StandardScaler()
features_standardized = scaler.fit_transform( X_trainval)
# Create logistic regression object
logistic_regression = LogisticRegression(random_state=0)
# Train model

model2= logistic_regression.fit(features_standardized ,y_trainval)
from sklearn.model_selection import KFold, cross_val_score
# Create k-Fold cross-validation
kf = KFold(n_splits=10, shuffle=True, random_state=1)


# Conduct k-fold cross-validation
cv_results = cross_val_score(model2,
X, # Feature matrix
y, # Target vector
cv=kf, # Cross-validation technique
scoring="accuracy", # Loss function
n_jobs=-1) # Use all CPU scores
# Calculate mean
cv_results.mean()
y_predic12=model2.predict(X_test)
y_predic12=(y_predic12>0.5)
score9=accuracy_score(y_test,y_predic12)
print("Test Score:{:.2f}%".format(score9*100))

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_predic12)
fig=plt.figure(figsize=(12,8))
sns.heatmap(
    cm,
    annot=True,
)
plt.title("Confusion Matrix")
cm
#naive bayes

import numpy as np
from sklearn.naive_bayes import BernoulliNB
# Create three binary features
# features = np.random.randint(2, size=(100, 1))
features =X
# Create a binary target vector
target = y
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
 
features_standardized = scaler.fit_transform(X_trainval)
# Create Bernoulli Naive Bayes object with prior probabilities of each class
classifer = BernoulliNB(class_prior=[0.25, 0.5])
# Train model
model3 = classifer.fit(features_standardized,y_trainval)
from sklearn.model_selection import KFold, cross_val_score
# Create k-Fold cross-validation
kf = KFold(n_splits=10, shuffle=True, random_state=1)


# Conduct k-fold cross-validation
cv_results = cross_val_score(model3,
X, # Feature matrix
y, # Target vector
cv=kf, # Cross-validation technique
scoring="accuracy", # Loss function
n_jobs=-1) # Use all CPU scores
# Calculate mean
cv_results.mean()
j=model3.predict(X_test)
y_pre=model3.predict(X_test)
y_pre=(y_pre>0.5)
score14=accuracy_score(y_test,y_pre)
print("Test Score:{:.2f}%".format(score14*100))

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pre)
fig=plt.figure(figsize=(12,8))
sns.heatmap(
    cm,
    annot=True,
)
plt.title("Confusion Matrix")
cm
# #svm model
# from sklearn.svm import LinearSVC

# from sklearn.preprocessing import StandardScaler

# import numpy as np

# features =X
# target = y
# # Standardize features


# # # Create support vector classifier
# svc = LinearSVC(C=1.0)
# # Train model
# model4 = svc.fit(X_trainval,y_trainval)
# from sklearn.model_selection import KFold, cross_val_score
# # Create k-Fold cross-validation
# kf = KFold(n_splits=10, shuffle=True, random_state=1)


# # Conduct k-fold cross-validation
# cv_results = cross_val_score(model4,
# X, # Feature matrix
# y, # Target vector
# cv=kf, # Cross-validation technique
# scoring="accuracy", # Loss function
# n_jobs=-1) # Use all CPU scores
# # Calculate mean
# cv_results.mean()
# y_pre4=model3.predict(X_test)
# y_pre4=(y_pre4>0.5)
# score3=accuracy_score(y_test,y_pre4)
# print("Test Score:{:.2f}%".format(score3*100))

# from sklearn.metrics import confusion_matrix
# cm1=confusion_matrix(y_test,y_pre4)
# fig=plt.figure(figsize=(12,8))
# sns.heatmap(
#     cm,
#     annot=True,
# )
# plt.title("Confusion Matrix")
# cm1
print('ok1')
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import KFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.tree import DecisionTreeClassifier


features = X
target = y
# Create decision tree classifier object
decisiontree = DecisionTreeClassifier(random_state=0)
# Train model
model14 = decisiontree.fit(X_trainval,y_trainval)
y_pre5=model14.predict(X_test)
y_pre5=(y_pre5>0.5)
score5=accuracy_score(y_test,y_pre5)
print("Test Score:{:.2f}%".format(score5*100))
from sklearn.model_selection import KFold, cross_val_score
# Create k-Fold cross-validation
kf = KFold(n_splits=10, shuffle=True, random_state=1)

# Conduct k-fold cross-validation
cv_results = cross_val_score(model14,
X, # Feature matrix
y, # Target vector
cv=kf, # Cross-validation technique
scoring="accuracy", # Loss function
n_jobs=-1) # Use all CPU scores
# Calculate mean
cv_results.mean()
from sklearn.metrics import confusion_matrix
cm1=confusion_matrix(y_test,y_pre5)
fig=plt.figure(figsize=(12,8))
sns.heatmap(
    cm,
    annot=True,
)
plt.title("Confusion Matrix")
cm1
# Load libraries
from sklearn.ensemble import RandomForestClassifier


# Create random forest classifier object
randomforest = RandomForestClassifier(random_state=0, n_jobs=-1)
# Train model
model16 = randomforest.fit(X_trainval,y_trainval)
from sklearn.model_selection import KFold, cross_val_score
# Create k-Fold cross-validation
kf = KFold(n_splits=10, shuffle=True, random_state=1)

# Conduct k-fold cross-validation
cv_results = cross_val_score(model16,
X, # Feature matr
y, # Target vector
cv=kf, # Cross-validation technique
scoring="accuracy", # Loss function
n_jobs=-1) # Use all CPU scores
# Calculate mean
cv_results.mean()
y_pre6=model16.predict(X_test)
y_pre6=(y_pre6>0.5)
score6=accuracy_score(y_test,y_pre6)
print('random forest score')
print("Test Score:{:.2f}%".format(score6*100))

from sklearn.metrics import confusion_matrix
cm1=confusion_matrix(y_test,y_pre6)
fig=plt.figure(figsize=(12,8))
sns.heatmap(
    cm,
    annot=True,
)
plt.title("Confusion Matrix")
cm1
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import KFold, cross_val_score
# Standardize features
scaler = StandardScaler()
features_std = scaler.fit_transform(X_trainval)
# Create k-mean object
cluster = KMeans(n_clusters=2, random_state=0)
# Train model
cluster= cluster.fit(features_std)

# Create k-Fold cross-validation
kf = KFold(n_splits=10, shuffle=True, random_state=1)

# Conduct k-fold cross-validation
cv_results = cross_val_score(cluster,
X, # Feature matrix
y, # Target vector
cv=kf, # Cross-validation technique
scoring="accuracy", # Loss function
n_jobs=-1) # Use all CPU scores
# Calculate mean
cv_results.mean()
y_pre7=cluster.predict(X_test)
y_pre7=(y_pre7>0.5)
score7=accuracy_score(y_test,y_pre7)
print('kmeans score')
print("Test Score:{:.2f}%".format(score7*100))

from sklearn.metrics import confusion_matrix
cm1=confusion_matrix(y_test,y_pre7)
fig=plt.figure(figsize=(12,8))
sns.heatmap(
    cm,
    annot=True,
)
plt.title("Confusion Matrix")
cm1
#The function take model and message as parameter
print('fin')

def classify_message(model,message):
    
    #We will treat message as a paragraphs containing multiple sentences(lines)
    #we will extract individual lines
    for sentences in message:
        sentences=nltk.sent_tokenize(message)
        
        #Iterate over individual sentences
        for sentence in sentences:
            #replace all special characters
            words=re.sub("[^a-zA-Z]"," ",sentence)
            
            #perform word tokenization of all non-english-stopwords
            if words not in set(stopwords.words('english')):
                word=nltk.word_tokenize(words)
                word=" ".join(word)
    
    #perform one_hot on tokenized word            
    oneHot=[one_hot(word,n=vocab_size)]
    
    #create an embedded documnet using pad_sequences 
    #this can be fed to our model
    text=pad_sequences(oneHot,maxlen=sentence_len,padding="pre")
    
    #predict the text using model
    predict=model.predict(text)
    
    #if predict value is greater than 0.5 its a spam
    if predict>0.5:
        print("It is a spam")
    #else the message is not a spam    
    else:
        print("It is not a spam")

mes="free cash just sign in"
r=classify_message(model,mes) 
print(r)
