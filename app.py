
import  sys
sys.path.append('./pipeline')
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from pipeline.Tokens import tokenization
from pipeline.cleandata import datacleaning
from pipeline.pos_filter import POS_filter
from pipeline.stemming  import stemming
from pipeline.lemmatization import  lemma
from pipeline.tf_idf import tfidf
from pipeline.train_save import train_and_save
from pipeline.evaluate import evaluate
import os
import numpy as np

data=pd.read_csv('data\data.csv',index_col=0)
data=datacleaning(data)
data=tokenization(data)
data=lemma(data)
data=POS_filter(data)

#Embeddings
tfidf_embedding=tfidf(data)
tfidf_matrix=tfidf_embedding.transform(data['filtered_text'])


label_encoder = LabelEncoder()
y=label_encoder.fit_transform(data['sentiment'])
X=tfidf_matrix
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

folder_path = './data'
os.makedirs(folder_path, exist_ok=True)

# Save X_train and y_train as .npy files for github actions
np.save(os.path.join(folder_path, 'X_train.npy'), X_train)
np.save(os.path.join(folder_path, 'y_train.npy'), y_train)

#Train and Save the model 
train_and_save(X_train,y_train)


