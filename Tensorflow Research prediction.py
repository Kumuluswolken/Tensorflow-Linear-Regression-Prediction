#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
df=pd.read_csv("predict.csv")


# In[2]:


df=df.drop(["Serial No."],axis=1)
df=df.rename(index=str, columns={"GRE Score": "GRE", "TOEFL Score": "TOEFL","University Rating": "University_Rating","Chance of Admit":'Chance'})


# In[3]:


cols_to_norm=['GRE',"TOEFL",'University_Rating','SOP','LOR','CGPA','Chance']
df[cols_to_norm]=df[cols_to_norm].apply(lambda x:(x-x.min())/(x.max()-x.min()))
#Here I build the features
GRE_feature=tf.feature_column.numeric_column('GRE')
TOEFL_feature=tf.feature_column.numeric_column('TOEFL')
Rating_feature=tf.feature_column.numeric_column('University_Rating')
SOP_feature=tf.feature_column.numeric_column('SOP')
LOR_feature=tf.feature_column.numeric_column('LOR')
CGPA_feature=tf.feature_column.numeric_column('CGPA')
Chance_feature=tf.feature_column.numeric_column('Chance')
feat_column=[GRE_feature,TOEFL_feature,Rating_feature,SOP_feature, LOR_feature,CGPA_feature,Chance_feature]
#Here I build two datasets, 1 that is used for the features and 1 that is to be predicted
X_data=df.drop('Research', axis=1)
labels=df['Research']
#Here I perform the train and test split
X_train,X_test,y_train,y_test=train_test_split(X_data,labels,test_size=0.33,random_state=101)
input_func=tf.estimator.inputs.pandas_input_fn(x=X_train,y=y_train,batch_size=10,num_epochs=1000,shuffle=True)
#The model is built
model=tf.estimator.LinearClassifier(feature_columns=feat_column, n_classes=2)
#The model is trained
model.train(input_fn=input_func, steps=1000)
pred_input_func=tf.estimator.inputs.pandas_input_fn(x=X_test,batch_size=10,num_epochs=1,shuffle=False)
#Predictions are being made
predictions=model.predict(input_fn=pred_input_func)
final_preds=[]
for pred in predictions:
    final_preds.append(pred['class_ids'][0])
#The performance is outputted
print (classification_report(y_test,final_preds))


# In[ ]:




