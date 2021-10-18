import pandas as pd 
import numpy as np 
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
#Accessing dataset
data=pd.read_csv('C:\\Users\\namit\\OneDrive\\Desktop\\presentation\\New folder\\mnist.csv')
print(data.head())
#Checking how the data looks
a=data.iloc[5,1:].values
a=a.reshape(28,28).astype('uint8')
plt.imshow(a)
plt.show()
#Separating labels and values
df_x=data.iloc[:,1:]
df_y=data.iloc[:,0]
#Splitting data inorder to test and train
x_train,x_test,y_train,y_test=train_test_split(df_x,df_y,test_size=0.2,random_state=4)
#Creating the model and fitting data
rf=RandomForestClassifier(n_estimators=100)
rf.fit(x_train,y_train)
#Predicting values
pred=rf.predict(x_test)
print(pred)
#Finding accuracy
count=0
for i in y_test.values:
    if i in pred:
        count=count+1
print("Accuracy=",count/len(pred))        

