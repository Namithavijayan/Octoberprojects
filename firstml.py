import pandas
from pandas import read_csv
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
#importing dataset
dataset=read_csv('C:\\Users\\namit\\OneDrive\\Desktop\\presentation\\New folder\\iris.csv')
#Checking data
print(dataset.shape)
print(dataset.head(10))
#Summarizing data
print(dataset.describe())
#Class distribution
print(dataset.groupby('Species').size())
#Univariate plot - box and whisker
dataset.plot(kind='box',subplots=True,layout=(3,3),sharex=False,sharey=False)
pyplot.show()
#Histogram
dataset.hist()
pyplot.show()
#Multivariate plot
scatter_matrix(dataset)
pyplot.show()
#creating validation dataset
array=dataset.values
x=array[:,0:4]
y=array[:,4]
#Splitting data
x_train,x_valid,y_train,y_valid=train_test_split(x,y,test_size=0.2,random_state=10)
#Creating models
models=[]
models.append(['LR',LogisticRegression(solver='liblinear',multi_class='ovr')])
models.append(['LDR',LinearDiscriminantAnalysis()])
models.append(['KNN',KNeighborsClassifier()])
models.append(['NB',GaussianNB()])
models.append(['SVM',SVC(gamma='auto')])
#Evaluating accuracy of models
result=[]
names=[]
for name,model in models:
    Kfold=StratifiedKFold(n_splits=10,random_state=1,shuffle=True)
    cv_result=cross_val_score(model,x_train,y_train,cv=Kfold,scoring='accuracy')
    result.append(cv_result)
    names.append(name)
    print('\s: \f (\f)',name,cv_result.mean(),cv_result.std())
#Compare models
pyplot.boxplot(result,labels=names)
pyplot.title('Algorithm Accuracy')
pyplot.show()
#LDR showed more accuracy than others
model=LinearDiscriminantAnalysis()
model.fit(x_train,y_train)
pred=model.predict(x_valid)
#Evaluating accuracy of LDR
print(accuracy_score(y_valid,pred))
print(confusion_matrix(y_valid,pred))
print(classification_report(y_valid,pred))