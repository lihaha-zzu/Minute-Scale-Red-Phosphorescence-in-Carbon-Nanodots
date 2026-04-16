#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from xgboost import XGBClassifier       
from sklearn.metrics import make_scorer,recall_score, precision_score, f1_score,accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import roc_curve, roc_auc_score
import warnings
warnings.filterwarnings("ignore")


# In[4]:


data = pd.read_excel(r'C:\Users\lifukui\Desktop\DF\NEW\DF-NEW-去异2-寿命两段-三分类.xlsx')
#data = pd.read_csv('C:\Users\lifukui\Desktop\DF\DF.xlsx')


# In[5]:


df= data.drop('Wavelength', axis = 1)
df= df.drop('Lifetime', axis = 1)
df= df.drop('wt', axis = 1)
df= df.drop('wλ', axis = 1)
df= df.drop('Q', axis = 1)
df= df.drop('Table', axis = 1)
#df= df.drop('lifetimetable', axis = 1)
#df= df.drop('wavetable', axis = 1)
df


# In[6]:


X = df
y = data.iloc[:,data.columns == "Table"]
y = np.ravel(y)


# In[7]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,random_state=42)
print("X_train's shape is", X_train.shape,"; y_train's shape is", y_train.shape)
print("X_test's shape is", X_test.shape,"; y_test's shape is", y_test.shape)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler = scaler.fit(X_train)
X_train_stand = scaler.transform(X_train)
X_test_stand = scaler.transform(X_test)
X_stand = scaler.transform(X)
X_train_stand=pd.DataFrame(X_train_stand,columns=X_train.columns)
X_test_stand=pd.DataFrame(X_test_stand,columns=X_test.columns)
X_stand=pd.DataFrame(X_stand,columns=X.columns)
from sklearn.model_selection import KFold
kfold = KFold(n_splits = 5,shuffle = True,random_state = 42)
kfold3 = KFold(n_splits = 3,shuffle = True,random_state = 42)


# In[8]:


from xgboost.sklearn import XGBClassifier

xgb = XGBClassifier(n_estimators=3,learning_rate=0.3,subsample=1.0,min_child_weight=1,max_depth=8,gamma=0.01,
colsample_bytree=0.9,random_state=42,alpha = 0.01)
xgb_accuracy = cross_val_score(xgb,X_stand,y,cv=kfold,scoring=make_scorer(accuracy_score))
xgb_precision = cross_val_score(xgb,X_stand,y,cv=kfold,scoring=make_scorer(precision_score, average='macro'))
xgb_recall = cross_val_score(xgb,X_stand,y,cv=kfold,scoring=make_scorer(recall_score, average='macro'))
xgb_f1 = cross_val_score(xgb,X_stand,y,cv=kfold,scoring=make_scorer(f1_score, average='macro'))
mean_xgb_accuracy = (sum(xgb_accuracy)/5)
mean_xgb_precision = (sum(xgb_precision)/5)
mean_xgb_recall = (sum(xgb_recall)/5)
mean_xgb_f1 = (sum(xgb_f1)/5)
print('mean_xgb_accuracy:',mean_xgb_accuracy)
print('mean_xgb_precision:',mean_xgb_precision)
print('mean_xgb_recall:',mean_xgb_recall)
print('mean_xgb_f1:',mean_xgb_f1)
print('xgb_accuracy:',xgb_accuracy)
print('xgb_precision:',xgb_precision)
print('xgb_recall:',xgb_recall)   
print('xgb_f1:',xgb_f1)


# In[9]:


#alpha = 0.01  # 可以通过交叉验证选择最优值

xgb = XGBClassifier(n_estimators=3,learning_rate=0.3,subsample=1.0,min_child_weight=1,max_depth=8,gamma=0.01,
colsample_bytree=0.9,random_state=42,alpha = 0.01)
xgb.fit(X_train_stand, y_train)
xgbp = xgb.predict(X_test_stand)
xgbp_proba = xgb.predict_proba(X_test_stand)
#print(xgbp)
#print(y_test)
print(accuracy_score(y_test, xgbp))
xgb.feature_importances_


# In[10]:


cm = confusion_matrix(y_test, xgbp)
plt.figure(figsize=(5, 4))
disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=['0','1','2'])
disp.plot(cmap='Blues', values_format='d')
plt.title('Confusion Matrix')
#plt.savefig("C:/Users/lifukui/Desktop/DF/picture/fenlei/Confusion Matrix.pdf")

plt.show()


# In[11]:


xgb.feature_importances_


# In[ ]:





# In[12]:


import itertools
import pandas as pd

parameter_values = {
    'Temperature': [130,135,140,145,150,155,160,165,170,175,185,190,195,200,205,210],
    'RHB': [500,550,600,700,750,800,850,900,950,1000,1050,1100,1150,1200,1250,1300,1350,1400],
    'P-Ratio': [0.4,0.5,0.55,0.6,0.7,0.75,0.8,0.85,0.9,0.95,1.0,1.1,1.05,1.15,1.2],
    'Time': [3,4,5,6,7,8,9,10,11,12,13,14],
    'Water': [0,50,100,150,200,250,300],
    'alcohol': [0,50,100,150,200,250,300]
}

parameter_names = list(parameter_values.keys())
parameter_list = list(parameter_values.values())

cartesian_product = list(itertools.product(*parameter_list))

prediction = pd.DataFrame(cartesian_product, columns=parameter_names)


# In[14]:


X_prediction = scaler.transform(prediction)


# In[15]:


xgb_prediction = xgb.predict(X_prediction)


# In[16]:


prediction['prediction'] =xgb_prediction


# In[19]:


prediction

