#!/usr/bin/env python
# coding: utf-8

# In[8]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn import svm
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB            
from sklearn.metrics import make_scorer,recall_score, precision_score, f1_score,accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.preprocessing import label_binarize
import warnings
warnings.filterwarnings("ignore")


# In[9]:


data = pd.read_excel(r'C:\Users\lifukui\Desktop\DF\NEW\DF-NEW-去异2-寿命两段-三分类.xlsx')
#data = pd.read_csv('C:\Users\lifukui\Desktop\DF\DF.xlsx')


# In[10]:


data


# In[11]:


df= data.drop('Wavelength', axis = 1)
df= df.drop('Lifetime', axis = 1)
df= df.drop('wt', axis = 1)
df= df.drop('wλ', axis = 1)
df= df.drop('Q', axis = 1)
df= df.drop('Table', axis = 1)
#df= df.drop('lifetimetable', axis = 1)
#df= df.drop('wavetable', axis = 1)
df


# In[12]:


X = df
y = data.iloc[:,data.columns == "Table"]


# In[13]:


y = label_binarize(y, classes=[0, 1, 2])


# In[14]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,random_state=42)
print("X_train's shape is", X_train.shape,"; y_train's shape is", y_train.shape)
print("X_test's shape is", X_test.shape,"; y_test's shape is", y_test.shape)


# In[15]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler = scaler.fit(X_train)
X_train_stand = scaler.transform(X_train)
X_test_stand = scaler.transform(X_test)
X_stand = scaler.transform(X)


# In[16]:


X_train_stand=pd.DataFrame(X_train_stand,columns=X_train.columns)
X_test_stand=pd.DataFrame(X_test_stand,columns=X_test.columns)
X_stand=pd.DataFrame(X_stand,columns=X.columns)


# In[17]:


X_train_stand


# In[18]:


from sklearn.model_selection import KFold
kfold = KFold(n_splits = 5,shuffle = True,random_state = 42)
kfold3 = KFold(n_splits = 3,shuffle = True,random_state = 42)


# #XGBOOST

# In[19]:


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


# In[20]:


#alpha = 0.01  # 可以通过交叉验证选择最优值

xgb = XGBClassifier(n_estimators=3,learning_rate=0.3,subsample=1.0,min_child_weight=1,max_depth=8,gamma=0.01,
colsample_bytree=0.9,random_state=42,alpha = 0.01)
xgb.fit(X_train_stand, y_train)
xgbp = xgb.predict(X_test_stand)
xgbp_proba = xgb.predict_proba(X_test_stand)
#print(xgbp)
#print(y_test)
print(accuracy_score(y_test, xgbp))


# In[23]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import train_test_split




# 将分类器包装成一对多的形式
classifier = OneVsRestClassifier(xgb)

# 训练模型
classifier.fit(X_train, y_train)

# 预测概率
y_score = classifier.predict_proba(X_test)

# 得到每一类的概率
n_classes = y.shape[1]
fpr = dict()
tpr = dict()
roc_auc = dict()

for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
    
fig,ax = plt.subplots()

# 绘制所有ROC曲线

colors = ['blue', 'red', 'green']
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=2,
             label='ROC curve of class {0} (area = {1:0.2f})'
             ''.format(i, roc_auc[i]))


plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic for multi-class problem')
plt.legend(loc="lower right")
plt.show()
print("FPR:", fpr)
print("TPR:", tpr)
print("AUC:", roc_auc)

#fig.savefig('E:/研究生学习/机器学习+紫苏/picture/all shap value fengwo.pdf',dpi=1200, bbox_inches='tight')

fig.savefig('C:/Users/lifukui/Desktop/DF/picture/fenlei/ROC.pdf',dpi=1200, bbox_inches='tight')

