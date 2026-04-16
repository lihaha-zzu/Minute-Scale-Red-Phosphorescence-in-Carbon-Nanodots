#!/usr/bin/env python
# coding: utf-8

# In[6]:


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
import warnings
warnings.filterwarnings("ignore")


# In[8]:


data = pd.read_excel(r'C:\Users\lifukui\Desktop\DF\NEW\DF-NEW-去异2-寿命两段-三分类.xlsx')
#data = pd.read_csv('C:\Users\lifukui\Desktop\DF\DF.xlsx')


# In[10]:


data


# In[12]:


df= data.drop('Wavelength', axis = 1)
df= df.drop('Lifetime', axis = 1)
df= df.drop('wt', axis = 1)
df= df.drop('wλ', axis = 1)
df= df.drop('Q', axis = 1)
df= df.drop('Table', axis = 1)
#df= df.drop('lifetimetable', axis = 1)
#df= df.drop('wavetable', axis = 1)
df


# In[279]:


correlation_matrix = df.corr()
cmap = sns.diverging_palette(220,8,as_cmap = True)
plt.figure(figsize=(5, 4))
sns.heatmap(correlation_matrix, annot=True, cmap=cmap, vmin = -1,vmax= 1,
            cbar_kws={'orientation':'vertical','ticks':[-1,-0.5,0,0.5,1]},
            linewidths=.5)
plt.savefig("C:/Users/lifukui/Desktop/DF/picture/fenlei/pierxun.pdf")
#plt.savefig("E:/研究生学习/机器学习+紫苏/picture/pierxun.pdf")
plt.show()
#np.savetxt("E:/研究生学习/机器学习+紫苏/iuhg.dat",correlation_matrix)


# In[214]:


X = df


# In[216]:


y = data.iloc[:,data.columns == "Table"]


# In[134]:


#from imblearn.over_sampling import SMOTE
#smote = SMOTE(random_state=42)
#X_res, y_res = smote.fit_resample(X, y)
#print("过采样后类别分布:", np.bincount(y_res))


# In[218]:


plt.figure(figsize=(5, 4))
sns.histplot( y, kde=True)
plt.xlabel('lifetimetable')
plt.ylabel('count')
#plt.savefig("E:/研究生学习/机器学习+紫苏/picture/Labels Distribution Histograms.pdf")
plt.show()


# In[220]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,random_state=42)
print("X_train's shape is", X_train.shape,"; y_train's shape is", y_train.shape)
print("X_test's shape is", X_test.shape,"; y_test's shape is", y_test.shape)


# In[222]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler = scaler.fit(X_train)
X_train_stand = scaler.transform(X_train)
X_test_stand = scaler.transform(X_test)
X_stand = scaler.transform(X)


# In[224]:


X_train_stand=pd.DataFrame(X_train_stand,columns=X_train.columns)
X_test_stand=pd.DataFrame(X_test_stand,columns=X_test.columns)
X_stand=pd.DataFrame(X_stand,columns=X.columns)


# In[226]:


X_train_stand


# In[228]:


from sklearn.model_selection import KFold
kfold = KFold(n_splits = 5,shuffle = True,random_state = 42)
kfold3 = KFold(n_splits = 3,shuffle = True,random_state = 42)


# #XGBOOST

# In[231]:


import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform, randint


# 1. 数据准备和检查
print("数据检查...")
print(f"X_train_stand 类型: {type(X_train_stand)}, 形状: {X_train_stand.shape}")
print(f"y_train 类型: {type(y_train)}, 形状: {y_train.shape}")

# 确保y是一维数组（处理DataFrame或Series）
if isinstance(y_train, pd.DataFrame):
    y_train = y_train.values.ravel()  # DataFrame转NumPy数组再展平
elif isinstance(y_train, pd.Series):
    y_train = y_train.values  # Series转NumPy数组
# 如果是NumPy数组，确保是一维
elif isinstance(y_train, np.ndarray):
    y_train = y_train.ravel()

if isinstance(y_test, pd.DataFrame):
    y_test = y_test.values.ravel()
elif isinstance(y_test, pd.Series):
    y_test = y_test.values
elif isinstance(y_test, np.ndarray):
    y_test = y_test.ravel()

print(f"\n处理后 y_train 类型: {type(y_train)}, 形状: {y_train.shape}")
print(f"类别分布:\n{pd.Series(y_train).value_counts()}")

# 确保X是NumPy数组（如果原是DataFrame）
if isinstance(X_train_stand, pd.DataFrame):
    X_train_stand = X_train_stand.values
if isinstance(X_test_stand, pd.DataFrame):
    X_test_stand = X_test_stand.values

# 确保标签是0到n_classes-1的整数
le = LabelEncoder()
y_train_encoded = le.fit_transform(y_train)
y_test_encoded = le.transform(y_test)
n_classes = len(le.classes_)

print(f"\n编码后的类别: {le.classes_}")
print(f"编码后y_train的示例: {y_train_encoded[:10]}")

# 2. 转换为DMatrix格式
dtrain = xgb.DMatrix(X_train_stand, label=y_train_encoded)
dtest = xgb.DMatrix(X_test_stand, label=y_test_encoded)

# 3. 基础模型训练
def train_xgb(params, dtrain, dtest, early_stopping_rounds=50, num_boost_round=1000):
    evals_result = {}
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=num_boost_round,
        evals=[(dtrain, 'train'), (dtest, 'val')],
        early_stopping_rounds=early_stopping_rounds,
        evals_result=evals_result,
        verbose_eval=50
    )
    return model, evals_result

# 多分类参数
base_params = {
    'objective': 'multi:softmax',  # 多分类目标函数
    'num_class': n_classes,        # 类别数量
    'eval_metric': 'mlogloss',     # 多分类对数损失
    'seed': 42,
    'eta': 0.1,
    'max_depth': 6,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'base_score': 0.5
}

print("\n训练基础模型...")
base_model, base_evals = train_xgb(base_params, dtrain, dtest)
print(f"基础模型最佳迭代次数: {base_model.best_iteration}")
print(f"基础模型最佳验证分数: {base_model.best_score:.4f}")

# 其余代码保持不变...

# 其余代码保持不变...

# 4. 学习率调优
def tune_learning_rate(dtrain, dtest, lr_range=[0.01, 0.05, 0.1, 0.2, 0.3]):
    results = {}
    for eta in lr_range:
        params = base_params.copy()
        params['eta'] = eta
        model, evals = train_xgb(params, dtrain, dtest)
        results[eta] = {
            'best_iteration': model.best_iteration,
            'best_score': model.best_score,
            'evals': evals
        }
        print(f"Learning rate: {eta:.3f} | Rounds: {model.best_iteration} | Score: {model.best_score:.4f}")
    
    # 可视化结果
    plt.figure(figsize=(10, 6))
    for eta, res in results.items():
        plt.plot(res['evals']['val']['mlogloss'], 
                label=f"η={eta}, rounds={res['best_iteration']}, score={res['best_score']:.4f}")
    plt.axhline(y=min([res['best_score'] for res in results.values()]), color='gray', linestyle='--')
    plt.title('Validation Multi-Class LogLoss for Different Learning Rates')
    plt.xlabel('Boosting Rounds')
    plt.ylabel('Multi-Class LogLoss')
    plt.legend()
    plt.grid()
    plt.show()
    
    best_eta = min(results.items(), key=lambda x: x[1]['best_score'])[0]
    print(f"\n推荐学习率: {best_eta}")
    return best_eta

print("\n调优学习率...")
best_eta = tune_learning_rate(dtrain, dtest)
base_params['eta'] = best_eta

# 5. 树参数调优（使用sklearn API方便调参）
xgb_clf = xgb.XGBClassifier(
    objective='multi:softmax',
    num_class=n_classes,
    eval_metric='mlogloss',
    learning_rate=best_eta,
    seed=42,
    use_label_encoder=False
)

# 定义参数网格
param_grid = {
    'max_depth': [3, 4, 5, 6, 7, 8],
    'min_child_weight': [1, 2, 3, 4, 5],
    'gamma': [0, 0.1, 0.2, 0.3, 0.4],
    'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
    'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0]
}

# 使用随机搜索
random_search = RandomizedSearchCV(
    estimator=xgb_clf,
    param_distributions=param_grid,
    n_iter=50,
    scoring='neg_log_loss',
    cv=5,
    verbose=3,
    random_state=42,
    n_jobs=-1
)

print("\n执行随机搜索调优树参数...")
random_search.fit(X_train_stand, y_train_encoded)

# 输出最佳参数
print("\n最佳参数组合:")
best_params = random_search.best_params_
for k, v in best_params.items():
    print(f"{k}: {v}")
print(f"最佳验证分数: {-random_search.best_score_:.4f}")

# 更新参数
tuned_params = base_params.copy()
tuned_params.update(best_params)

# 6. 正则化参数调优
def tune_regularization(dtrain, dtest, params):
    alpha_values = [0, 0.001, 0.01, 0.1, 1]
    lambda_values = [0.001, 0.01, 0.1, 1, 10]
    
    best_score = float('inf')
    best_alpha = 0
    best_lambda = 1
    
    for alpha in alpha_values:
        for lambd in lambda_values:
            current_params = params.copy()
            current_params['alpha'] = alpha
            current_params['lambda'] = lambd
            
            model = xgb.train(
                current_params,
                dtrain,
                num_boost_round=1000,
                evals=[(dtrain, 'train'), (dtest, 'val')],
                early_stopping_rounds=50,
                verbose_eval=False
            )
            
            if model.best_score < best_score:
                best_score = model.best_score
                best_alpha = alpha
                best_lambda = lambd
                print(f"New best: alpha={alpha}, lambda={lambd}, score={best_score:.4f}")
    
    print(f"\n最佳正则化参数: alpha={best_alpha}, lambda={best_lambda}")
    return best_alpha, best_lambda

print("\n调优正则化参数...")
best_alpha, best_lambda = tune_regularization(dtrain, dtest, tuned_params)
tuned_params['alpha'] = best_alpha
tuned_params['lambda'] = best_lambda

# 7. 使用最佳参数训练最终模型
print("\n训练最终模型...")
final_model, final_evals = train_xgb(tuned_params, dtrain, dtest)

# 8. 模型评估
def evaluate_model(model, X_test_stand, y_test_encoded, le):
    dtest = xgb.DMatrix(X_test_stand)
    y_pred = model.predict(dtest).astype(int)
    y_true = y_test_encoded
    
    print("\n模型评估指标:")
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
    print(f"Precision (macro): {precision_score(y_true, y_pred, average='macro'):.4f}")
    print(f"Recall (macro): {recall_score(y_true, y_pred, average='macro'):.4f}")
    print(f"F1 Score (macro): {f1_score(y_true, y_pred, average='macro'):.4f}")
    
    # 混淆矩阵
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(le.classes_))
    plt.xticks(tick_marks, le.classes_, rotation=45)
    plt.yticks(tick_marks, le.classes_)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > cm.max()/2 else "black")
    plt.tight_layout()
    plt.show()

print("\n最终模型评估:")
evaluate_model(final_model, X_test_stand, y_test_encoded, le)

# 9. 特征重要性
plt.figure(figsize=(10, 6))
xgb.plot_importance(final_model, max_num_features=20)
plt.title('Feature Importance')
plt.show()



# In[156]:


superpa = []
for i in range(200):
    rfc = XGBClassifier(n_estimators=i+1,n_jobs=-1,random_state=42)
    rfc_s = cross_val_score(rfc,X_stand,y,cv=kfold3,scoring=make_scorer(accuracy_score)).mean()
    superpa.append(rfc_s)
print(max(superpa),superpa.index(max(superpa)))
plt.figure(figsize=[20,5])
plt.plot(range(1,201),superpa)
plt.show()


# In[249]:


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


# In[253]:


#alpha = 0.01  # 可以通过交叉验证选择最优值

xgb = XGBClassifier(n_estimators=3,learning_rate=0.3,subsample=1.0,min_child_weight=1,max_depth=8,gamma=0.01,
colsample_bytree=0.9,random_state=42,alpha = 0.01)
xgb.fit(X_train_stand, y_train)
xgbp = xgb.predict(X_test_stand)
xgbp_proba = xgb.predict_proba(X_test_stand)
#print(xgbp)
#print(y_test)
print(accuracy_score(y_test, xgbp))


# In[277]:


cm = confusion_matrix(y_test, xgbp)
plt.figure(figsize=(5, 4))
disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=['0','1','2'])
disp.plot(cmap='Blues', values_format='d')
plt.title('Confusion Matrix')
plt.savefig("C:/Users/lifukui/Desktop/DF/picture/fenlei/Confusion Matrix.pdf")

plt.show()


# <h2 style="font-size:24px;">RF</h2>

# In[257]:


ran= RandomForestClassifier(n_estimators=3,
                            criterion='gini',
                            max_depth =8,
                            random_state=42)
ran_accuracy = cross_val_score(ran,X_stand,y,cv=kfold,scoring=make_scorer(accuracy_score))
ran_precision = cross_val_score(ran,X_stand,y,cv=kfold,scoring=make_scorer(precision_score, average='macro'))
ran_recall = cross_val_score(ran,X_stand,y,cv=kfold,scoring=make_scorer(recall_score, average='macro'))
ran_f1 = cross_val_score(ran,X_stand,y,cv=kfold,scoring=make_scorer(f1_score, average='macro'))
mean_ran_accuracy = (sum(ran_accuracy)/5)
mean_ran_precision = (sum(ran_precision)/5)
mean_ran_recall = (sum(ran_recall)/5)
mean_ran_f1 = (sum(ran_f1)/5)
print('mean_ran_accuracy:',mean_ran_accuracy)
print('mean_ran_precision:',mean_ran_precision)
print('mean_ran_recall:',mean_ran_recall)
print('mean_ran_f1:',mean_ran_f1)
print('ran_accuracy:',ran_accuracy)
print('ran_precision:',ran_precision)
print('ran_recall:',ran_recall)   
print('ran_f1:',ran_f1)


# <h2 style="font-size:24px;">GauNB</h2>

# In[259]:


gau = GaussianNB()
gau_accuracy = cross_val_score(gau,X_stand,y,cv=kfold,scoring=make_scorer(accuracy_score))
gau_precision = cross_val_score(gau,X_stand,y,cv=kfold,scoring=make_scorer(precision_score, average='macro'))
gau_recall = cross_val_score(gau,X_stand,y,cv=kfold,scoring=make_scorer(recall_score, average='macro'))
gau_f1 = cross_val_score(gau,X_stand,y,cv=kfold,scoring=make_scorer(f1_score, average='macro'))
mean_gau_accuracy = (sum(gau_accuracy)/5)
mean_gau_precision = (sum(gau_precision)/5)
mean_gau_recall = (sum(gau_recall)/5)
mean_gau_f1 = (sum(gau_f1)/5)
print('mean_gau_accuracy:',mean_gau_accuracy)
print('mean_gau_precision:',mean_gau_precision)
print('mean_gau_recall:',mean_gau_recall)
print('mean_gau_f1:',mean_gau_f1)
print('gau_accuracy:',gau_accuracy)
print('gau_precision:',gau_precision)
print('gau_recall:',gau_recall)   
print('gau_f1:',gau_f1)


# <h2 style="font-size:24px;">SVM</h2>

# In[261]:


from sklearn import svm          
svm = svm.LinearSVC(C = 50,max_iter = 1000)
svm_accuracy = cross_val_score(svm,X_stand,y,cv=kfold,scoring=make_scorer(accuracy_score))
svm_precision = cross_val_score(svm,X_stand,y,cv=kfold,scoring=make_scorer(precision_score, average='macro'))
svm_recall = cross_val_score(svm,X_stand,y,cv=kfold,scoring=make_scorer(recall_score, average='macro'))
svm_f1 = cross_val_score(svm,X_stand,y,cv=kfold,scoring=make_scorer(f1_score, average='macro'))
mean_svm_accuracy = (sum(svm_accuracy)/5)
mean_svm_precision = (sum(svm_precision)/5)
mean_svm_recall = (sum(svm_recall)/5)
mean_svm_f1 = (sum(svm_f1)/5)
print('mean_svm_accuracy:',mean_svm_accuracy)
print('mean_svm_precision:',mean_svm_precision)
print('mean_svm_recall:',mean_svm_recall)
print('mean_svm_f1:',mean_svm_f1)
print('svm_accuracy:',svm_accuracy)
print('svm_precision:',svm_precision)
print('svm_recall:',svm_recall)   
print('svm_f1:',svm_f1)


# <h2 style="font-size:24px;">KNN</h2>

# In[263]:


knn = KNeighborsClassifier()
param_grid = {  
    'n_neighbors':[i for i in range(1,40)],
}
grid_search = GridSearchCV(knn, param_grid, cv=kfold3, scoring=make_scorer(accuracy_score),n_jobs = -1, verbose = 2)
grid_search.fit(X_train_stand, y_train)
print(grid_search.best_params_)


# In[265]:


knn = KNeighborsClassifier(n_neighbors=5)
param_grid = {  
    'weights': ['uniform', 'distance'], 
    'p': [1, 2]
}
grid_search = GridSearchCV(knn, param_grid, cv=kfold3, scoring=make_scorer(accuracy_score),n_jobs = -1, verbose = 2)
grid_search.fit(X_train_stand, y_train)
print(grid_search.best_params_)


# In[267]:


from sklearn.neighbors import KNeighborsClassifier
knn =KNeighborsClassifier(n_neighbors=5,weights ='uniform',p = 2 )
knn_accuracy = cross_val_score(knn,X_stand,y,cv=kfold,scoring=make_scorer(accuracy_score))
knn_precision = cross_val_score(knn,X_stand,y,cv=kfold,scoring=make_scorer(precision_score, average='macro'))
knn_recall = cross_val_score(knn,X_stand,y,cv=kfold,scoring=make_scorer(recall_score, average='macro'))
knn_f1 = cross_val_score(knn,X_stand,y,cv=kfold,scoring=make_scorer(f1_score, average='macro'))
mean_knn_accuracy = (sum(knn_accuracy)/5)
mean_knn_precision = (sum(knn_precision)/5)
mean_knn_recall = (sum(knn_recall)/5)
mean_knn_f1 = (sum(knn_f1)/5)
print('mean_knn_accuracy:',mean_knn_accuracy)
print('mean_knn_precision:',mean_knn_precision)
print('mean_knn_recall:',mean_knn_recall)
print('mean_knn_f1:',mean_knn_f1)
print('knn_accuracy:',knn_accuracy)
print('knn_precision:',knn_precision)
print('knn_recall:',knn_recall)   
print('knn_f1:',knn_f1)


# <h2 style="font-size:24px;">DTRClassifer</h2>

# In[269]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score

# 假设您已经有以下数据:
# X_stand - 标准化后的特征数据
# y - 目标变量
# kfold - 交叉验证分割器 (例如 KFold(n_splits=5, shuffle=True, random_state=42))

# 初始化决策树分类器
dtr = DecisionTreeClassifier(
    #criterion='gini',          # 使用基尼系数
    max_depth=20,              # 限制树深度
   # min_samples_split=5,       # 节点最少5个样本才分裂
    min_samples_leaf=1,        # 叶节点最少2个样本
   # max_features='sqrt',       # 考虑sqrt(n_features)个特征
    random_state=42)          # 固定随机种子
    #class_weight='balanced')    # 平衡类别权重


# 使用交叉验证评估模型性能
dtr_accuracy = cross_val_score(dtr, X_stand, y, cv=kfold, scoring=make_scorer(accuracy_score))
dtr_precision = cross_val_score(dtr, X_stand, y, cv=kfold, scoring=make_scorer(precision_score, average='macro'))
dtr_recall = cross_val_score(dtr, X_stand, y, cv=kfold, scoring=make_scorer(recall_score, average='macro'))
dtr_f1 = cross_val_score(dtr, X_stand, y, cv=kfold, scoring=make_scorer(f1_score, average='macro'))

# 计算平均指标
mean_dtr_accuracy = sum(dtr_accuracy) / len(dtr_accuracy)
mean_dtr_precision = sum(dtr_precision) / len(dtr_precision)
mean_dtr_recall = sum(dtr_recall) / len(dtr_recall)
mean_dtr_f1 = sum(dtr_f1) / len(dtr_f1)

# 打印结果
print('mean_dtr_accuracy:', mean_dtr_accuracy)
print('mean_dtr_precision:', mean_dtr_precision)
print('mean_dtr_recall:', mean_dtr_recall)
print('mean_dtr_f1:', mean_dtr_f1)
print('dtr_accuracy:', dtr_accuracy)
print('dtr_precision:', dtr_precision)
print('dtr_recall:', dtr_recall)   
print('dtr_f1:', dtr_f1)


# In[ ]:




