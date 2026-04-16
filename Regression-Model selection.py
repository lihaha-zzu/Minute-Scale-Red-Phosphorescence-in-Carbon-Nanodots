#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


# In[2]:


data = pd.read_excel(r'C:\Users\lifukui\Desktop\DF\picture\huigui-2\DF-NEW-去异2-寿命两段-三分类-第二类.xlsx')
#data = pd.read_csv('C:\Users\lifukui\Desktop\DF\DF.xlsx')


# In[3]:


data


# In[4]:


df= data.drop('Wavelength', axis = 1)
df= df.drop('Lifetime', axis = 1)

df= df.drop('origin', axis = 1)
#df= df.drop('Water', axis = 1)
#df= df.drop('alcohol', axis = 1)

df


# In[5]:


correlation_matrix = df.corr()
cmap = sns.diverging_palette(220,8,as_cmap = True)
plt.figure(figsize=(5, 4))
sns.heatmap(correlation_matrix, annot=True, cmap=cmap, vmin = -1,vmax= 1,
            cbar_kws={'orientation':'vertical','ticks':[-1,-0.5,0,0.5,1]},
            linewidths=.5)
#plt.savefig("E:/研究生学习/机器学习+紫苏/picture/pierxun.pdf")
plt.show()
#np.savetxt("E:/研究生学习/机器学习+紫苏/iuhg.dat",correlation_matrix)


# In[6]:


X = df


# In[7]:


y = data.iloc[:,data.columns == "Lifetime"]


# In[8]:


plt.figure(figsize=(5, 4))
sns.histplot(y, kde=True)
plt.xlabel('lifetimetable')
plt.ylabel('count')
#plt.savefig("E:/研究生学习/机器学习+紫苏/picture/Labels Distribution Histograms.pdf")
plt.show()


# In[9]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,random_state=42)
print("X_train's shape is", X_train.shape,"; y_train's shape is", y_train.shape)
print("X_test's shape is", X_test.shape,"; y_test's shape is", y_test.shape)


# In[10]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler = scaler.fit(X_train)
X_train_stand = scaler.transform(X_train)
X_test_stand = scaler.transform(X_test)
X_stand = scaler.transform(X)


# In[11]:


X_train_stand=pd.DataFrame(X_train_stand,columns=X_train.columns)
X_test_stand=pd.DataFrame(X_test_stand,columns=X_test.columns)
X_stand=pd.DataFrame(X_stand,columns=X.columns)


# In[12]:


from sklearn.model_selection import KFold
kfold = KFold(n_splits = 5,shuffle = True,random_state = 42)
kfold3 = KFold(n_splits = 3,shuffle = True,random_state = 42)


# #XGBOOST

# In[14]:


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas import DataFrame, read_csv
from IPython.display import display_html
import seaborn as sns 
import warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error #均方误差
from sklearn.metrics import mean_absolute_error #平方绝对误差
from sklearn.metrics import r2_score#R square


# In[15]:


def prediction_vs_ground_truth_fig(y_train, y_train_hat, y_test, y_test_hat):
    from sklearn import metrics
    fontsize = 12
    plt.figure(figsize=(3.5,3))
    plt.style.use('default')
    plt.rc('xtick', labelsize=fontsize)
    plt.rc('ytick', labelsize=fontsize)
    plt.rcParams['font.family']="Arial"
    a = plt.scatter(y_train, y_train_hat, s=25,c='#40c0c0')
    plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'k:', lw=1.5)
    plt.xlabel('Observation', fontsize=fontsize)
    plt.ylabel('Prediction', fontsize=fontsize)
    #plt.xticks([99,99.2,99.4,99.6,99.8,100])
    #plt.yticks([80,82,84,86,88,90,92,94,96,98,100])
    plt.xticks([50,100,150,200,250])
    plt.yticks([50,100,150,200,250])
    plt.tick_params(direction='in')
    #plt.xlim([99,101]) 
    #plt.ylim([80,110])
    plt.xlim([50,250]) 
    plt.ylim([50,250])
    plt.title(('Train RMSE: {:.2e}'.format(np.sqrt(metrics.mean_squared_error(y_train, y_train_hat))),\
               'Test RMSE: {:.2e}'.format(np.sqrt(metrics.mean_squared_error(y_test, y_test_hat)))), fontsize=fontsize)
    b = plt.scatter(y_test, y_test_hat, s=25,edgecolor='#a53361', facecolor='none')
    plt.legend((a,b),('Train','Test'),fontsize=fontsize,handletextpad=0.1,borderpad=0.1)
    plt.rcParams['font.family']="Arial"
    plt.tight_layout()
    #plt.savefig('show.pdf', dpi = 1200)
    plt.show()


# <h2 style="font-size:24px;">XGBoost</h2>

# In[17]:


from xgboost.sklearn import XGBRegressor



from pprint import pprint

rf_regressor = XGBRegressor(
                            learning_rate =0.5,
                            random_state = 42,
                            max_depth=8, 
                            min_child_weight= 1,
                            n_estimators=30,
                            reg_alpha=6,
                       
                            
                            reg_lambda=0.01
   # 'gamma': [0,0.1, 0.3,0.7, 1],
    #reg_alpha= 0.01,
  
    #'max_depth': [3, 5, 15, 50],
    #'min_child_weight': [1, 3, 5, 7,10]
                           )
# Fit to the training set
rf_regressor.fit(X_train_stand, y_train)
# Perform predictions on both training and test sets
y_train_hat = rf_regressor.predict(X_train_stand)
y_test_hat = rf_regressor.predict(X_test_stand)

# Visualize the results
prediction_vs_ground_truth_fig(y_train, y_train_hat, y_test, y_test_hat)
rf_regressor_cross = cross_val_score(rf_regressor,X_train_stand,y_train,cv=kfold,scoring='neg_mean_absolute_error')
sum_rf_regressor_cross = (sum(rf_regressor_cross)/5)
print(sum_rf_regressor_cross)
print(rf_regressor_cross)
print(mean_squared_error(y_train, y_train_hat),mean_squared_error(y_test, y_test_hat))
print(mean_absolute_error(y_train, y_train_hat),mean_absolute_error(y_test, y_test_hat))
print(r2_score(y_train, y_train_hat),r2_score(y_test, y_test_hat))
rf_regressor.feature_importances_
#plt.savefig("E:/研究生学习/机器学习+紫苏/picture/XGBhuigui.pdf")


# In[42]:


from xgboost.sklearn import XGBRegressor



from pprint import pprint

rf_regressor = XGBRegressor(
                            colsample_bytree=0.6, 
                            gamma=0,
                            learning_rate=0.1, 
                            n_estimators=150,
                            max_depth=4,
                            reg_alpha=1,
                            reg_lambda= 0.1, 
                            subsample=0.8,
                            random_state = 42
                           )
# Fit to the training set
rf_regressor.fit(X_train_stand, y_train)
# Perform predictions on both training and test sets
y_train_hat = rf_regressor.predict(X_train_stand)
y_test_hat = rf_regressor.predict(X_test_stand)

# Visualize the results
prediction_vs_ground_truth_fig(y_train, y_train_hat, y_test, y_test_hat)
rf_regressor_cross = cross_val_score(rf_regressor,X_train_stand,y_train,cv=kfold,scoring='neg_mean_absolute_error')
sum_rf_regressor_cross = (sum(rf_regressor_cross)/5)
print(sum_rf_regressor_cross)
print(rf_regressor_cross)
print(mean_squared_error(y_train, y_train_hat),mean_squared_error(y_test, y_test_hat))
print(mean_absolute_error(y_train, y_train_hat),mean_absolute_error(y_test, y_test_hat))
print(r2_score(y_train, y_train_hat),r2_score(y_test, y_test_hat))
rf_regressor.feature_importances_
#plt.savefig("E:/研究生学习/机器学习+紫苏/picture/XGBhuigui.pdf")


# In[ ]:


import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, mean_squared_error
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

# 1. 准备数据
# 假设X是特征，y是目标变量
# X, y = load_your_data()  # 替换为您的数据加载代码
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. 创建XGBoost回归器
xgb_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)

# 3. 定义参数网格
param_grid = {
    'n_estimators':[10,20,50,100,150] ,# 树的数量
    'max_depth': [3, 5, 7],          # 树的最大深度
    'learning_rate': [0.01, 0.1, 0.2],  # 学习率
    'subsample': [0.6, 0.8, 1.0],    # 样本采样比例
    'colsample_bytree': [0.6, 0.8, 1.0],  # 特征采样比例
    'gamma': [0, 0.1, 0.2],          # 节点分裂所需的最小损失减少
    'reg_alpha': [0, 0.1, 1,5,10],        # L1正则化项
    'reg_lambda': [0.1, 1, 5,10]       # L2正则化项
}

# 4. 定义评分指标（回归问题常用负均方误差）
scorer = make_scorer(mean_squared_error, greater_is_better=False)

# 5. 创建网格搜索对象
grid_search = GridSearchCV(
    estimator=xgb_model,
    param_grid=param_grid,
    scoring=scorer,
    cv=5,  # 5折交叉验证
    n_jobs=-1,  # 使用所有CPU核心
    verbose=2  # 显示详细日志
)

# 6. 执行网格搜索
grid_search.fit(X_train, y_train)

# 7. 输出最佳参数和分数
print("最佳参数组合: ", grid_search.best_params_)
print("最佳模型分数(负MSE): ", grid_search.best_score_)

# 8. 使用最佳模型进行预测
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"测试集MSE: {mse:.4f}")

# 9. 保存最佳模型
# best_model.save_model('best_xgboost_model.json')


# In[20]:


from xgboost.sklearn import XGBRegressor
superpa = []
for i in range(200):
    rfc = XGBRegressor(n_estimators=i+1,n_jobs=-1)
    rfc_s = cross_val_score(rfc,X_stand,y,cv=kfold,scoring= 'neg_mean_absolute_error', n_jobs= -1).mean()
    superpa.append(rfc_s)
print(max(superpa),superpa.index(max(superpa)))
plt.figure(figsize=[20,5])
plt.plot(range(1,201),superpa)
plt.show()


# <h2 style="font-size:24px;">DTR</h2>

# In[54]:


# 导入所需的库
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import numpy as np



# 创建决策树回归器实例
rf_regressor = DecisionTreeRegressor(max_depth=10,random_state=42)

# 拟合模型
rf_regressor.fit(X_train_stand, y_train)
# Perform predictions on both training and test sets
y_train_hat = rf_regressor.predict(X_train_stand)
y_test_hat = rf_regressor.predict(X_test_stand)

# Visualize the results
prediction_vs_ground_truth_fig(y_train, y_train_hat, y_test, y_test_hat)
rf_regressor_cross = cross_val_score(rf_regressor,X_train_stand,y_train,cv=kfold,scoring='neg_mean_absolute_error')
sum_rf_regressor_cross = (sum(rf_regressor_cross)/5)
print(sum_rf_regressor_cross)
print(rf_regressor_cross)
print(mean_squared_error(y_train, y_train_hat),mean_squared_error(y_test, y_test_hat))
print(mean_absolute_error(y_train, y_train_hat),mean_absolute_error(y_test, y_test_hat))
print(r2_score(y_train, y_train_hat),r2_score(y_test, y_test_hat))
rf_regressor.feature_importances_


# <h2 style="font-size:24px;">KNN</h2>

# In[97]:


# 导入必要的库
import numpy as np
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import KNeighborsRegressor

from sklearn.metrics import mean_squared_error


# 创建K近邻回归模型实例
knn = KNeighborsRegressor()

# 定义要搜索的超参数网格
param_grid = {
    'n_neighbors': [1,2,3,4,5],
    #np.arange(1, 10),  # 尝试从1到30的K值
    'weights': ['uniform', 'distance'],  # 权重方案，uniform为所有点相同权重，distance为距离加权
    'p': [1, 2]  # 距离度量的幂参数，1表示曼哈顿距离，2表示欧几里得距离
}

# 创建GridSearchCV实例
grid_search = GridSearchCV(knn, param_grid, cv=5, scoring='neg_mean_squared_error')

# 在训练数据上运行网格搜索
grid_search.fit(X_train_stand, y_train)

# 打印最佳的超参数组合
print("Best parameters found: ", grid_search.best_params_)

# 使用最佳模型进行预测
best_knn = grid_search.best_estimator_
y_pred = best_knn.predict(X_test)

# 计算并打印均方误差
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error on test set: ", mse)


# In[100]:


# 创建K近邻回归模型实例
knn = KNeighborsRegressor(n_neighbors=3,
                          p=1,
                       
                         )  # 这里设置K为5

# 拟合模型
knn.fit(X_train, y_train)

# 预测
y_test_hat = knn.predict(X_test)

# Visualize the results
prediction_vs_ground_truth_fig(y_train, y_train_hat, y_test, y_test_hat)
rf_regressor_cross = cross_val_score(rf_regressor,X_stand,y,cv=5,scoring='neg_mean_absolute_error')
sum_rf_regressor_cross = (sum(rf_regressor_cross)/5)
print(sum_rf_regressor_cross)
print(rf_regressor_cross)
print(mean_squared_error(y_train, y_train_hat),mean_squared_error(y_test, y_test_hat))
print(mean_absolute_error(y_train, y_train_hat),mean_absolute_error(y_test, y_test_hat))
print(r2_score(y_train, y_train_hat),r2_score(y_test, y_test_hat))


# <h2 style="font-size:24px;">RF</h2>

# In[110]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
rf_regressor = RandomForestRegressor(n_estimators=20,
                                     max_depth =6,
                                     #min_samples_leaf = 1,
                                     #min_samples_split=10,
                                     random_state=42)

# Fit to the training set
rf_regressor.fit(X_train_stand, y_train)
# Perform predictions on both training and test sets
y_train_hat = rf_regressor.predict(X_train_stand)
y_test_hat = rf_regressor.predict(X_test_stand)

# Visualize the results
prediction_vs_ground_truth_fig(y_train, y_train_hat, y_test, y_test_hat)
rf_regressor_cross = cross_val_score(rf_regressor,X_stand,y,cv=kfold,scoring='neg_mean_absolute_error')
sum_rf_regressor_cross = (sum(rf_regressor_cross)/5)
print(sum_rf_regressor_cross)
print(rf_regressor_cross)
print(mean_squared_error(y_train, y_train_hat),mean_squared_error(y_test, y_test_hat))
print(mean_absolute_error(y_train, y_train_hat),mean_absolute_error(y_test, y_test_hat))
print(r2_score(y_train, y_train_hat),r2_score(y_test, y_test_hat))
rf_regressor.feature_importances_


# <h2 style="font-size:24px;">GaussianProcessRegressor</h2>

# In[86]:


from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.model_selection import GridSearchCV


# 定义高斯过程回归模型
kernel = C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-2, 1e2))
gp = GaussianProcessRegressor(kernel=kernel, alpha=0.1)

# 定义参数网格
param_grid = {
    'kernel__k1__constant_value': [0.1, 1.0, 10.0],
    'kernel__k2__length_scale': [0.1, 1.0, 10.0],
    'alpha': [0.01, 0.1, 1.0],
    'n_restarts_optimizer': [0, 1, 2],
}

# 创建GridSearchCV对象
grid_search = GridSearchCV(gp, param_grid, cv=5, scoring='neg_mean_squared_error')

# 拟合模型
grid_search.fit(X_train_stand, y_train)

# 打印最佳参数
print("Best parameters:", grid_search.best_params_)


# In[94]:


from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF
kernel = ConstantKernel(constant_value=10) * RBF(length_scale=0.1)

rf_regressor = GaussianProcessRegressor(
                alpha= 0.1, 
                
                n_restarts_optimizer= 2
                
                           )

# Fit to the training set
rf_regressor.fit(X_train_stand, y_train)
# Perform predictions on both training and test sets
y_train_hat = rf_regressor.predict(X_train_stand)
y_test_hat = rf_regressor.predict(X_test_stand)

# Visualize the results
prediction_vs_ground_truth_fig(y_train, y_train_hat, y_test, y_test_hat)
rf_regressor_cross = cross_val_score(rf_regressor,X_stand,y,cv=5,scoring='neg_mean_absolute_error')
sum_rf_regressor_cross = (sum(rf_regressor_cross)/5)
print(sum_rf_regressor_cross)
print(rf_regressor_cross)
print(mean_squared_error(y_train, y_train_hat),mean_squared_error(y_test, y_test_hat))
print(mean_absolute_error(y_train, y_train_hat),mean_absolute_error(y_test, y_test_hat))
print(r2_score(y_train, y_train_hat),r2_score(y_test, y_test_hat))


# <h2 style="font-size:24px;">GBDT</h2>

# In[83]:


from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV


rf_regressor = GradientBoostingRegressor(
                                     n_estimators=150,
                                     max_depth =12,
                                     #min_samples_leaf = 1,
                                     #min_samples_split=2,
                                     random_state=42
)

# Fit to the training set
rf_regressor.fit(X_train_stand, y_train)
# Perform predictions on both training and test sets
y_train_hat = rf_regressor.predict(X_train_stand)
y_test_hat = rf_regressor.predict(X_test_stand)

# Visualize the results
prediction_vs_ground_truth_fig(y_train, y_train_hat, y_test, y_test_hat)
rf_regressor_cross = cross_val_score(rf_regressor,X_stand,y,cv=5,scoring='neg_mean_absolute_error')
sum_rf_regressor_cross = (sum(rf_regressor_cross)/5)
print(sum_rf_regressor_cross)
print(rf_regressor_cross)
print(mean_squared_error(y_train, y_train_hat),mean_squared_error(y_test, y_test_hat))
print(mean_absolute_error(y_train, y_train_hat),mean_absolute_error(y_test, y_test_hat))
print(r2_score(y_train, y_train_hat),r2_score(y_test, y_test_hat))
rf_regressor.feature_importances_


# In[ ]:




