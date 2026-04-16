#!/usr/bin/env python
# coding: utf-8

# In[3]:


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


# In[5]:


data = pd.read_excel(r'C:\Users\Fukuili\Desktop\DF-NEW-去异2-寿命两段-三分类-第二类.xlsx')
#data = pd.read_csv('C:\Users\lifukui\Desktop\DF\DF.xlsx')


# In[6]:


data


# In[7]:


df= data.drop('Wavelength', axis = 1)
df= df.drop('Lifetime', axis = 1)

df= df.drop('origin', axis = 1)
#df= df.drop('Water', axis = 1)
#df= df.drop('alcohol', axis = 1)

df


# In[8]:


correlation_matrix = df.corr()
cmap = sns.diverging_palette(220,8,as_cmap = True)
plt.figure(figsize=(5, 4))
sns.heatmap(correlation_matrix, annot=True, cmap=cmap, vmin = -1,vmax= 1,
            cbar_kws={'orientation':'vertical','ticks':[-1,-0.5,0,0.5,1]},
            linewidths=.5)
#plt.savefig("C:/Users/lifukui/Desktop/DF/picture/huigui-2/piercun.pdf")
plt.show()


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

# In[13]:


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


# In[14]:


def prediction_vs_ground_truth_fig(y_train, y_train_hat, y_test, y_test_hat):
    from sklearn import metrics
    fontsize = 12
    plt.figure(figsize=(3.5,3))
    plt.style.use('default')
    plt.rc('xtick', labelsize=fontsize)
    plt.rc('ytick', labelsize=fontsize)
    plt.rcParams['font.family']="Arial"
    a = plt.scatter(y_train, y_train_hat, s=25,c='#1e88e5')
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
    b = plt.scatter(y_test, y_test_hat, s=25,edgecolor='#ff0d57', facecolor='none')
    plt.legend((a,b),('Train','Test'),fontsize=fontsize,handletextpad=0.1,borderpad=0.1)
    plt.rcParams['font.family']="Arial"
    plt.tight_layout()
    #plt.savefig('show.pdf', dpi = 1200)
    plt.show()


# <h2 style="font-size:24px;">XGBoost</h2>

# In[15]:


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


# In[ ]:





# In[16]:


import shap
shap_values = shap.TreeExplainer(rf_regressor).shap_values(X_train_stand)
shap_interaction_values = shap.TreeExplainer(rf_regressor).shap_interaction_values(X_train_stand)


# In[17]:


plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values, X_train_stand, 
                 plot_type="bar",
                 show=False)  # 关键：禁用自动显示

plt.tight_layout()

# 先保存再显示
#save_path = "C:/Users/lifukui/Desktop/DF/picture/huigui-2/shap全局重要性.pdf"
#plt.savefig(save_path, dpi=1200, bbox_inches='tight')
plt.show()  # 如需查看再显示


# In[18]:


plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values, X_train_stand,show=False)

plt.tight_layout()

#save_path = "C:/Users/lifukui/Desktop/DF/picture/huigui-2/all shap value fengwo.pdf"
#plt.savefig(save_path, dpi=1200, bbox_inches='tight')
plt.show()
#fig.savefig('E:/研究生学习/机器学习+紫苏/picture/all shap value fengwo.pdf',dpi=1200, bbox_inches='tight')


# In[ ]:





# In[36]:


plt.figure(figsize=(10, 8))
shap.dependence_plot(0, shap_values, X_train_stand,show=False)

plt.tight_layout()

#save_path = "C:/Users/lifukui/Desktop/DF/picture/huigui-2/all shap value fengwo RHB.pdf"
#plt.savefig(save_path, dpi=1200, bbox_inches='tight')
plt.show()
#fig.savefig('E:/研究生学习/机器学习+紫苏/picture/all shap value fengwo.pdf',dpi=1200, bbox_inches='tight')


# In[20]:


plt.figure(figsize=(10, 8))
shap.dependence_plot(2, shap_values, X_train_stand,show=False)

plt.tight_layout()

#save_path = "C:/Users/lifukui/Desktop/DF/picture/huigui-2/all shap value fengwo p-ration.pdf"
#plt.savefig(save_path, dpi=1200, bbox_inches='tight')
plt.show()
#fig.savefig('E:/研究生学习/机器学习+紫苏/picture/all shap value fengwo.pdf',dpi=1200, bbox_inches='tight')


# In[21]:


plt.figure(figsize=(10, 8))
shap.dependence_plot(0, shap_values, X_train_stand,show=False)

plt.tight_layout()

#save_path = "C:/Users/lifukui/Desktop/DF/picture/huigui-2/all shap value fengwo T.pdf"
#plt.savefig(save_path, dpi=1200, bbox_inches='tight')
plt.show()
#fig.savefig('E:/研究生学习/机器学习+紫苏/picture/all shap value fengwo.pdf',dpi=1200, bbox_inches='tight')


# In[22]:


plt.figure(figsize=(10, 8))
shap.dependence_plot(3, shap_values, X_train_stand,show=False)

plt.tight_layout()

#save_path = "C:/Users/lifukui/Desktop/DF/picture/huigui-2/all shap value fengwo time.pdf"
#plt.savefig(save_path, dpi=1200, bbox_inches='tight')
plt.show()
#fig.savefig('E:/研究生学习/机器学习+紫苏/picture/all shap value fengwo.pdf',dpi=1200, bbox_inches='tight')


# In[26]:


plt.figure(figsize=(10, 8))
shap.dependence_plot((0, 3), shap_interaction_values, X_train_stand, show=False)
plt.tight_layout()
#save_path = "C:/Users/lifukui/Desktop/DF/picture/huigui-2/all shap value fengwot T&p-ratio.pdf"
#plt.savefig(save_path, dpi=1200, bbox_inches='tight')
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[22]:


shap.initjs()
explainer = shap.TreeExplainer(rf_regressor)
shap_values = explainer.shap_values(X_train_stand)

shap.force_plot(explainer.expected_value, shap_values, X_train_stand)
#plt.savefig("C:/Users/lifukui/Desktop/All sample feature influence diagram.pdf")


# In[26]:


shap.initjs()  # 必须初始化JS

# 直接显示交互式力图（不需要matplotlib参数）
shap.force_plot(explainer.expected_value, 
                shap_values[61,:], 
                X_train_stand.iloc[61,:],
                link='logit')
#plt.savefig("C:/Users/lifukui/Desktop/DF/picture/huigui-2/Single sample feature influence diagram-10.pdf")


# In[23]:


import pandas as pd

# 临时取消显示限制
with pd.option_context('display.max_rows', None,  # 显示所有行
                     'display.max_columns', None,  # 显示所有列
                     'display.width', None,        # 自动调整宽度
                     'display.max_colwidth', None): # 显示完整列内容

    print(X_train)


# In[ ]:





# In[ ]:





# In[78]:


from IPython.display import Image, display
import matplotlib.pyplot as plt

# 方法一：使用matplotlib版本保存（推荐）
plt.figure()
shap.force_plot(
    explainer.expected_value,
    shap_values[30,:],
    X_train_stand.iloc[30,:],
    matplotlib=True,
    show=False
)
plt.tight_layout()
plt.savefig("C:/Users/lifukui/Desktop/DF/picture/huigui-2/Single_sample_feature_influence_diagram-30.pdf",
            format='pdf',
            bbox_inches='tight',
            dpi=300)
plt.close()


# In[77]:


from IPython.display import Image, display
import matplotlib.pyplot as plt

# 方法一：使用matplotlib版本保存（推荐）
plt.figure()
shap.force_plot(
    explainer.expected_value,
    shap_values[10,:],
    X_train_stand.iloc[10,:],
    matplotlib=True,
    show=False
)
plt.tight_layout()
plt.savefig("C:/Users/lifukui/Desktop/DF/picture/huigui-2/Single_sample_feature_influence_diagram-10.pdf",
            format='pdf',
            bbox_inches='tight',
            dpi=300)
plt.close()


# In[ ]:





# In[80]:


from IPython.display import Image, display
import matplotlib.pyplot as plt

# 方法一：使用matplotlib版本保存（推荐）
plt.figure()
shap.force_plot(
    explainer.expected_value,
    shap_values[60,:],
    X_train_stand.iloc[60,:],
    matplotlib=True,
    show=False
)
plt.tight_layout()
plt.savefig("C:/Users/lifukui/Desktop/DF/picture/huigui-2/Single_sample_feature_influence_diagram-60.pdf",
            format='pdf',
            bbox_inches='tight',
            dpi=300)
plt.close()


# In[81]:


from IPython.display import Image, display
import matplotlib.pyplot as plt

# 方法一：使用matplotlib版本保存（推荐）
plt.figure()
shap.force_plot(
    explainer.expected_value,
    shap_values[20,:],
    X_train_stand.iloc[20,:],
    matplotlib=True,
    show=False
)
plt.tight_layout()
plt.savefig("C:/Users/lifukui/Desktop/DF/picture/huigui-2/Single_sample_feature_influence_diagram-20.pdf",
            format='pdf',
            bbox_inches='tight',
            dpi=300)
plt.close()


# In[ ]:





# In[23]:


import itertools
import pandas as pd

parameter_values = {
    'Temperature': [165,170,175,180,185,190,195,200],
    'RHB': [500,550,600,700,750,800,850,900,950,1000,1050,1100,1150,1200,1250,1300,1350,1400],
    'P-Ratio': [0.4,0.5,0.55,0.6,0.7,0.75,0.8,0.85,0.9,0.95,1.0,1.1,1.05,1.15,1.2],
    'Time': [6,7,8,9,10,11,12,13,14],
    'Water': [0,1,2,3,4,5,6],
    'alcohol': [0,50,100]
}

parameter_names = list(parameter_values.keys())
parameter_list = list(parameter_values.values())

cartesian_product = list(itertools.product(*parameter_list))

prediction = pd.DataFrame(cartesian_product, columns=parameter_names)


# In[24]:


X_prediction = scaler.transform(prediction)


# In[25]:


xgb_prediction = rf_regressor.predict(X_prediction)


# In[26]:


prediction['prediction'] =xgb_prediction


# In[27]:


prediction


# In[44]:


# 将整个DataFrame写入Excel
prediction.to_excel('C:/Users/lifukui/Desktop/DF/picture/huigui-2/prediction_data.xlsx', 
                   index=False,  # 不写入行索引
                   sheet_name='Prediction')  # 指定工作表名


# In[ ]:





# In[35]:


import plotly.express as px

# 抽样1万行避免过载
df_sample = prediction.sample(n=5000, random_state=42)

fig = px.parallel_coordinates(
    df_sample,
    color="prediction",
    dimensions=['Temperature', 'RHB', 'P-Ratio', 'Time', 'Water', 'alcohol', 'prediction'],
    color_continuous_scale=px.colors.sequential.Viridis,
    title="Parallel Coordinates Plot with Prediction Coloring"
)
fig.update_layout(height=600)
fig.show()



# In[48]:


import matplotlib
matplotlib.use('Agg')  # 关键修改
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

# 数据准备
df_sample = prediction.sample(n=50000, random_state=42)  # 减少数据量测试
cols = ['Temperature', 'RHB', 'P-Ratio', 'Time', 'Water', 'alcohol','prediction']
n_cols = len(cols)

# 创建图形
fig = plt.figure(figsize=(16, 7))
ax = fig.add_subplot(111)  # 显式创建子图

# 数据标准化
df_norm = df_sample[cols].apply(lambda x: (x - x.min()) / (x.max() - x.min()))

# 颜色映射
norm = Normalize(vmin=df_sample['prediction'].min(), 
                vmax=df_sample['prediction'].max())
cmap = plt.get_cmap('RdBu_r')

# 绘制线条（进一步优化）
for i in range(len(df_norm)):
    if i % 250 == 0:  # 稀疏绘制测试
        values = df_norm.iloc[i].values
        color = cmap(norm(df_sample['prediction'].iloc[i]))
        ax.plot(range(n_cols), values, 
                color=color, 
                alpha=0.5, 
                linewidth=0.5,
                solid_capstyle='round')

# 坐标轴设置
ax.set_xticks(range(n_cols))
ax.set_xticklabels(cols, rotation=45, ha='right', fontsize=10)
ax.set_xlim(-0.5, n_cols-0.5)
ax.grid(True, alpha=0.2, linestyle=':')

# 颜色条
cmap = plt.get_cmap('RdBu_r')
sm = ScalarMappable(norm=norm, cmap=cmap)
cbar = fig.colorbar(sm, ax=ax, pad=0.02, aspect=40)
cbar.set_label('Prediction Score', rotation=270, labelpad=20)

# 标题和布局
ax.set_title("Parallel Coordinates (Simplified)", fontsize=14, pad=20)
plt.tight_layout()

# 保存PDF（关键修改）
output_path = 'C:/Users/lifukui/Desktop/DF/picture/huigui-2/笛卡尔.pdf'
fig.savefig(output_path,
            format='pdf',
            bbox_inches='tight',
            facecolor='w',
            edgecolor='none',
            dpi=4000,
            metadata={'Creator': '', 'Producer': ''})  # 清空元数据

plt.close(fig)  # 必须关闭图形

print(f"PDF已保存到: {output_path}")
print(f"文件大小: {os.path.getsize(output_path)/1024:.1f} KB")


# In[ ]:




