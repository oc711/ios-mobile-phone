#!/usr/bin/env python
# coding: utf-8

# 决策树的优点：
# 
# 一、           决策树易于理解和解释.人们在通过解释后都有能力去理解决策树所表达的意义。
# 二、           对于决策树，数据的准备往往是简单或者是不必要的.其他的技术往往要求先把数据一般化，比如去掉多余的或者空白的属性。
# 三、           能够同时处理数据型和常规型属性。其他的技术往往要求数据属性的单一。
# 四、           决策树是一个白盒模型。如果给定一个观察的模型，那么根据所产生的决策树很容易推出相应的逻辑表达式。
# 五、           易于通过静态测试来对模型进行评测。表示有可能测量该模型的可信度。
# 六、          在相对短的时间内能够对大型数据源做出可行且效果良好的结果。
# 七、           可以对有许多属性的数据集构造决策树。
# 八、           决策树可很好地扩展到大型数据库中，同时它的大小独立于数据库的大小。
# 
# 决策树的缺点：
# 
# 一、           对于那些各类别样本数量不一致的数据，在决策树当中,信息增益的结果偏向于那些具有更多数值的特征。
# 二、           决策树处理缺失数据时的困难。
# 三、           过度拟合问题的出现。
# 四、           忽略数据集中属性之间的相关性。

# In[1]:


# !pip install graphviz
#!pip install missingno


# In[1]:


# Necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import seaborn as sns
import graphviz
import os
import time
import missingno as msno
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_extraction import DictVectorizer
from sklearn.externals.six import StringIO 

from sklearn import tree
from sklearn.model_selection import train_test_split,StratifiedKFold,learning_curve,validation_curve,ShuffleSplit
from sklearn.metrics import precision_score,accuracy_score,recall_score,f1_score,log_loss,mean_squared_error
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
### Seaborn style
sns.set_style("whitegrid")


# # Data Preparation 数据准备

# In[2]:


data = pd.read_csv("./ios_mobile_pcaps_association.csv",sep="#",encoding = "utf-8")
# data = pd.read_csv("./ios_mobile_pcaps_probe.csv",sep="#",encoding = "utf-8")
# data = pd.read_csv("./ios_mobile_pcaps.csv",sep="#",encoding = "utf-8")


# In[3]:


data.shape


# In[4]:


data.file = data.file.astype('str')


# In[5]:


data['model'] = data['file'].apply(lambda x: x[ :10].strip())


# In[6]:


data['model'].unique()


# In[36]:


data['model'] = data['model'].apply(lambda x : "iPhone 1" if x=="iPhone 1 2" else x)
data['model'] = data['model'].apply(lambda x : "iPhone 4" if x=="iPhone 4 2" else x)
data['model'] = data['model'].apply(lambda x : "iPhone 5" if (x=="iPhone 5 2" or x=="iPhone 5 5") else x)
data['model'] = data['model'].apply(lambda x : "iPhone 5s" if (x=="iPhone 5s-") else x)
data['model'] = data['model'].apply(lambda x : "iPhone 6" if (x=="iPhone 6 2" or x=="iPhone 6 5") else x)
data['model'] = data['model'].apply(lambda x : "iPhone 7" if (x=="iPhone 7 2" or x=="iPhone 7 5"  or x=="iPhone 7 6"  or x=="iPhone 7 d"  or x=="iPhone 7 i" or x=="iPhone 7 n") else x)
data['model'] = data['model'].apply(lambda x : "iPhone 8" if (x=="iPhone 8 2" or x=="iPhone 8 5")else x)
data['model'] = data['model'].apply(lambda x : "iPhone X" if (x=="iPhone X 2" or x=="iPhone X 5" or x=="iPhone X n" or x=="iPhone X i" or x=="iPhone X b") else x)
data['model'] = data['model'].apply(lambda x : "iPhone XS MAX" if x=="iphone XS" else x)


# In[37]:


data['model'].unique() #共有18个型号的iPhone手机作为分类预测目标


# In[38]:


data['target'] = data['model'].replace([   'iPhone 1', 'iPhone 11', 'iPhone 3GS', 'iPhone 4', 'iPhone 4s',
                                           'iPhone 5', 'iPhone 5c', 'iPhone 5s', 'iPhone 6', 'iPhone 6+',
                                           'iPhone 6s', 'iPhone 6s+', 'iPhone 7', 'iPhone 7+', 'iPhone 8',
                                           'iPhone 8+', 'iPhone SE', 'iPhone X', 'iPhone XR', 'iPhone XS'],
                                             range(1, 21))


# In[39]:


train = pd.DataFrame()


# In[41]:


#缺失值可视化
msno.matrix(data, labels=True)


# In[42]:


data.Arrival_Time = data.Arrival_Time.astype('str')
data['year'] = data['Arrival_Time'].apply(lambda x: x[x.find(',')+1:x.find(',')+6].strip())
data['month']= data['Arrival_Time'].apply(lambda x: x[:4].strip())
data['day']= data['Arrival_Time'].apply(lambda x: x[4:x.find(',')].strip())


# In[43]:


data.mcs = data.mcs.astype('str')
data['mcs'] = data['mcs'].apply(lambda x: x[:x.find('spatial')].strip() if x!="nan" else 0 )


# In[44]:


train = data.fillna(0)


# In[51]:


features = [x for x in train.columns if x not in ['msg_type','file','Arrival_Time','ht_short_gi','ext_supp_rates','ht_capabilities',
                                                  'supp_rates','model', 'target', 'Data_rate', 'Channel','oui_type','oui',
                                                  'Duration', 'Signal_noise_ratio','day','max_rx_ampdu_length','mcs']]
print(features)


# In[52]:


#字符串特征转数值特征 
# train['msg_type'] = pd.factorize(data['msg_type'])[0].astype(np.uint16)
train['oui_dict'] = pd.factorize(data['oui_dict'])[0].astype(np.uint16)
#train['ht_short_gi'] = pd.factorize(data['ht_short_gi'])[0].astype(np.uint16)
train['ssid_param'] = pd.factorize(data['ssid_param'])[0].astype(np.uint16)
# train['access_network_type'] = pd.factorize(data['access_network_type'])[0].astype(np.uint16)
# train['ext_supp_rates'] = pd.factorize(data['ext_supp_rates'])[0].astype(np.uint16)
# train['ht_capabilities'] = pd.factorize(data['ht_capabilities'])[0].astype(np.uint16)
train['supp_rates'] = pd.factorize(data['supp_rates'])[0].astype(np.uint16)
train['mcs'] = pd.factorize(data['mcs'])[0].astype(np.uint16)
train['year'] = pd.factorize(data['year'])[0].astype(np.uint16)
train['month'] = pd.factorize(data['month'])[0].astype(np.uint16)
train['day'] = pd.factorize(data['day'])[0].astype(np.uint16)


# In[53]:


y = train['target']
X = train[features]


# In[20]:


# 归一化，将属性缩放到一个指定的最大和最小值（通常是1-0）之间
# 使用这种方法的目的包括：
# 1、对于方差非常小的属性可以增强其稳定性。
# 2、维持稀疏矩阵中为0的条目。
# from sklearn.preprocessing import MinMaxScaler
# min_max_scaler = preprocessing.MinMaxScaler()
# X = min_max_scaler.transform(X)


# In[26]:


#标准化（Z-Score） 公式为：(X-mean)/std 计算时对每个属性/每列分别进行。
# from sklearn.preprocessing import scale 
# scale(X, axis=0, with_mean=True, with_std=True, copy=True)  


# # Data Visualization 原始数据可视化分析

# In[20]:


#Histogram
plt.figure(figsize=(21,7))
ax = sns.countplot(x="model", data=train)


# # Correlation Analysis Visualization 关联性可视化分析

# In[190]:


# sns.set(style="white", palette="muted", color_codes=True)
# Set up the matplotlib figure
# f, axes = plt.subplots(2, 2, figsize=(11, 11))
# sns.despine(left=True)

# 感兴趣的特征与目标向量的关联性分析热力图
sns.jointplot(x = -X['year'],y = y,color="y", kind = "kde")
sns.jointplot(x = -X['mpdu_density'],y = y, kind = "kde",color="m")

# plt.setp(axes, yticks=[])
# plt.tight_layout()


# In[54]:


def corr_analyse(df,title='Correlation Analyse'):
    df_corr = df.corr()
    sns.set(font_scale=0.7)
    
    f, ax = plt.subplots(figsize=(11,11),dpi=100)
    sns.heatmap(df_corr,annot=True,linecolor="white", linewidths=0.1,fmt ='.2f',vmin=-1,vmax=1,square=True,cmap='RdBu',ax=ax)
    
    from matplotlib import rcParams 
    rcParams['axes.titlepad'] = 20 # Space between the title and graph 
    ax.set_title(title, fontsize=20, fontweight="bold") 
    
    plt.ylabel('Class Label',fontsize=18)
    plt.xlabel('Class Label',fontsize=18)
    label_y = ax.get_yticklabels()
    plt.setp(label_y, rotation=360, horizontalalignment='right')
    ax.tick_params(axis='y',labelsize=10)
    label_x = ax.get_xticklabels()
    plt.setp(label_x, rotation=45, horizontalalignment='right')
    ax.tick_params(axis='x',labelsize=10)
    # show 
    plt.savefig('./Decision Tree Correlation Analyse.png')
    plt.show()


# In[55]:


corr_analyse(X)


# # Algorithme (mannuel): DT 手写决策树

# In[193]:


# 对y的各种可能的取值出现的个数进行计数.。其他函数利用该函数来计算数据集和的混杂程度
def uniquecounts(rows):
    res = {}
    for row in rows:
        #计数结果存在最后一列
        r = row[len(row)-1]
        if r not in res:res[r]=0
        res[r] += 1
    return res


# In[194]:


# Entropy
def entropy(rows):
    from math import log
    log2 = lambda x:log(x)/log(2)
    res = uniquecounts(rows)
    #calculate entropy
    e = 0.0
    for r in res.keys():
        p = float(res[r])/len(rows)
        e = e - p*log2(p)
    return e


# ID3算法
# 
# ID3算法的核心是在决策树的各个结点上应用信息增益准则进行特征选择。具体做法是：
# 
# 从根节点开始，对结点计算所有可能特征的信息增益，选择信息增益最大的特征作为结点的特征，并由该特征的不同取值构建子节点；
# 对子节点递归地调用以上方法，构建决策树；
# 直到所有特征的信息增益均很小或者没有特征可选时为止。

# C4.5算法
# 
# C4.5算法与ID3算法的区别主要在于它在生产决策树的过程中，使用信息增益比来进行特征选择。

# CART算法
# 
# 分类与回归树（classification and regression tree,CART）与C4.5算法一样，由ID3算法演化而来。CART假设决策树是一个二叉树，它通过递归地二分每个特征，将特征空间划分为有限个单元，并在这些单元上确定预测的概率分布。
# 
# CART算法中，对于回归树，采用的是平方误差最小化准则；对于分类树，采用基尼指数最小化准则。
# 
# 

# In[195]:


#定义节点的属性
class decisionnode:
    def __init__(self,col = -1,value = None, results = None, tb = None,fb = None):
        self.col = col   # col是待检验的判断条件所对应的列索引值
        self.value = value # value对应于为了使结果为True，当前列必须匹配的值
        self.results = results #保存的是针对当前分支的结果，它是一个字典
        self.tb = tb ## desision node,对应于结果为true时，树上相对于当前节点的子树上的节点
        self.fb = fb ## desision node,对应于结果为true时，树上相对于当前节点的子树上的节点


# In[196]:


# 基尼不纯度
# 随机放置的数据项出现于错误分类中的概率
def gini_impurity(rows):
    total = len(rows)
    counts = uniquecounts(rows)
    imp = 0
    for k in counts.keys():
        p = float(counts[k])/total
        imp += p*(1-p)
    return imp


# In[197]:


#在某一列上对数据集进行拆分。可应用于数值型或因子型变量
def divideset(rows,column,value):
    #定义一个函数，判断当前数据行属于第一组还是第二组
    split_function = None
    if isinstance(value,int) or isinstance(value,float):
        split_function = lambda row:row[column] >= value
    else:
        split_function = lambda row:row[column]==value
    # 将数据集拆分成两个集合，并返回
    set1 = [row for row in rows if split_function(row)]
    set2 = [row for row in rows if not split_function(row)]
    return(set1,set2)


# In[198]:


# 以递归方式构造树

def buildtree(rows,scoref = entropy):
    if len(rows)==0 : return decisionnode()
    current_score = scoref(rows)
    
    # 定义一些变量以记录最佳拆分条件
    best_gain = 0.0
    best_criteria = None
    best_sets = None
    
    column_count = len(rows[0]) - 1
    for col in range(0,column_count):
        #在当前列中生成一个由不同值构成的序列
        column_values = {}
        for row in rows:
            column_values[row[col]] = 1 # 初始化
        #根据这一列中的每个值，尝试对数据集进行拆分
        for value in column_values.keys():
            (set1,set2) = divideset(rows,col,value)
            
            # 信息增益
            p = float(len(set1))/len(rows)
            gain = current_score - p*scoref(set1) - (1-p)*scoref(set2)
            if gain>best_gain and len(set1)>0 and len(set2)>0:
                best_gain = gain
                best_criteria = (col,value)
                best_sets = (set1,set2)
                
    #创建子分支
    if best_gain>0:
        trueBranch = buildtree(best_sets[0])  #递归调用
        falseBranch = buildtree(best_sets[1])
        return decisionnode(col = best_criteria[0],value = best_criteria[1],
                            tb = trueBranch,fb = falseBranch)
    else:
        return decisionnode(results = uniquecounts(rows))


# In[70]:


# 决策树的显示
def printtree(tree,indent = ''):
    # 是否是叶节点
    if tree.results!=None:
        print str(tree.results)
    else:
        # 打印判断条件
        print str(tree.col)+":"+str(tree.value)+"? "
        #打印分支
        print indent+"T->",
        printtree(tree.tb,indent+" ")
        print indent+"F->",
        printtree(tree.fb,indent+" ")


# In[ ]:


# 对新的观测数据进行分类

def classify(observation,tree):
    if tree.results!= None:
        return tree.results
    else:
        v = observation[tree.col]
        branch = None
        if isinstance(v,int) or isinstance(v,float):
            if v>= tree.value: branch = tree.tb
            else: branch = tree.fb
        else:
            if v==tree.value : branch = tree.tb
            else: branch = tree.fb
        return classify(observation,branch)


# # Pruning 剪枝

# 如果是使用sklearn库的决策树生成的话，剪枝方法有限，仅仅只能改变其中参数来进行剪枝。
# DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=7,
#             max_features=None, max_leaf_nodes=None,
#             min_impurity_split=1e-07, min_samples_leaf=10,
#             min_samples_split=20, min_weight_fraction_leaf=0.0,
#             presort=False, random_state=None, splitter='best')
# 
# 

# In[57]:


# 树最大深度 与 算法精确度关系分析
test = []
for i in range(20):
    clf=tree.DecisionTreeClassifier(max_depth=i+1,criterion="entropy",random_state=1234)
    clf=clf.fit(X_train,y_train)
    score = clf.score(X_test, y_test)
    test.append(score)
plt.plot(range(1,21),test,color='blue',label='max_depth')
plt.title('"score / max_depth "')
plt.xlabel('max_depth')
plt.ylabel('score')


# In[72]:


# 叶子节点最小的样本权重 与 算法精确度 关系分析
test = []
for i in range(10):
    clf=tree.DecisionTreeClassifier(max_depth=8,
                                    criterion="entropy",
                                    splitter='best',
                                    min_weight_fraction_leaf = i*0.05,
                                    random_state=1234)
    clf=clf.fit(X_train,y_train)
    score = clf.score(X_test, y_test)
    test.append(score)
    
plt.plot(range(1,11),test,color='blue',label='max_depth')
plt.title('score / min_weight_fraction_leaf ')
plt.xlabel('i')
plt.ylabel('score')


# In[71]:


# 最小子叶分割样本数 与 算法精确度 关系分析
test = []
for i in range(10):
    clf=tree.DecisionTreeClassifier(max_depth=8,
                                    criterion="entropy",
                                    splitter='best',
                                    min_samples_split=10+i*10,
                                    random_state=123)
    clf=clf.fit(X_train,y_train)
    score = clf.score(X_test, y_test)
    test.append(score)
    
plt.plot(range(1,11),test,color='blue',label='max_depth')
plt.title('score / min_weight_fraction_leaf ')
plt.xlabel('i')
plt.ylabel('score')


# #  Trainning 模型训练

# In[56]:


#划分训练集和测试
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)


# In[83]:


dtc =tree.DecisionTreeClassifier(
                                 criterion='entropy',
                                 splitter='best',
                                 random_state=432,
                                 min_samples_split=30,
                                 max_depth=8
                                )


# In[84]:


clf = dtc.fit(X_train, y_train)
y_predict = clf.predict(X_test)


# In[81]:


#绘制学习曲线
"""
Generate 3 plots: the test and training learning curve, the training
samples vs fit times curve, the fit times vs score curve.

Parameters
----------
estimator : object type that implements the "fit" and "predict" methods
    An object of that type which is cloned for each validation.
title : string
    Title for the chart.
X : array-like, shape (n_samples, n_features)
    Training vector, where n_samples is the number of samples and
    n_features is the number of features.
y : array-like, shape (n_samples) or (n_samples, n_features), optional
    Target relative to X for classification or regression;
    None for unsupervised learning.
ylim : tuple, shape (ymin, ymax), optional
    Defines minimum and maximum yvalues plotted.
cv : int, cross-validation generator or an iterable, optional
    Determines the cross-validation splitting strategy.
    Possible inputs for cv are:
      - None, to use the default 5-fold cross-validation,
      - integer, to specify the number of folds.
      - :term:`CV splitter`,
      - An iterable yielding (train, test) splits as arrays of indices.
    For integer/None inputs, if ``y`` is binary or multiclass,
    :class:`StratifiedKFold` used. If the estimator is not a classifier
    or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.
    Refer :ref:`User Guide <cross_validation>` for the various
    cross-validators that can be used here.
train_sizes : array-like, shape (n_ticks,), dtype float or int
    Relative or absolute numbers of training examples that will be used to
    generate the learning curve. If the dtype is float, it is regarded as a
    fraction of the maximum size of the training set (that is determined
    by the selected validation method), i.e. it has to be within (0, 1].
    Otherwise it is interpreted as absolute sizes of the training sets.
    Note that for classification the number of samples usually have to
    be big enough to contain at least one sample from each class.
    (default: np.linspace(0.1, 1.0, 5))//用来指定训练集占交叉验证cv训练集中的百分比
"""
def plot_learning_curve(estimator, title, X, y, cv=5, train_sizes=np.linspace(0.1, 1, 20)):
        
    plt.figure(figsize=(8, 8))
    plt.title(title,fontsize=14)
    
    plt.xlabel("Number of training samples",fontsize=14)
    plt.ylabel("Accuracy score",fontsize=14)
    
    #设置坐标轴刻度
    plt.yticks(np.linspace(0, 1, 20))

    train_sizes, train_scores, test_scores= learning_curve(estimator, X, y, cv=cv,train_sizes=train_sizes)
    print(train_sizes)
    plt.xticks(train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)


    # Plot learning curve
    plt.grid(True)
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,train_scores_mean + train_scores_std, alpha=0.1,color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,test_scores_mean + test_scores_std, alpha=0.1,color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",label="Cross-validation score")
    plt.legend(loc="best",fontsize=14)
    # plt.savefig('./validation_curve.png', dpi=300)
    return plt


# In[85]:


title = "Learning Curves (Decision Tree)"
cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=42) #定义交叉验证CV
plot_learning_curve(clf, title, X, y)
plt.show()


# In[68]:


#根据模型学习曲线分析：
# 当训练集和测试集的误差收敛但却很高时，为高偏差。
# 左上角的偏差很高，训练集和验证集的准确率都很低，很可能是欠拟合。
# 我们可以增加模型参数，比如，构建更多的特征，减小正则项。
# 此时通过增加数据量是不起作用的。

# 当训练集和测试集的误差之间有大的差距时，为高方差。
# 当训练集的准确率比其他独立数据集上的测试结果的准确率要高时，一般都是过拟合。
# 右上角方差很高，训练集和验证集的准确率相差太多，应该是过拟合。即我们目前的情况
# 我们可以增大训练集，降低模型复杂度，增大正则项，或者通过特征选择减少特征数。

# 理想情况是是找到偏差和方差都很小的情况，即收敛且误差较小。


# In[86]:


#测试集精确率
print('accuracy_score: ', "%.2f"%(100*accuracy_score(y_test, y_predict)),'%') #预测准确率输出


# In[87]:


#测试集加权平均精确率
print('precision_score: ',"%.2f"%(100* precision_score(y_test, y_predict, average='weighted')),'%')
#测试集加权平均召回率
print('recall_score:    ',"%.2f"%(100*recall_score(y_test, y_predict, average='weighted')),'%')
#测试集加权平均f1-score
print('f1_score:        ',"%.2f"%(100*f1_score(y_test, y_predict, average='weighted')),'%')


# # Cross Validation 交叉验证

# In[88]:


nr_fold=5 #5折交叉验证
random_state=144
folds = StratifiedKFold(n_splits=nr_fold, 
                        shuffle=True, 
                        random_state=random_state)


# In[101]:


params = {
          'criterion':'entropy',
          'splitter':'best',
          'random_state':random_state,
          'min_samples_split':10,
          'max_depth':9
         }


# In[102]:


#缓存变量
oof_preds  = np.zeros((len(X_train)))

test_preds = []
importances = pd.DataFrame()#重要性分析
clfs = []
start_time=time.time()

for fold_, (trn_, val_) in enumerate(folds.split(X_train,y_train)):

        ## model poour les 54 classes
        trn_x, trn_y = X_train[features].iloc[trn_], y_train.iloc[trn_]
        val_x, val_y = X_train[features].iloc[val_], y_train.iloc[val_]
    
        clf = tree.DecisionTreeClassifier(**params)
        clf.fit(trn_x, trn_y)

        oof_preds[val_] = clf.predict(val_x)
        test_preds.append(clf.predict(X_test) )  
        
        imp_df = pd.DataFrame()
        imp_df['feature'] = features
        imp_df['gain'] = clf.feature_importances_
        imp_df['fold'] = fold_ + 1
        
        importances = pd.concat([importances, imp_df], axis=0, sort=False)
        clfs.append(clf)
        
        fold_accuracy = accuracy_score(val_y, oof_preds[val_])
        fold_logloss = log_loss(val_y,clf.predict_proba(val_x))

        print('Decision Tree Model no {}-fold accuracy_score is {}, LogLoss score is {},'.format(fold_ + 1,fold_accuracy,fold_logloss))
        end_time = time.time()
        print("it takes %.3f seconds to perform cross validation" % (end_time - start_time))


# #  Importances 变量重要性分析

# In[103]:


#Features's importances
def print_importances(importances_,title="Importance Analyse"):
    
    mean_gain = importances_[['gain', 'feature']].groupby('feature').mean()
    importances_['mean_gain'] = importances_['feature'].map(mean_gain['gain'])
    
    plt.figure(figsize=(13, 11))
    plt.title(title,fontsize=20,fontweight="bold")
    plt.tick_params(labelsize=14)
    plt.xlabel('Features',fontsize=20)
    plt.ylabel('Gain',fontsize=20)
    data_imp = importances_.sort_values('mean_gain', ascending=False)
    sns.barplot(x='gain', y='feature', data=data_imp[:100])
    
    plt.tight_layout()
    plt.savefig('./importances_DT.png')
    plt.show()


# In[104]:


print_importances(importances)


# In[130]:


# 前十位变量含义解析：
#     mpdu_density     : Message Protocol Data Unit 信息协议数据单元的密度
#     mac_timestamp    : MAC捕捉的时间戳
#     power_cap_max    : 设备最大功率
#     frame_length     : 帧长
#     power_cap_min    : 设备最小功率
#     dt_captured      : Time delta from previous captured frame 距离上一个捕获帧的时间间隔
#     Noise_level_dBm  : 噪声等级
#         如果噪声水平过高，则可能会降低无线信号强度，并降低性能。
#         噪声电平以-dBm格式（0至-100）测量。 这是相对于一毫瓦的测量功率的功率比，以分贝（dB）为单位。
#         该值越接近0，则噪声水平越高。
#         负值表示较少的背景噪声。 例如，-96dBm的噪声水平低于-20dBm。
#     Signal_Stregth    : 信号强度
#         信号强度是无线客户端接收到的无线信号功率电平。
#         强大的信号强度可实现更可靠的连接和更高的速度。
#         信号强度以-dBm格式（0至-100）表示。 这是相对于一毫瓦的测量功率的功率比，以分贝（dB）为单位。
#         值越接近0，信号越强。 例如，-41dBm比-61dBm更好。
#     Frequency : 报文发送的频率，这个变量与Channel（频道号）以及Data_rate(传输速率Mb/s)有较大的共线性，推测为频道号有固定的频率和速率，三者保留一个。
#     Signal_Noise_Ratio: 信噪比
#         信噪比（SNR）是信号强度和噪声水平之间的功率比。该值表示为+ dBm值。
#         通常，您应具有至少+ 25dBm的信噪比。 低于+ 25dBm的值会导致较差的性能和速度。
#         例如：如果您具有-41dBm的信号强度和-50dBm的噪声水平，这将导致+ 9dBm的较差的信噪比。


# In[105]:


#trainning acc 训练集精确度
print('Accuracy: ',"%.2f"%(100*accuracy_score(y_train,oof_preds)),'%')


# # Vote （交叉验证结果）投票

# In[106]:


# k = 5
vote_df = pd.DataFrame()
vote_df['cv0'] = test_preds[0] 
vote_df['cv1'] = test_preds[1] 
vote_df['cv2'] = test_preds[2] 
vote_df['cv3'] = test_preds[3] 
vote_df['cv4'] = test_preds[4]


# In[107]:


vote_df 


# In[108]:


result = []
for idx, i in vote_df.iterrows(): 
    l = list(i) 
    l_vote_counts = [l.count(x) for x in l] 
    m = max(l_vote_counts) 
    result.append(l[l_vote_counts.index(m)])


# In[109]:


vote_df['final_vote'] = result


# In[110]:


vote_df.shape


# In[111]:


vote_df.head()


# In[112]:


#test acc 测试集精确度
print('Accuracy: ',"%.2f"%(100*accuracy_score(y_test,vote_df['final_vote'])),'%')

#测试集精确度达到：99.43%


# # Graph Visualization （决策树）模型可视化

# In[113]:


dot_data = tree.export_graphviz(
                                 clfs[3],out_file=None,
                                 feature_names = features,
                                 class_names = np.unique(data['model']),
                                 filled = True,
                                 special_characters = True
                                )


# In[114]:


graph = graphviz.Source(dot_data)


# In[115]:


import os
os.environ["Path"] += os.pathsep + 'C:\\Program Files (x86)\\graphviz-2.38\\release\\bin'


# In[116]:


graph.render('Decision Tree of IOS Mobile Phone Model', view=True)


# # Evaluation 模型评估

# In[117]:


#Matrix of Confusion 混淆矩阵
from sklearn.metrics import confusion_matrix
def plot_confusion_matrix(cm,title='Confusion Matrix'):
    
    plt.figure(figsize=(11, 11), dpi=100)
    # 在混淆矩阵中每格的概率值
    ind_array = np.arange(len(classes))
    x, y = np.meshgrid(ind_array, ind_array)
    for x_val, y_val in zip(x.flatten(), y.flatten()):
        c = cm[y_val][x_val]
        if c > 0.001:
            plt.text(x_val, y_val, "%d" % (c,), color='red', fontsize=9, va='center', ha='center')
    
    plt.imshow(cm, cmap=plt.cm.BuGn)
    plt.title(title,fontsize=20,fontweight="bold")
    plt.colorbar()
    xlocations = np.array(range(len(classes)))
    plt.yticks(xlocations, classes)
    plt.xticks(xlocations, classes,rotation=45,horizontalalignment='right')
    plt.ylabel('Actual label',fontsize=14)
    plt.xlabel('Predict label',fontsize=14)
    
    # offset the tick
    tick_marks = np.array(range(len(classes)))
    plt.gca().set_xticks(tick_marks, minor=True)
    plt.gca().set_yticks(tick_marks, minor=True)
    plt.grid(True, which='minor', color='red',linestyle='--')
    
    # show confusion matrix
    plt.savefig('./Decision_Tree_confusion_matrix.png')
    plt.show()


# In[118]:


# 获取混淆矩阵
classes = np.unique(data['model'])
cm = confusion_matrix(y_test,vote_df['final_vote'])
plot_confusion_matrix(cm)


# In[119]:


#ClassificationReport 分类报告
from sklearn.metrics import classification_report 
sampleClassificationReport = classification_report(y_test,vote_df['final_vote'],target_names=classes)
print(sampleClassificationReport)


# In[ ]:





# # Validation 泛化性验证

# In[120]:


validation_data = pd.read_csv("./ios_mobile_validation_association.csv",sep="#",encoding = "utf-8")
# validation_data = pd.read_csv("./ios_mobile_validation_probe.csv",sep="#",encoding = "utf-8")
# validation_data = pd.read_csv("./ios_mobile_validation.csv",sep="#",encoding = "utf-8")


# In[121]:


validation_data.shape


# In[122]:


validation_data['model'] = validation_data['file'].apply(lambda x: x[ :10].strip())


# In[123]:


validation_data['model'].unique() 


# In[124]:


validation_data['model'] = validation_data['model'].apply(lambda x : "iPhone 5" if  x=="iPhone 5 5" else x)
validation_data['model'] = validation_data['model'].apply(lambda x : "iPhone 5s" if  x=="iPhone 5s-" else x)
validation_data['model'] = validation_data['model'].apply(lambda x : "iPhone 7" if (x=="iPhone 7 2" or x=="iPhone 7 d") else x)
validation_data['model'] = validation_data['model'].apply(lambda x : "iPhone 8" if (x=="iPhone 8 d" or x=="iPhone 8 5")else x)
validation_data['model'] = validation_data['model'].apply(lambda x : "iPhone X" if (x=="iPhone X 2" or x=="iPhone X 5" or x=="iPhone X b" or x=="iPhone X d") else x)
validation_data['model'] = validation_data['model'].apply(lambda x : "iPhone XR" if x=="iPhone XR_" else x)


# In[125]:


validation_data['model'].unique() 


# In[126]:


validation_data['target'] = validation_data['model'].replace([   'iPhone 1', 'iPhone 11', 'iPhone 3GS', 'iPhone 4', 'iPhone 4s',
                                           'iPhone 5', 'iPhone 5c', 'iPhone 5s', 'iPhone 6', 'iPhone 6+',
                                           'iPhone 6s', 'iPhone 6s+', 'iPhone 7', 'iPhone 7+', 'iPhone 8',
                                           'iPhone 8+', 'iPhone SE', 'iPhone X', 'iPhone XR', 'iPhone XS'],
                                             range(1, 21))


# In[127]:


validation_data[['index','target','model']]


# In[128]:


plt.figure(figsize=(15,5))
ax = sns.countplot(x="model", data=validation_data)


# In[129]:


validation_X = pd.DataFrame()
validation_y_origin = validation_data[['index','target']] 


# In[130]:


validation_data.Arrival_Time = validation_data.Arrival_Time.astype('str')
validation_data['year'] = validation_data['Arrival_Time'].apply(lambda x: x[x.find(',')+1:x.find(',')+6].strip())
validation_data['month']= validation_data['Arrival_Time'].apply(lambda x: x[:4].strip())
validation_data['day']= validation_data['Arrival_Time'].apply(lambda x: x[4:x.find(',')].strip())


# In[131]:


validation_data.mcs = validation_data.mcs.astype('str')
validation_data['mcs'] = validation_data['mcs'].apply(lambda x: x[:x.find('spatial')].strip() if x!="nan" else 0 )


# In[132]:


validation_train = pd.DataFrame()
validation_train = validation_data.fillna(0)


# In[133]:


# validation_train['msg_type'] = pd.factorize(validation_data['msg_type'])[0].astype(np.uint16)
validation_train['oui_dict'] = pd.factorize(validation_data['oui_dict'])[0].astype(np.uint16)
#validation_train['ht_short_gi'] = pd.factorize(validation_data['ht_short_gi'])[0].astype(np.uint16)
validation_train['ssid_param'] = pd.factorize(validation_data['ssid_param'])[0].astype(np.uint16)
validation_train['access_network_type'] = pd.factorize(validation_data['access_network_type'])[0].astype(np.uint16)
# train['ext_supp_rates'] = pd.factorize(data['ext_supp_rates'])[0].astype(np.uint16)
# train['ht_capabilities'] = pd.factorize(data['ht_capabilities'])[0].astype(np.uint16)
# train['supp_rates'] = pd.factorize(data['supp_rates'])[0].astype(np.uint16)
validation_train['mcs'] = pd.factorize(validation_data['mcs'])[0].astype(np.uint16)
validation_train['year'] = pd.factorize(validation_data['year'])[0].astype(np.uint16)
validation_train['month'] = pd.factorize(validation_data['month'])[0].astype(np.uint16)
validation_train['day'] = pd.factorize(validation_data['day'])[0].astype(np.uint16)


# In[134]:


validation_X  = validation_train[features]


# In[135]:


validation_X.shape


# In[86]:


#Prediction


# In[146]:


validation_preds = clfs[3].predict(validation_X)


# In[147]:


validation_preds 


# In[148]:


validation_y_preds = pd.DataFrame()
validation_y_preds['index'] = range(1,len(validation_preds)+1)
validation_y_preds['target'] = validation_preds 


# In[149]:


validation_y_preds['model'] = validation_y_preds['target'].replace(range(1, 21),[   'iPhone 1', 'iPhone 11', 'iPhone 3GS', 'iPhone 4', 'iPhone 4s',
                                           'iPhone 5', 'iPhone 5c', 'iPhone 5s', 'iPhone 6', 'iPhone 6+',
                                           'iPhone 6s', 'iPhone 6s+', 'iPhone 7', 'iPhone 7+', 'iPhone 8',
                                           'iPhone 8+', 'iPhone SE', 'iPhone X', 'iPhone XR', 'iPhone XS'])


# In[150]:


validation_y_preds


# In[153]:


#validation acc 外部验证集精确度
# print('Accuracy: ',accuracy_score(validation_y_origin['target'],validation_y_preds['target'])
print('Accuracy: ',"%.2f"%(100*sum(validation_y_origin['target']==validation_y_preds['target'])/len(validation_y_origin)),'%')

# 通过分离关联帧和probe帧数据，外部验证集精确度提升至52.63%，
# 但由于数据总量减少（几乎减半），模型的泛化性能仍旧不能够支撑进一步作业。


# In[154]:


validation_y_preds.to_csv("./ios_mobile_validation_preds.csv",sep="#",encoding = "utf-8")


# In[165]:


#总结: 
 #     基于关联帧的IOS mobile细分类算法研究初步告一段落，目前的结果并不能支撑该功能接入业务线，
#     主要原因是数据总量过小（7K）不足以支撑机器学习模型的训练，需要等待后续相关数据的扩充。

