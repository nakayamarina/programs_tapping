
# coding: utf-8

# # SVMによる相関の高いボクセルを用いた学習と性能評価（多変量解析）
# ----
#
# 引数：raw_rest.csv/raw_tapping.csvがあるディレクトリまでのパス
#
# ----
#
# 入力：raw_rest.csv/raw_tapping.csv
#
# ----
#
# 出力：ACCURACY[loo or k_cv]_CORmultivariate_SVM.csv ボクセルごとの識別性能評価結果一覧
#
# ----
#
# 相関の高いボクセルを用いて多変量解析を行う．
# k分割交差検証法により1グループをテストデータの，k-1グループを教師データとし，SVMを用いて学習，精度評価．
# ベクトル：各ボクセルにおけるある時刻のZ-score（ボクセル数ベクトル）

# In[1]:

print('############ ML_SVM_CORvariate_kCV.py program excution ############')


# In[2]:

import numpy as np
import pandas as pd
import sys
from sklearn import cross_validation
from sklearn import svm
from sklearn.model_selection import train_test_split


# In[86]:

args = sys.argv
PATH = args[1]

# jupyter notebookのときはここで指定
# PATH = '../State-2fe_MaskBrodmann/20181029tm/mb/COR10vox_RawData/'

# 検証手法
kCV = 10

# 検証手法
col_name = str(kCV) + 'CV'


# ## SVM_kCV関数
# 引数としてデータをX，ラベルをyで受け取る．
# 交差検証法の一つk分割交差検証法で識別精度評価を行う．

# In[87]:

def SVM_kCV(X, y):

    # 線形SVMのインスタンスを生成
    model = svm.SVC(kernel = 'linear', C = 1)

    # k分割し，1グループをテストデータ，残りグループを教師データにして評価
    # すべてのグループに対して行う
    # 評価結果（識別率）を格納
    CVscore = cross_validation.cross_val_score(model, X, y, cv = kCV)

    # 評価結果（識別率）の平均を求める
    result = CVscore.mean()

    # パーセントに直す
    result = round(result * 100, 1)

    print('k = ' + str(kCV) + '：' + str(CVscore))

    return result


# # main関数

# In[88]:

if __name__ == '__main__':

    # 読み込みたいファイルのパス
    PATH_rest = PATH + 'raw_rest.csv'
    PATH_tapping = PATH + 'raw_tapping.csv'

    # csvファイル読み込み
    # headerは設定せず，転置後にset_index()する（header = 0にすると列名が変えられる）
    rest = pd.read_csv(PATH_rest, header = 0, index_col = 0)

    tapping = pd.read_csv(PATH_tapping, header = 0, index_col = 0)


# In[89]:

# 各タスクのデータを結合
all_data = pd.concat([rest, tapping], axis = 0)

# ベクトル化
X = all_data.as_matrix()


# In[90]:

# ラベル作成 rest = 0, tapping = 1
label_rest = np.zeros(len(rest))
label_tapping = np.ones(len(tapping))

y = np.r_[label_rest, label_tapping]


# In[91]:

# 学習と評価
result = SVM_kCV(X, y)
print(result)


# In[92]:

# データフレーム化する際のインデックス名作成
index_name = str(rest.shape[1]) + 'voxels'

# データフレーム化
result_df = pd.DataFrame({col_name:[result]}, index = [index_name])


# In[93]:

# csv書き出し
PATH_RESULT = PATH + 'ACCURACY[' + str(kCV) + 'CV]_CORmultivariate' + '_SVM.csv'
result_df.to_csv(PATH_RESULT)


# In[ ]:
