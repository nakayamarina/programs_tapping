
# coding: utf-8

# # SVMによるボクセルごとの学習と識別性能評価（時系列解析）

# ---
#
# 引数：div_rest.csv/div_tapping.csvがあるディレクトリまでのパス
#
# ---
#
# 入力：div_rest.csv/div_tapping.csv
#
# ---
#
# 出力：ACCURACY[loo or k_cv]_voxels_SVM.csv　識別性能評価結果一覧
#
# ---
#
# ボクセルごとに区切られた時系列データをSVMを用いて学習し，交差検証法（k-分割交差検証，leave-one-out交差検証）を用いて識別性能評価を行う．
# ベクトル：各ボクセルの区切られた時系列データ
#

# In[1]:

print('############# ML_SVM_voxels.py program excution ##############')


# In[2]:

import numpy as np
import pandas as pd
import sys

from sklearn import cross_validation
from sklearn import svm
from sklearn.model_selection import train_test_split


# コマンドライン引数でraw_rest.csv/raw_tapping.csvがあるディレクトリまでのパスを取得

# In[92]:

args = sys.argv
PATH = args[1]

# jupyter notebookのときはここで指定
# PATH = '../State-2fe_Active/20181029rn/64ch/8divData/'

# 検証手法
col_name = 'leave-one-out'

# ボクセル数
voxels = 7


# ## SVM_LOO関数
# 引数として教師データをX，ラベルをyで受け取る．
# 交差検証法の一つleave-one-out交差検証で識別精度評価を行う．
#
# * (1個をテストデータ，残りを教師データにして学習・評価) * すべてのデータ個
# * 得られたすべてのデータ個の評価結果（識別率）の平均を求めてパーセントに直す
# * 評価結果（識別率）をmain関数に返す

# In[93]:

def SVM_LOO(X, y):

    LOOscore = np.zeros(len(X))

    # 1個をテストデータ，残りを教師データにして学習・評価
    # すべてのデータに対して行う
    for i in range(len(X)):

        print('------ ' + str(i + 1) + ' / ' + str(len(X)) + '回 -----')

        # テストデータ
        X_test = X[i].reshape(1, -1)
        y_test = y[i].reshape(1, -1)

        # テストデータとして使用するデータを除いた教師データを作成
        X_train = np.delete(X, i, 0)
        y_train = np.delete(y, i, 0)

        # 線形SVMのインスタンスを生成
        model = svm.SVC(kernel = 'linear', C = 1)

        # モデルの学習
        model.fit(X_train, y_train)

        # 評価結果（識別率）を格納
        LOOscore[i] = model.score(X_test, y_test)


    # 評価結果（識別率）の平均を求める
    result = LOOscore.mean()

    # パーセントに直す
    result = round(result * 100, 1)

    print(str(LOOscore) + '\n')

    return result


# ## SVM_kCV関数
# 引数とし教師データをX，ラベルをyで受け取る．
# 交差検証法の一つk-分割交差検証で識別精度評価を行う．
#
# * 学習
# * (k分割し，1グループをテストデータ，残りグループを教師データにして評価) * k
# * 得られたk個の評価結果（識別率）の平均を求めてパーセントに直す
# * 評価結果（識別率）をmain関数に返す

# In[94]:

def SVM_kCV(X, y):

    # 線形SVMのインスタンスを生成
    model = svm.SVC(kernel = 'linear', C = 1)

    # k分割し，1グループをテストデータ，残りグループを教師データにして評価
    # すべてのグループに対して行う
    # 評価結果（識別率）を格納
    CVscore = cross_validation.cross_val_score(model, X, y, cv = cv_k)

    # 評価結果（識別率）の平均を求める
    result = CVscore.mean()

    # パーセントに直す
    result = round(result * 100, 1)

    print('k = ' + str(cv_k) + '：' + str(CVscore))

    return result



# ## TrainingData関数
# 引数として読み込みたいタスクごとのデータをreset, tapで受け取る．
# * 機械学習にかけれるようにデータのベクトル化とラベルを作成
# * ベクトル化したデータとラベルをmain関数に返す

# In[95]:

def TrainingData(rest, tap):

    # 各タスクのデータを縦結合
    all_data = pd.concat([rest, tap], axis = 0)

    # ベクトル化
    X = all_data.as_matrix()

    # ラベル作成 rest = 0, tap = 1
    label_rest = np.zeros(len(rest.index))
    label_tap = np.ones(len(tap.index))

    y = np.r_[label_rest, label_tap]


    return X, y




# ## main関数

# In[103]:

if __name__ == '__main__':

    # 読み込みたいファイルのパス
    PATH_rest = PATH + 'div_rest.csv'
    PATH_tap = PATH + 'div_tapping.csv'

    # csvファイル読み込み
    # headerは設定せず，転置後にset_index()する（header = 0にすると列名が変えられる）
    rest = pd.read_csv(PATH_rest, header = None).T
    rest = rest.set_index(0)

    tap = pd.read_csv(PATH_tap, header = None).T
    tap = tap.set_index(0)

    # 分割数を求める：列数 / ボクセル数
    divNum = len(rest.columns) // voxels

    # 全ボクセルの識別率を格納するデータフレーム
    voxAc = pd.DataFrame(index = sorted(list(set(rest.index))), columns = [col_name])

    for i in range(voxels):

        # ボクセル名
        voxName = 'Voxel' + str(i+1)

        print(voxName)

        # 各ボクセルごとにデータを取得
        restVox = rest.loc[voxName]
        tapVox = tap.loc[voxName]


        # データとラベルの準備
        data, labels = TrainingData(restVox, tapVox)


        # 学習と交差検証
        print('leave-one-out cross-validation')

        result_loo = SVM_LOO(data, labels)
        print(result_loo)

        # データフレームに格納
        voxAc.at[voxName, col_name] = result_loo



# In[104]:

# csv書き出し
PATH_RESULT = PATH + 'ACCURACY[loo]_voxels_SVM.csv'
voxAc.to_csv(PATH_RESULT, index = True)


# In[ ]:
