
# coding: utf-8

# # 穴の種類を限定

# ---
#
# 引数：TDAでベクトル化した(全ての穴の種類をベクトルとして含んだもの)csvファイルがあるディレクトリまでのパス
#
# ---
#
# 入力：TDAでベクトル化したcsvファイル
#
# ---
#
# 出力：
# * TDAvec_autocor_tap_(パラメータ).csv
# * TDAvec_autocor_rest_(パラメータ).csv
#
# ---
#
# 0-dim，1-dim，2-dimの穴の情報を含んだベクトルデータにおいて，穴の種類を選択しベクトル化し直す．
#

# In[8]:

import numpy as np
import pandas as pd
import sys


# コマンドライン引数で提案手法でベクトル化したcsvファイルがあるディレクトリまでのパスを取得

# In[9]:

args = sys.argv
PATH = args[1]

# jupyter notebookのときはここで指定
# PATH = '../State-2fe_Active/20181029rn/64ch/RawData/'

# ベクトル化する際の手法
PM = 'TDAvec_autocor'

# ベクトル化された際のきざみ時間
kizamiNum = [100, 300]


# In[10]:


# ベクトル化する穴の種類
# 使わないもの（使う穴が2つの場合）
NotHole = 2
# or
# 使うもの（使う穴が1つの場合）
UseHole = 10
# を設定する．設定しない場合は10とかにする


# ## ReVec関数

# 引数として読み込みたいファイル名をfileで受け取る．
# 条件に応じて再ベクトル化する（reVec）．
# 出力ファイル用に条件を含んだファイルの名前を作成しておく（parameters）．
# reVec，parametersを返す．

# In[11]:

def ReVec(file, kizamiNum):

    # 読み込みたいファイルのパス
    PATH_file = PATH + file

    # csvファイル読み込み
    data = pd.read_csv(PATH_file, header = 0)

    # 条件に応じた穴の種類のベクトル抽出，出力ファイル用の名前作成
    if (NotHole == 0 and UseHole == 10) :

        reVec = data.iloc[:, kizamiNum:kizamiNum*3]

        parameters = '12dim' + str(kizamiNum)

    elif (NotHole == 1 and UseHole == 10) :

        reVec1 = data.iloc[:, 0:kizamiNum]
        reVec2 = data.iloc[:, kizamiNum*2:kizamiNum*3]
        reVec = pd.concat([reVec1, reVec2], axis = 1)

        parameters = '02dim' + str(kizamiNum)

    elif (NotHole == 2 and UseHole == 10) :

        reVec = data.iloc[:, 0:kizamiNum*2]

        parameters = '01dim' + str(kizamiNum)

    elif (UseHole == 0 and NotHole == 10) :

        reVec = data.iloc[:, 0:kizamiNum]

        parameters = '0dim' + str(kizamiNum)

    elif (UseHole == 1 and NotHole == 10) :

        reVec = data.iloc[:, kizamiNum:kizamiNum*2]

        parameters = '1dim' + str(kizamiNum)

    elif (UseHole == 2 and NotHole == 10) :

        reVec = data.iloc[:, kizamiNum*2:kizamiNum*3]
        parameters = '2dim' + str(kizamiNum)

    return reVec, parameters


# ## main関数

# In[12]:

if __name__ == '__main__':

    for kizami in range(len(kizamiNum)):

        # ベクトル化し直すデータ
        ReVec_restData = 'TDAvec_autocor_rest_012dim' + str(kizamiNum[kizami]) + '.csv'
        ReVec_tappingData = 'TDAvec_autocor_tapping_012dim' + str(kizamiNum[kizami]) + '.csv'

        restReVec, paraName = ReVec(ReVec_restData, kizamiNum[kizami])
        tappingReVec, paraName = ReVec(ReVec_tappingData, kizamiNum[kizami])

        # csv書き出し
        PATH_restReVec = PATH + PM + '_rest_' + paraName + '.csv'
        restReVec.to_csv(PATH_restReVec, index = False, header = True)
        print(PATH_restReVec)

        PATH_tappingReVec = PATH + PM + '_tapping_' + paraName + '.csv'
        tappingReVec.to_csv(PATH_tappingReVec, index = False, header = True)
        print(PATH_tappingReVec)



# In[ ]:




# In[ ]:
