
# coding: utf-8

# ## fMRI時系列データを区切る前処理
# ----
#
# 引数：raw_tapping.csv/raw_rest.csvがあるディレクトリまでのパス，分割数
#
# ----
#
# 入力：raw_tapping.csv/raw_rest.csv
#
# ----
#
# 出力： /(分割数)divData/div_tapping.csv, /(分割数)divData/div_rest.csv
#
# ----
#
# State Designで計測したfMRIデータを数ブロックに区切る．（ボクセルごとに識別率を算出するため）

# In[4]:

print('########### Preprocessing_divided.py program excution ##########')


# In[18]:

import glob
import sys
import pandas as pd
import matplotlib.pyplot as plt
import os


# コマンドライン引数でraw_tapping.csv/raw_rest.csvがあるディレクトリまでのパスと分割数を取得

# In[19]:

# args = sys.argvs
# PATH = args[1]
#divNum = int(args[2])

# jupyter notebookのときはここで指定
PATH = '../State-2fe_Brodmann/20181029rn/64ch/RawData/'
divNum = 8


# 後で出力するcsvファイルを保存するディレクトリを作成

# In[20]:

# 作成したいディレクトリ名・パス
DIR_div = PATH + '../' + str(divNum) + 'divData'
PATH_div = DIR_div + '/'

# 既に存在する場合は何もせず，存在していない場合はディレクトリ作成
if not os.path.exists(DIR_div):
    os.mkdir(DIR_div)


# # DataDiv関数
# 引数としてボクセルの時系列fMRIデータをvoxで受け取る.
# divNum(分割数)に応じて一つの時系列fMRIデータを分割し，横結合することで新しいデータフレームを生成．

# In[21]:

def DataDiv(vox):

    # ボクセルのfMRIデータをdivNum分割したデータ格納用データフレーム
    voxData = pd.DataFrame(index = [], columns = [])

    # divNum分割した時の1ブロックあたりのscan数
    divScan = len(vox) // divNum

    for i in range(divNum):

        # データを分割
        divVox = vox.iloc[(divScan*i):(divScan*(i+1))]

        # データフレームに結合するためには行名を合わせとく必要があるので連番を行名とする
        divVox.index = range(divScan)

        # 新しいデータフレームに追加
        voxData = pd.concat([voxData, divVox], axis = 1)

    return voxData


# # main関数

# In[22]:

if __name__ == '__main__':

    # 読み込みたいファイルのパス
    PATH_rest = PATH + 'raw_rest.csv'
    PATH_tap = PATH + 'raw_tapping.csv'

    # csvファイル読み込み
    rest = pd.read_csv(PATH_rest, header = 0)
    tap = pd.read_csv(PATH_tap, header = 0)



# In[23]:

# rest時の全ボクセルをdivNum分割してまとめたデータ格納用データフレーム
restData = pd.DataFrame(index = [], columns = [])

for i in rest.columns:

    print(i)

    # ボクセルごとにdivNum分割
    restVox = DataDiv(rest[i])

    # 全ボクセルをまとめていく
    restData = pd.concat([restData, restVox], axis = 1)


# In[ ]:

# csv書きだし
PATH_rest = PATH_div + 'div_rest.csv'
restData.to_csv(PATH_rest, index = False)


# In[ ]:

# tapping時の全ボクセルをdivNum分割してまとめたデータ格納用データフレーム
tapData = pd.DataFrame(index = [], columns = [])

for i in tap.columns:

    print(i)

    # ボクセルごとにdivNum分割
    tapVox = DataDiv(tap[i])

    # 全ボクセルをまとめていく
    tapData = pd.concat([tapData, tapVox], axis = 1)


# In[ ]:

# csv書きだし
PATH_tap = PATH_div + 'div_tapping.csv'
tapData.to_csv(PATH_tap, index = False)


# In[ ]:
