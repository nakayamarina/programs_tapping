
# coding: utf-8

# # 実験から得られたfMRIデータの前処理
# ----
#
# 引数：Y01.csv, Y02.csv,... の入ったVexelディレクトリがあるディレクトリまでのパス
#
# ---
#
# 入力：Y01.csv, Y02.csv,...
#
# ---
#
# 出力：
# * RawData/raw_all.csv : すべてのボクセルrest,tapping時のZ-scoreをまとめたもの
# * RawData/raw_rest.csv : rest時のZ-scoreだけをまとめたもの
# * RawData/raw_tapping.csv : tapping時のZ-scoreだけをまとめたもの
# * RawData/Raw_image/voxel[ボクセル番号]_rest.png：rest時の各ボクセルのデータをプロットしたもの
# * RawData/Raw_image/voxel[ボクセル番号]_tapping.png：tapping時の各ボクセルのデータをプロットしたもの
#
# [ボクセル番号]には列名にもあるボクセルの数
#
# ----
#
#
# /VoxelディレクトリのY01.csv, Y02.csv, ... のデータには，選択してきた数ボクセルそれぞれのZ-score（賦活度合いみたいなもの）が記録されている．
#
# ここでは，全タスク，各タスクごとに分別した時系列データを得る．
#
#

# In[1]:

print('########## Preprocessing_tasks.py program excution ############')


# In[2]:

import glob
import sys
import pandas as pd
import matplotlib.pyplot as plt
import os


# コマンドライン引数で/Voxelディレクトリがあるディレクトリまでのパスを取得

# In[14]:

args = sys.argv
PATH_pre = args[1]

# jupyter notebookのときはここで指定
# PATH_pre = '../State-2fe_Active/20181029rn/mb/'

# /Voxelディレクトリまでのパス
PATH = PATH_pre + 'Voxel/'


# 後で出力するcsvファイルを保存するディレクトリ（RawData）、pngファイルを保存するディレクトリ（Raw_image）を作成

# In[15]:

# RawDataのディレクトリ名・パス
DIR_RAW = PATH + '../RawData'
PATH_RAW = DIR_RAW + '/'

# すでに存在する場合は何もせず，存在していない場合はディレクトリ作成
if not os.path.exists(DIR_RAW):
    os.mkdir(DIR_RAW)

# Raw_imageのディレクトリ名・パス
DIR_image = PATH_RAW + 'Raw_image'
PATH_image = DIR_image + '/'

# すでに存在する場合は何もせず，存在していない場合はディレクトリ作成
if not os.path.exists(DIR_image):
    os.mkdir(DIR_image)

headcoil = PATH_pre.split('/')[3]

# plotするなら1，plotしないなら0
imgPlot = 1

if headcoil == 'mb':

    # 1タスクのスキャン数
    taskNum = 296

else :

    # 1タスクのスキャン数
    taskNum = 96


# ## splitTask関数
#
# 引数に１ブロックのscan数を受け取り，タスクごとにデータを分けてcsvファイルで書き出し

# In[16]:

def splitTask(brain):

    # 各タスクごとのマスク作成
    maskRest = ([True] * taskNum) + ([False] * taskNum)
    maskTapping = ([False] * taskNum) + ([True] * taskNum)

    # mask適用
    dataRest = brain[maskRest]
    dataTapping = brain[maskTapping]

    # csv書き出し
    PATH_rest = PATH_RAW + 'raw_rest.csv'
    dataRest.to_csv(PATH_rest, index = False)

    PATH_tapping = PATH_RAW + 'raw_tapping.csv'
    dataTapping.to_csv(PATH_tapping, index = False)

    if imgPlot == 1:

        plotIMAGE(dataRest, 'rest')
        plotIMAGE(dataTapping, 'tapping')


# ## plotIMAGE関数
#

# In[17]:

def plotIMAGE(data, task):

    # indexが連番になっていないのでreset_indexで番号を振り直す
    # drop=Trueにしなければ古いindexが新しい列として追加されてしまう
    data = data.reset_index(drop = True)

    # ボクセル（列）の数だけ繰り返す
    for i in range(len(data.columns)):

        # この後に出力するpngファイル名
        FILE_NAME = DIR_image + '/voxel' + str(i+1) + '_' + task + '.png'

        # データをplot
        plt.plot(data.iloc[:, i], label = 'fMRIdata')

        # グラフのタイトル
        graph_name = 'fMRIdata : ' + task + '-voxel' + str(i+1)
        plt.title(graph_name)
        plt.ylim([-5,5])
        plt.ylabel('Z-score')
        plt.xlabel('Time(scan)')

        # グラフの凡例
        plt.legend()

        # ファイル名をつけて保存，終了
        plt.savefig(FILE_NAME)
        plt.close()

        print(FILE_NAME)


# ## main関数

# * fMRIデータ読み込み
# * 全ボクセルデータ連結
# * 全ボクセルデータをcsvで書き出し

# In[18]:

if __name__ == '__main__':
    # /Voxelディレクトリ内のcsvファイルのパスを取得
    csv_file = PATH + '*.csv'
    files = []
    files = glob.glob(csv_file)
    files.sort()


# In[19]:

# 1つ目のファイルを読み込む

# 列名
row_name = "Voxel1"

# 列名をつけてデータフレームとして読み込み（row_nameの後に','をつけることで1列だけ名前をつけることができる）
brain = pd.read_csv(files[0], names=(row_name,))


# In[20]:

# 同様に2つ目以降のファイルをデータフレームとして読み込み，1つ目のデータフレームに横連結
for i in range(1, len(files)):

    row_name = "Voxel" + str(i+1)
    data = pd.read_csv(files[i], names=(row_name,))

    brain = pd.concat([brain, data], axis = 1)



# In[21]:

splitTask(brain)


# In[ ]:




# In[ ]:
