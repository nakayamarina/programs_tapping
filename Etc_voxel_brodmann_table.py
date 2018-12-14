
# coding: utf-8

# # ボクセル番号とブロードマンエリア対応表

# ----
#
# 引数：4D化した.niiファイルがあるフォルダまでのパス / 書き出すcsvの保存先
#
# ---
#
# 入力：4D化した.niiファイルがあるフォルダまでのパス / 書き出すcsvの保存先 / 各種mask.nii（chunks_list.csv, targets_list.csv）
#
# ---
#
# 出力：Table_voxel-brodmann.csv（chunks_list.csv, targets_list.csv）
#
# ---
#
# ボクセル番号からブロードマンエリアを特定するための対応表を作る．
# そのためには，被験者の脳データの4D.nii，全ブロードマンエリアのマスク，各ブロードマンエリアのマスクを作っておく必要がある．
# 詳しくはPreprocessing_nii2zscore.py参照．

# In[1]:

from mvpa2.suite import *
from mvpa2.datasets.mri import fmri_dataset
import os
import os.path
from os.path import join as pathjoin
from pprint import pprint
# from nifti import NiftiImage
import glob
import numpy as np
import pandas as pd
import sys
import pickle
# import dill
import csv

import nibabel as nib


# In[203]:

# !!!!!!!!! scanNumらへん，datasetのパラメータ書き換え！！

args = sys.argv
PATH = args[1]
PATH_save = args[2]

#jupyter notebookのときはここで指定
# PATH = '../../Data_mri/tappingState-2fe/20181029rn/mb/'
# PATH_save = '../State-2fe_MaskBrodmann/20181029rn/mb/'

# 前処理済みならswrがついた.niiファイルを選択
PATH_nii = PATH + '4D.nii'

# マスクの種類数：全ブロードマンエリア + 各ブロードマンエリア（75種）= 76
maskNum = 76

# マスク名のリストとブロードマンエリア名のリスト作成
mask_list = []
ba_list = []

for i in range(maskNum):

    mask_list = mask_list + [PATH + 'rwmask' + str(i) + '.nii']
    ba_list = ba_list + ['BrodmannArea' + str(i)]

mask_list[0] = PATH + 'rwmaskBA.nii'

# ブロードマンエリア1-47の他に名前のついた部位があるのでその名前リスト
etc_area = ['Amygdala', 'AnteriorCommissure', 'CaudateBody', 'CaudateHead',
            'CaudateTail', 'CorpusCallsum', 'Dentate', 'Hippocampus', 'Hypothalamus',
            'LateralDorsalNucleus', 'LateralGeniculumBody', 'LateralGlobusPallidus',
            'LateralPosteriorNuckleus', 'MammillaryBody', 'MedialDorsalNucleus', 'OpticTract',
            'MedialGeniculumBody', 'MedialGlobusPallidus', 'MidlineNucleus',
            'Pulvinar', 'Putamen', 'RedNucleus', 'SubstaniaNigra', 'SubthalamicNucleus',
            'VentralAnteriorNucleus', 'VentralLateralNucleus',
            'VentralPosteriorLateralNucleus', 'VentralPosteriorMediaNucleus']

ba_list[0] = 'BrodmannAreaAll'
ba_list[48:maskNum] = etc_area

# In[116]:

headcoil = PATH.split('/')[5]

if headcoil == 'mb':

    # 総スキャン数
    scan_num = 592

    # 1タスクのスキャン数
    taskNum = 296

elif headcoil == '64ch':

    # 総スキャン数
    scan_num = 192

    # 1タスクのスキャン数
    taskNum = 96


# In[117]:

# RawDataのディレクトリ名・パス
DIR_RAW = PATH_save + 'RawData'
PATH_RAW = DIR_RAW + '/'

# すでに存在する場合は何もせず，存在していない場合はディレクトリ作成
if not os.path.exists(DIR_RAW):
    os.mkdir(DIR_RAW)


# # CorrespondenceTable関数

# In[6]:

def CorrespondenceTable(target, chunk, mask):

    # データセットの整形

    dataset = fmri_dataset(nifti, targets=target, chunks=chunk, mask=mask ,sprefix='voxel', tprefix='time', add_fa=None)

    print('dataset ready')

    poly_detrend(dataset, polyord=1, chunks_attr='chunks')

    dataset = dataset[np.array([l in ['0', '1']
                               for l in dataset.targets], dtype='bool')]

    # ボクセル数を取得し，ボクセル名作成
    voxNum = dataset.shape[1]

    VoxName = []

    for i in range(voxNum):

        name = 'Voxel' + str(i+1)
        VoxName.append(name)


    # ボクセル位置(x, y, z)を取得
    voxPosition = dataset.fa.values()
    voxPosition = list(voxPosition)[0][:]
    voxPosition = pd.DataFrame(voxPosition, columns = ['x', 'y', 'z'], index = VoxName)

    return voxPosition


# # Main関数

# In[179]:

if __name__ == '__main__':

    ########## ボクセルデータ（情報）抽出準備 #########

    # 4D化した.niiファイル名リストを作成

    nifti = [PATH_nii]


    # In[8]:

    # 教師データの作成（この作業なしにやる方法がわからないので，必要のない作業ではあるがやる）

    task = ['0'] * taskNum
    task2 = ['1'] * taskNum

    task.extend(task2)
    target = pd.DataFrame(task)

    target = pd.DataFrame(task)

    PATH_target = PATH + 'targets_list.csv'
    target.to_csv(PATH_target, index = False, header = None)
    print('target')

    targets_list = []

    targets_file = open(PATH_target, 'rU')
    dataReader = csv.reader(targets_file)

    for row in dataReader:
        targets_list.append(row[0])


    # In[13]:

    # チャンク（試行数リスト？）の作成（この作業なしにやる方法がわからないので，必要のない作業ではあるがやる）

    chunk = ['1'] * scan_num

    chunks = pd.DataFrame(chunk)

    PATH_chunk = PATH + 'chunks_list.csv'
    chunks.to_csv(PATH_chunk, index = False, header = None)
    print('chunks')

    chunks_list = []

    chunks = open(PATH_chunk, 'rU')

    for x in chunks:
        chunks_list.append(x.rstrip('\r\n'))

    chunks.close()

    # 各ブロードマンエリアの対応表を結合する用データフレーム
    BAeachs = pd.DataFrame(index = [], columns = [])

    # In[180]:

    ########## マスクごとに抽出したボクセルの座標一覧取得 #########

    for i in range(len(mask_list)):

        mask = mask_list[i]
        print(mask)

        # 最初のmaskは全ブロードマンエリアのもの
        if i == 0:

            print('-> ' + str(i))

            BA = CorrespondenceTable(targets_list, chunks_list, mask)
            print(ba_list[i] + ' : ' + str(len(BA)))

            # ボクセル番号を列として追加しておく
            BAvoxName = pd.DataFrame(BA.index, columns = ['BAvoxelNum'], index = BA.index)
            BA = pd.concat([BA, BAvoxName], axis = 1)

        # 時系列データ（ボクセル）が存在しないブロードマンエリア12，14，15．16は除外
        elif (i == 12 or i == 14 or i == 15 or i == 16 or i == 26):

            print('-> NaN')

        else:

            print('-> ' + str(i))

            BAeach = CorrespondenceTable(targets_list, chunks_list, mask)
            print(ba_list[i] + ' : ' + str(len(BAeach)))

    #             # ボクセル番号を列として追加しておく
    #             BAvoxName = pd.DataFrame(BAeach.index, columns = ['voxelNum'], index = BAeach.index)
    #             BAeach = pd.concat([BAeach, BAvoxName], axis = 1)

            # ブロードマンエリア名を結合
            BAname = pd.DataFrame([ba_list[i]] * len(BAeach), columns = ['BrodmannArea'], index = BAeach.index)

            BAeachname = pd.concat([BAeach, BAname], axis = 1)

            # 結合する
            BAeachs = pd.concat([BAeachs, BAeachname])

    PATH_be = PATH_RAW + 'VoxelTable.csv'
    print(PATH_be)
    BAeachs.to_csv(PATH_be)

    # In[192]:

    ########## 全ブロードマンエリアのボクセルナンバーとブロードマンエリア名の対応表作成（不明ボクセル情報取得） #########

    # maskから得た全ブロードマンエリアの座標と各ブロードマンエリア結合によって得られた座標をキーとして結合
    # how = 'left'を指定することで，各ブロードマンエリアの座標のどれとも当てはまらないものはNaNになる
    VoxBaTable = pd.merge(BA, BAeachs, on = ['x', 'y', 'z'], how = 'left')

    # 欠損（どのブロードマンエリアとも当てはまらない座標をもつもの）はunknownと名付ける
    uk = 'unknown'
    VoxBaTable = VoxBaTable.fillna({'BrodmannArea':uk})


    # In[193]:

    # NaNを含むもの一覧を取得，csv書き出ししておく
    allNanTable = VoxBaTable[VoxBaTable['BrodmannArea'] == uk]
    PATH_aNT = PATH_RAW + 'VoxelBrodmannTable_Nanall.csv'
    print(PATH_aNT)
    allNanTable.to_csv(PATH_aNT)

    # 行名を連番にしておく
    index_name = range(len(allNanTable))
    allNanTable.index = index_name


    # In[195]:

    # 対応表をcsv書き出し

    # ボクセルナンバーを行名にしておく
    VoxBaTable_new = VoxBaTable
    VoxBaTable_new = VoxBaTable_new.set_index(['BAvoxelNum'])

    PATH_vbt = PATH_RAW + 'VoxelBrodmannTable.csv'
    print(PATH_vbt)
    VoxBaTable_new.to_csv(PATH_vbt)


    # In[177]:

    ########## （不明ボクセル情報取得） #########

    # maskから得た全ブロードマンエリアの座標と各ブロードマンエリア結合によって得られた座標をキーとして結合
    # how = 'right'を指定することで，全ブロードマンエリアの座標のどれとも当てはまらないものはNaNになる
    VoxBaTable_right = pd.merge(BA, BAeachs, on = ['x', 'y', 'z'], how = 'right')

    # 欠損（どのブロードマンエリアとも当てはまらない座標をもつもの）はunknownと名付ける
    uk = 'unknown'
    VoxBaTable_right = VoxBaTable_right.fillna({'BAvoxelNum':uk})


    # In[178]:

    # NaNを含むもの一覧を取得，csv書き出ししておく
    eachNanTable = VoxBaTable_right[VoxBaTable_right['BAvoxelNum'] == uk]
    PATH_eNT = PATH_RAW + 'VoxelBrodmannTable_NanEach.csv'
    print(PATH_eNT)

    # 行名を連番にしておく
    index_name = range(len(eachNanTable))
    eachNanTable.index = index_name

    eachNanTable.to_csv(PATH_eNT)


    # In[201]:

    ########## 作成した対応表における各ブロードマンエリアのボクセル数一覧作成 #########

    # 格納用データフレーム
    col_name = ['Number of Voxels']
    # unknown分も数えるため
    ba_list_new = ba_list + [uk]
    BAvoxNums = pd.DataFrame(index = ba_list_new, columns = col_name)

    i = 0

    for ba in ba_list_new:

        if ba == ba_list[0]:

            # 全ボクセル数
            num = len(VoxBaTable)
            BAvoxNums.loc[ba, col_name] = num

        else:

            # 各ブロードマンエリアのボクセル数
            num = len(VoxBaTable[VoxBaTable['BrodmannArea'] == ba])
            BAvoxNums.loc[ba, col_name] = num


        print(ba + ' : ' + str(num))

        i = i + 1

    # maskBAから得られたボクセル数を取得，結合しておく（重複のせいで増えてるっぽい）
    BaNums_origin = pd.DataFrame(index = ['BrodmannAreaOriginal'], columns = col_name)
    BaNums_origin.iloc[0,0] = len(BA)
    BAvoxNums = pd.concat([BaNums_origin, BAvoxNums])

    PATH_BAvn = PATH_RAW + 'Number_of_BAvoxels.csv'
    BAvoxNums.to_csv(PATH_BAvn)


    # In[ ]:




# In[ ]:
