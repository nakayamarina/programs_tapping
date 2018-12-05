# PATH_DATA="../../Data_mri/tappingState-2fe/"
# PATH_SAVE="../"
#
# for dir in 20181029rn 20181029su 20181029tm
# do
#
#   for image_method in 64ch mb
#   do
#
#     PATH_NII="${PATH_DATA}${dir}/${image_method}/"
#
#
#     PATH_BA="${PATH_SAVE}State-2fe_MaskBrodmann/${dir}/${image_method}/"
#
#     echo "------------ ${PATH_NII} | ${PATH_BA} ---------------"
#
#     python Preprocessing_nii2zscore.py ${PATH_NII} ${PATH_BA} rwmaskBA.nii
#
#
#
#     PATH_MOTOR="${PATH_SAVE}State-2fe_MaskMotor/${dir}/${image_method}/"
#
#     echo "------------ ${PATH_NII} | ${PATH_MOTOR} ---------------"
#
#     python Preprocessing_nii2zscore.py ${PATH_NII} ${PATH_MOTOR} rwmask12346.nii
#
#   done
#
#
# done



PATH_BA="../State-2fe_MaskBrodmann/"
# PATH_MA="../State-2fe_MaskMotor/"
# PATH_DATA="../State-2fe_Active/"


DIRs=`ls -F ${PATH_BA} | grep /`

for dir in $DIRs
do

  for image_method in mb
  do
    #
    # PATH_voxel="${PATH_DATA}${dir}${image_method}/"
    #
    # echo "------------ ${PATH_voxel} ---------------"
    # python Preprocessing_tasks.py ${PATH_voxel}
    #

    # PATH_RAW="${PATH_DATA}${dir}${image_method}/RawData/"
    #
    # echo "------------ ${PATH_RAW} ---------------"

    # python Vec_TAUautocor.py ${PATH_RAW}
    #
    # Rscript Vec_TDAvec_autocor_custom100.r ${PATH_RAW}
    # Rscript Vec_TDAvec_autocor_custom300.r ${PATH_RAW}
    #
    # python Vec_TDAvec_revec.py ${PATH_RAW}
    #
    # python ML_SVM_spm.py ${PATH_RAW}
    #
    # python ML_SVM_timeseries.py ${PATH_RAW}
    #
    # python ML_SVM_TDAautocor.py ${PATH_RAW}

    # python ML_1dCNN_timeseries.py ${PATH_RAW}
    #
    # python ML_1dCNN_TDAautocor.py ${PATH_RAW}

    PATH_BACSV="${PATH_BA}${dir}${image_method}/RawData/"
    # PATH_MACSV="${PATH_MA}${dir}${image_method}/RawData/"

    echo "------------ ${PATH_BACSV} ---------------"

    python ML_SVM_VOXtimeseries.py ${PATH_BACSV}


    # echo "------------ ${PATH_MACSV} ---------------"
    #
    # python ML_SVM_VOXtimeseries.py ${PATH_MACSV}

  done


done
