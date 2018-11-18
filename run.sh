PATH_DATA="../State-2fe_Motor/"


DIRs=`ls -F ${PATH_DATA} | grep /`

for dir in $DIRs
do

  for image_method in 64ch mb
  do

    PATH_voxel="${PATH_DATA}${dir}${image_method}/"

    # echo "------------ ${PATH_voxel} ---------------"
    # python Preprocessing_tasks.py ${PATH_voxel}
    #
    #
    PATH_RAW="${PATH_DATA}${dir}${image_method}/RawData/"

    echo "------------ ${PATH_RAW} ---------------"

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

    python ML_1dCNN_timeseries.py ${PATH_RAW}

    python ML_1dCNN_TDAautocor.py ${PATH_RAW}

  done


done
