PATH_DATA="../State-2fe_Motor/"

for dir in 20181029rn/ 20181029su/ 20181029tm/
do

  for image_method in 64ch mb
  do


    PATH_RAW="${PATH_DATA}${dir}${image_method}/RawData/"
    PATH_DIV="${PATH_DATA}${dir}${image_method}/8divData/"

    echo "------------ ${PATH_RAW} ---------------"

    python Preprocessing_divided.py ${PATH_RAW} 8

    python ML_SVM_voxels.py ${PATH_DIV}

  done


done
