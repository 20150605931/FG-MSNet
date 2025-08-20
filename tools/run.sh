arr1=("MS_ResWeightedPartNet50_FL3" "MS_ResWeightedPartNet50_FL7")
for subtest in {0..4}
do
  for method in "${arr1[@]}"
  do
    python tools/train.py models/resnet/${method}.py --kflod-validation ${subtest}
  done
done

