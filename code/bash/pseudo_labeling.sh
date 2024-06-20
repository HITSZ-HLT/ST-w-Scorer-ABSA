# bash/pseudo_labeling.sh -d asqp/rest16 -c 3 -b test

while getopts ':d:c:b:' opt
do
    case $opt in
        c) CUDA_IDS="$OPTARG" ;;
        d) dataset="$OPTARG" ;;
        b) subname="$OPTARG" ;;
        ?)
        exit 1;;
    esac
done



if [ -z "${CUDA_IDS}" ]; then
    CUDA_IDS=0
fi


if [ -z "${subname}" ]; then
    subname="test"
fi


case "$dataset" in
    "asqp/rest16"|"asqp/rest15"|"acos/rest16")
        data_dir="data/raw/yelp/100k_1.json" ;;
    "acos/laptop16")
        data_dir="data/raw/laptop/100k_1.json" ;;
    *)
        echo "Unknown dataset" ;;
esac



seed=42
precision=bf16-mixed
eval_batch_size=500
max_seq_length=100

model_name_or_path="../output/quad/model/dataset=${dataset},b=${subname},seed=${seed}"
output_dir="../output/quad/pseudo_labeled/${dataset}.json"


echo "python train_quad.py predict: dataset ${dataset}, seed ${seed}"
CUDA_VISIBLE_DEVICES=${CUDA_IDS} python train_quad.py predict \
  --seed_everything ${seed} \
  --trainer.devices=1 \
  --trainer.accelerator=gpu \
  --trainer.precision=${precision} \
  --data.model_name_or_path "${model_name_or_path}" \
  --data.max_seq_length ${max_seq_length} \
  --data.data_dir "${data_dir}" \
  --data.eval_batch_size ${eval_batch_size} \
  --data.mode "predict" \
  --model.seed ${seed} \
  --model.model_name_or_path "${model_name_or_path}" \
  --model.output_dir "${output_dir}" \