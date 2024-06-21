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
    subname="scorer"
fi



data_dir="../output/quad/pseudo_labeled/${dataset}.json"
model_name_or_path="../output/scorer/model/dataset=${dataset},b=${subname},seed=42"
output_dir="../output/filter/"

precision=bf16-mixed
max_seq_length=100
eval_batch_size=80
seed=42


echo "python train_scorer.py filter: dataset ${dataset}, data_dir ${data_dir}"

CUDA_VISIBLE_DEVICES=${CUDA_IDS} python train_scorer.py predict \
  --seed_everything ${seed} \
  --trainer.devices=1 \
  --trainer.accelerator=gpu \
  --trainer.precision=${precision} \
  --data.model_name_or_path "${model_name_or_path}" \
  --data.max_seq_length ${max_seq_length} \
  --data.data_dir "${data_dir}" \
  --data.eval_batch_size ${eval_batch_size} \
  --data.mode "filter" \
  --model.dataset ${dataset} \
  --model.seed ${seed} \
  --model.model_name_or_path "${model_name_or_path}" \
  --model.output_dir "${output_dir}" \
  --model.subname ${subname} 