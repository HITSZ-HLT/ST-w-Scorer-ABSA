while getopts ':d:c:b:s:q:a:' opt
do
    case $opt in
        c) CUDA_IDS="$OPTARG" ;;
        d) dataset="$OPTARG" ;;
        b) subname="$OPTARG" ;;
        s) seed="$OPTARG" ;;
        q) quad_subname="$OPTARG" ;;
        a) date="$OPTARG" ;;
        ?)
        exit 1;;
    esac
done


if [ -z "${CUDA_IDS}" ]; then
    CUDA_IDS=0
fi


if [ -z "${seed}" ]; then
    seed=42
fi


if [ -z "${subname}" ]; then
    subname="scorer"
fi


if [ -z "${quad_subname}" ]; then
    quad_subname="quad"
fi



data_dir="../output/quad/${date}/${dataset}_${quad_subname}_${seed}.json"
model_name_or_path="../output/scorer/model/dataset=${dataset},b=${subname},seed=42"
output_dir="../output/rerank/"

precision=bf16-mixed
max_seq_length=100
eval_batch_size=20


echo "python train_scorer.py predict: dataset ${dataset}, data_dir ${data_dir}"

CUDA_VISIBLE_DEVICES=${CUDA_IDS} python train_scorer.py predict \
  --seed_everything ${seed} \
  --trainer.devices=1 \
  --trainer.accelerator=gpu \
  --trainer.precision=${precision} \
  --data.model_name_or_path "${model_name_or_path}" \
  --data.max_seq_length ${max_seq_length} \
  --data.data_dir "${data_dir}" \
  --data.eval_batch_size ${eval_batch_size} \
  --data.mode "rerank" \
  --model.dataset ${dataset} \
  --model.seed ${seed} \
  --model.model_name_or_path "${model_name_or_path}" \
  --model.output_dir "${output_dir}" \
  --model.subname ${subname} \
  --model.quad_subname ${quad_subname}