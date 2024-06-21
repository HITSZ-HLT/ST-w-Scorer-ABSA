while getopts ':d:c:b:s:l:m:t:j:e:a:k:z:' opt
do
    case $opt in
        c) CUDA_IDS="$OPTARG" ;;
        d) dataset="$OPTARG" ;;
        b) subname="$OPTARG" ;;
        s) seed="$OPTARG" ;;
        m) model_name_or_path="$OPTARG" ;;
        l) learning_rate="$OPTARG" ;;
        t) setting="$OPTARG" ;;
        j) objective="$OPTARG" ;;
        e) beta="$OPTARG" ;;
        a) alpha="$OPTARG" ;;
        k) k="$OPTARG" ;;
        z) train_batch_size="$OPTARG" ;;
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


if [ -z "${seed}" ]; then
    seed=42
fi


if [ -z "${model_name_or_path}" ]; then
    model_name_or_path="/data/zhangyice/2023/pretrained_models/t5-large"
fi


if [ -z "${learning_rate}" ]; then
    learning_rate=30
fi


if [ -z "${setting}" ]; then
    setting='01234+'
fi


if [ -z "${objective}" ]; then
    objective='list'
fi


if [ ! "${alpha}" ]
then
    alpha=1
fi


if [ ! "${beta}" ]
then
    beta=1
fi


if [ ! "${k}" ]
then
    k=2500
fi


if [ ! "${train_batch_size}" ]
then
    train_batch_size=10
fi



precision=bf16-mixed
gradient_clip_val=1
warmup_steps=100
weight_decay=0.01
max_seq_length=100

eval_batch_size=64
max_epochs=10
use_ai_preference=True

data_dir="data/t5/"
preference_data_dir="data/comp/"
output_dir="../output/scorer/"

echo "python train_scorer.py fit: dataset ${dataset}, seed ${seed}"

CUDA_VISIBLE_DEVICES=${CUDA_IDS} python train_scorer.py fit \
  --seed_everything ${seed} \
  --trainer.devices=1 \
  --trainer.enable_checkpointing=False \
  --trainer.enable_model_summary=False \
  --trainer.accelerator=gpu \
  --trainer.precision=${precision} \
  --trainer.max_epochs ${max_epochs} \
  --trainer.gradient_clip_val ${gradient_clip_val} \
  --trainer.check_val_every_n_epoch 1 \
  --data.model_name_or_path "${model_name_or_path}" \
  --data.max_seq_length ${max_seq_length} \
  --data.data_dir "${data_dir}" \
  --data.dataset ${dataset} \
  --data.preference_data_dir ${preference_data_dir} \
  --data.train_batch_size ${train_batch_size} \
  --data.eval_batch_size ${eval_batch_size} \
  --data.setting "${setting}" \
  --data.k ${k} \
  --data.use_ai_preference ${use_ai_preference} \
  --model.dataset ${dataset} \
  --model.seed ${seed} \
  --model.model_name_or_path "${model_name_or_path}" \
  --model.warmup_steps ${warmup_steps} \
  --model.weight_decay ${weight_decay} \
  --model.learning_rate ${learning_rate}e-5 \
  --model.output_dir "${output_dir}" \
  --model.subname ${subname} \
  --model.objective ${objective} \
  --model.beta ${beta} \
  --model.alpha ${alpha} \
  --model.setting ${setting} \
  --model.k ${k}
