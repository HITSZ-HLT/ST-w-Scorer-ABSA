while getopts ':d:c:b:s:l:m:h:f:t:' opt
do
    case $opt in
        c) CUDA_IDS="$OPTARG" ;;
        d) dataset="$OPTARG" ;;
        b) subname="$OPTARG" ;;
        s) seed="$OPTARG" ;;
        m) model_name_or_path="$OPTARG" ;;
        l) learning_rate="$OPTARG" ;;
        h) max_seq_length="$OPTARG" ;;
        f) filter_setting="$OPTARG" ;;
        t) self_training_data_dir="$OPTARG" ;;
        ?)
        exit 1;;
    esac
done



if [ -z "${CUDA_IDS}" ]; then
    CUDA_IDS=0
fi


if [ -z "${subname}" ]; then
    subname="quad"
fi


if [ -z "${seed}" ]; then
    seed=42
fi


if [ -z "${model_name_or_path}" ]; then
    model_name_or_path="t5-base"
fi


if [ -z "${learning_rate}" ]; then
    learning_rate=30
fi


if [ -z "${max_seq_length}" ]; then
    max_seq_length=-1
fi


if [ -z "${filter_setting}" ]; then
    filter_setting='none'
fi


if [ -z "${self_training_data_dir}" ]; then
    self_training_data_dir=''
fi



precision=bf16-mixed
gradient_clip_val=1
warmup_steps=0
weight_decay=0.01

train_batch_size=16
eval_batch_size=64
max_epochs=20


data_dir="data/t5/"
output_dir="../output/quad/"

echo "python train_quad.py fit: dataset ${dataset}, seed ${seed}, filter_setting ${filter_setting}, self_training_data_dir ${self_training_data_dir}"

CUDA_VISIBLE_DEVICES=${CUDA_IDS} python train_quad.py fit \
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
  --data.train_batch_size ${train_batch_size} \
  --data.eval_batch_size ${eval_batch_size} \
  --data.mode "train_test" \
  --data.self_training_data_dir "${self_training_data_dir}" \
  --data.filter_setting "${filter_setting}" \
  --model.dataset ${dataset} \
  --model.seed ${seed} \
  --model.model_name_or_path "${model_name_or_path}" \
  --model.warmup_steps ${warmup_steps} \
  --model.weight_decay ${weight_decay} \
  --model.learning_rate ${learning_rate}e-5 \
  --model.output_dir "${output_dir}" \
  --model.subname ${subname}

echo "${output_dir}model/dataset=${dataset},b=${subname},seed=${seed}"

CUDA_VISIBLE_DEVICES=${CUDA_IDS} python train_quad.py test \
  --seed_everything ${seed} \
  --trainer.devices=1 \
  --trainer.accelerator=gpu \
  --trainer.precision=${precision} \
  --data.model_name_or_path "${output_dir}model/dataset=${dataset},b=${subname},seed=${seed}" \
  --data.max_seq_length ${max_seq_length} \
  --data.data_dir "${data_dir}" \
  --data.dataset ${dataset} \
  --data.eval_batch_size ${eval_batch_size} \
  --data.mode "train_test" \
  --model.dataset ${dataset} \
  --model.seed ${seed} \
  --model.model_name_or_path "${output_dir}model/dataset=${dataset},b=${subname},seed=${seed}" \
  --model.output_dir "${output_dir}" \
  --model.subname ${subname}


if [ "${self_training_data_dir}" != "" ]; then
    echo "rm ${output_dir}model/dataset=${dataset},b=${subname},seed=${seed}"
    rm -r "${output_dir}model/dataset=${dataset},b=${subname},seed=${seed}"
fi
