while getopts ':c:m:d:b:s:l:n:h:f:t:' opt
do
    case $opt in
        c) CUDA_IDS="$OPTARG" ;;
        m) model_name_or_path="$OPTARG" ;;
        d) datasets="$OPTARG" ;;
        b) subname="$OPTARG" ;;
        s) seeds="$OPTARG" ;;
        l) learning_rate="$OPTARG" ;;
        n) n_worker="$OPTARG" ;;
        h) max_seq_length="$OPTARG" ;;
        f) filter_setting="$OPTARG" ;;
        t) self_training_data_dir="$OPTARG" ;;
        ?)
        exit 1;;
    esac
done


if [ -z "${seeds}" ]; then
    seeds="42 52 62 72 82"
fi

IFS=' ' read -r -a seed_array <<< "$seeds"



if [ -z "${datasets}" ]; then
    datasets="acos/rest16 acos/laptop16 asqp/rest15 asqp/rest16"
fi

IFS=' ' read -r -a dataset_array <<< "$datasets"


echo "${seeds}"
echo "${datasets}"


if [ -z "${model_name_or_path}" ]; then
    model_name_or_path='/data/zhangyice/2023/pretrained_models/t5-base'
fi


if [ -z "${learning_rate}" ]; then
    learning_rate=30
fi


if [ -z "${n_worker}" ]; then
    n_worker=3
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



parallel -j${n_worker} \
    bash/train_quad.sh -d {1} -c ${CUDA_IDS} -b ${subname} -s {2} -m ${model_name_or_path} -l ${learning_rate} -h ${max_seq_length} -f "${filter_setting}" -t "${self_training_data_dir}" \
    ::: ${dataset_array[@]} \
    ::: ${seed_array[@]}
