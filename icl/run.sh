set -x
base_dir=/mnt/data/peiyu.liu/DecoQuant_code/icl

function run_ori(){
    CUDA_VISIBLE_DEVICES=$2 python3 evaluate_icl_ori.py --model_path /mnt/liupeiyu/nlp_model/opt-1.3b --dataset ag_news --num_plaintext_demonstrations 0 --use_calibration $3 > $base_dir/logs/r16/1.3b/${1}_$(date "+%Y%m%d-%H%M%S").log 2>&1 &
}
function run(){
    CUDA_VISIBLE_DEVICES=$2 python3 evaluate_icl.py --model_path /mnt/liupeiyu/llama_checkpoint/llama-7b/llama-7b/ --dataset ag_news --num_plaintext_demonstrations 2 --use_calibration $3 > $base_dir/logs/r16/1.3b/${1}_$(date "+%Y%m%d-%H%M%S").log 2>&1 &
}
exp_name=$1
## subj-demo2
data_name=ag_news
# data_name=subj
# data_name=sst2
# data_name=rte
# data_name=boolq
# data_name=rte
# data_name=wic
# data_name=wsc
# data_name=multirc

model_name=/mnt/data/peiyu.liu/bak_data3/models/opt_1_3b
# model_name=/media/public/models/huggingface/llama-7b
# model_name=facebook/opt-6.7b

# run_ori eval_${data_name}_ori_0 4 --num_plaintext_demonstrations=0\ --dataset=${data_name}\ --model_path=${model_name}
# run_ori eval_${data_name}_ori_2 7 --num_plaintext_demonstrations=2\ --dataset=${data_name}\ --model_path=${model_name}
# run_ori eval_${data_name}_ori_10 6 --num_plaintext_demonstrations=10\ --dataset=${data_name}\ --model_path=${model_name}
# run eval_${data_name}_naive_quant2_demo2 4 --num_bits=2\ --exp=naive\ --num_plaintext_demonstrations=2\ --dataset=${data_name}\ --model_path=${model_name}
# run eval_${data_name}_naive_quant4_demo2 7 --num_bits=4\ --exp=naive\ --num_plaintext_demonstrations=2\ --dataset=${data_name}\ --model_path=${model_name}
# run eval_${data_name}_naive_quant2_demo10 4 --num_bits=2\ --exp=naive\ --num_plaintext_demonstrations=10\ --dataset=${data_name}\ --model_path=${model_name}
# run eval_${data_name}_naive_quant4_demo10 0 --num_bits=4\ --exp=naive\ --num_plaintext_demonstrations=10\ --dataset=${data_name}\ --model_path=${model_name}
run eval_${data_name}_ours_quant2_demo2 0 --num_bits=2\ --exp=ours\ --num_plaintext_demonstrations=2\ --dataset=${data_name}\ --model_path=${model_name}
# run eval_${data_name}_ours_quant4_demo2 6 --num_bits=4\ --exp=ours\ --num_plaintext_demonstrations=2\ --dataset=${data_name}\ --model_path=${model_name}
# run eval_${data_name}_ours_quant4_demo10_${exp_name} 4 --num_bits=4\ --exp=ours\ --num_plaintext_demonstrations=10\ --dataset=${data_name}\ --model_path=${model_name}
# run eval_${data_name}_ours_quant2_demo10 6 --num_bits=2\ --exp=ours\ --num_plaintext_demonstrations=10\ --dataset=${data_name}\ --model_path=${model_name}