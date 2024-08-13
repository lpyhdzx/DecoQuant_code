export OMP_NUM_THREADS=20


function run(){
    CUDA_VISIBLE_DEVICES=$2 nohup python -u smoothquant_opt_demo.py --exp=$3 --model=$4 --n_bits=$5 --dataset=lambada $6 > logs/opt_total/$1_$(date "+%Y%m%d-%H%M%S").log 2>&1 &
}
function run_debug(){
    CUDA_VISIBLE_DEVICES=$2 python -u smoothquant_opt_demo.py --exp=$3 --model=$4 --n_bits=$5 --dataset=lambada $6
}
function run_llama(){
    CUDA_VISIBLE_DEVICES=$2 nohup python -u smoothquant_llama_demo.py --exp=$3 --model=$4 --n_bits=$5 $6 > logs/llama_total/$1_$(date "+%Y%m%d-%H%M%S").log 2>&1 &
}
function run_llama_debug(){
    CUDA_VISIBLE_DEVICES=$2 python -u smoothquant_llama_demo.py --exp=$3 --model=$4 --n_bits=$5 $6
}

############################################################
##################### start exp for opt-1.3B ###############
############################################################
# run fp16_opt1_3 0 fp16 /media/liupeiyu/opt-1.3b 8 --limit=10000\ --weight_quant="noweight"\ --act_quant="per_token"

# #     W16A_
# # naive_W16A_
# run naive_8bit_opt1_3_W16A8 0 base_naive_kv /media/liupeiyu/opt-1.3b 8 --limit=10000\ --weight_quant="noweight"\ --act_quant="per_token"
# run naive_4bit_opt1_3_W16A4 [gpu] base_naive_kv /media/liupeiyu/opt-1.3b 4 --limit=10000\ --weight_quant="noweight"\ --act_quant="per_token"
# run naive_2bit_opt1_3_W16A2 [gpu] base_naive_kv /media/liupeiyu/opt-1.3b 2 --limit=10000\ --weight_quant="noweight"\ --act_quant="per_token"
# # # ours_W16A_
# run ours_8bit_opt1_3_W16A8 1 base_ours_kv /media/liupeiyu/opt-1.3b 8 --limit=10000\ --weight_quant="noweight"\ --act_quant="per_token"
# run ours_4bit_opt1_3_W16A4 [gpu] base_ours_kv /media/liupeiyu/opt-1.3b 4 --limit=10000\ --weight_quant="noweight"\ --act_quant="per_token"
# run ours_2bit_opt1_3_W16A2 [gpu] base_ours_kv /media/liupeiyu/opt-1.3b 2 --limit=10000\ --weight_quant="noweight"\ --act_quant="per_token"

# #     # W_A_
# # # naive_W_A_
# run naive_8bit_opt1_3_W8A8 [gpu] base_naive_kv /media/liupeiyu/opt-1.3b 8 --limit=10000\ --weight_quant="per_channel"\ --act_quant="per_token"
# run naive_4bit_opt1_3_W4A4 0 base_naive_kv /media/liupeiyu/opt-1.3b 4 --limit=10000\ --weight_quant="per_channel"\ --act_quant="per_token"
# run naive_2bit_opt1_3_W2A2 [gpu] base_naive_kv /media/liupeiyu/opt-1.3b 2 --limit=10000\ --weight_quant="per_channel"\ --act_quant="per_token"
# # # ours_W_A_
# run ours_8bit_opt1_3_W8A8 [gpu] base_ours_kv /media/liupeiyu/opt-1.3b 8 --limit=10000\ --weight_quant="per_channel"\ --act_quant="per_token"
# run ours_4bit_opt1_3_W4A4 [gpu] base_ours_kv /media/liupeiyu/opt-1.3b 4 --limit=10000\ --weight_quant="per_channel"\ --act_quant="per_token"
# run ours_2bit_opt1_3_W2A2 [gpu] base_ours_kv /media/liupeiyu/opt-1.3b 2 --limit=10000\ --weight_quant="per_channel"\ --act_quant="per_token"

# # smoothqaunt_W_A_
# run sq_8bit_opt1_3_W8A8 [gpu] smoothquant /media/liupeiyu/opt-1.3b 8 --limit=10000\ --weight_quant="per_channel"\ --act_quant="per_token"
# run sq_4bit_opt1_3_W4A4 8 smoothquant /media/liupeiyu/opt-1.3b 4 --limit=10000\ --weight_quant="per_channel"\ --act_quant="per_token"
# run sq_2bit_opt1_3_W2A2 8 smoothquant /media/liupeiyu/opt-1.3b 2 --limit=10000\ --weight_quant="per_channel"\ --act_quant="per_token"

# # smoothqaunt_W_A_
# run sq_8bit_opt67_W8A8 0 smoothquant /media/public/models/huggingface/facebook/opt-6.7b 8 --limit=10000\ --weight_quant="per_channel"\ --act_quant="per_token"
# run sq_4bit_opt67_W4A4 8 smoothquant /media/public/models/huggingface/facebook/opt-6.7b 4 --limit=10000\ --weight_quant="per_channel"\ --act_quant="per_token"
# run sq_2bit_opt67_W2A2 8 smoothquant /media/public/models/huggingface/facebook/opt-6.7b 2 --limit=10000\ --weight_quant="per_channel"\ --act_quant="per_token"