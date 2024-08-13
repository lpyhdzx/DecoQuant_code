import torch
import sys
sys.path.append("/home/liupeiyu/smoothquant")
import argparse

from transformers.models.opt.modeling_opt import OPTAttention, OPTDecoderLayer
from transformers.models.llama.modeling_llama import LlamaAttention
# from transformers import GPT2Tokenizer
from transformers import AutoConfig, AutoTokenizer,LlamaTokenizerFast,LlamaTokenizer
from smoothquant.smooth import smooth_lm
from smoothquant.fake_quant import W8A8LinearMPO,W8A8Linear
from datasets import load_dataset,load_from_disk
from tqdm import tqdm
# from modeling_opt import OPTForCausalLM
# MODEL_PATH = '/mnt/liupeiyu/nlp_model/opt-1.3b'
# MODEL_PATH = '/media/public/models/huggingface/facebook/opt-6.7b'
# MODEL_PATH = "/mnt/liupeiyu/nlp_model/opt-350m"

def quantize_model_mpo_kv(model, weight_quant='per_tensor', act_quant='per_tensor', quantize_bmm_input=True,n_bits=8):
    for name, m in model.model.named_modules():
        if isinstance(m, LlamaAttention):
            m.k_proj = W8A8LinearMPO.from_float(
                m.k_proj, weight_quant=weight_quant, act_quant=act_quant, quantize_output=quantize_bmm_input,n_bits=n_bits)
            m.v_proj = W8A8LinearMPO.from_float(
                m.v_proj, weight_quant=weight_quant, act_quant=act_quant, quantize_output=quantize_bmm_input,n_bits=n_bits)
    return model

def quantize_model_kv(model, weight_quant='per_tensor', act_quant='per_tensor', quantize_bmm_input=True,n_bits=8):
    for name, m in model.model.named_modules():
        if isinstance(m, LlamaAttention):
            m.k_proj = W8A8Linear.from_float(
                m.k_proj, weight_quant=weight_quant, act_quant=act_quant, quantize_output=quantize_bmm_input,n_bits=n_bits)
            m.v_proj = W8A8Linear.from_float(
                m.v_proj, weight_quant=weight_quant, act_quant=act_quant, quantize_output=quantize_bmm_input,n_bits=n_bits)
    return model
class Evaluator:
    def __init__(self, dataset, tokenizer, device, limit=1000):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.device = device
        self.limit = limit

        # tokenize the dataset
        def tokenize_function(examples):
            example = self.tokenizer(examples['text'])
            return example

        self.dataset = self.dataset.map(tokenize_function, batched=True)
        self.dataset.set_format(type='torch', columns=['input_ids'])

    @torch.no_grad()
    def evaluate(self, model):
        model.eval()
        # The task is to predict the last word of the input.
        total, hit = 0, 0
        for batch in tqdm(self.dataset):
            if total > self.limit:
                break
            input_ids = batch['input_ids'].to(self.device).unsqueeze(0)
            # input_ids = batch['input_ids'].reshape(1,-1).to(self.device)
            label = input_ids[:, -1]
            outputs = model(input_ids)
            last_token_logits = outputs.logits[:, -2, :]
            pred = last_token_logits.argmax(dim=-1)
            total += label.size(0)
            hit += (pred == label).sum().item()
        acc = hit / total
        return acc

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp",type=str,default="")
    parser.add_argument("--model",type=str,default="")
    parser.add_argument("--n_bits",type=int,default="")
    parser.add_argument("--scales",type=str,default="opt-6.7b")
    parser.add_argument("--limit",type=int,default=1000)

    parser.add_argument("--weight_quant",type=str,default="per_channel")
    parser.add_argument("--act_quant",type=str,default="per_token")
    parser.add_argument("--dataset",type=str,default="lambada")


    args = parser.parse_args()
    MODEL_PATH = args.model

    tokenizer = LlamaTokenizer.from_pretrained(args.model, use_fast=False)
    dataset = load_from_disk("/mnt/liupeiyu/nlp_data/lambada_val")
    evaluator = Evaluator(dataset, tokenizer, 'cuda', args.limit)
    if args.exp == "fp16":
        from transformers.models.llama.modeling_llama import LlamaForCausalLM
        model_fp16 = LlamaForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.float16)
        #### fp16
        acc_fp16 = evaluator.evaluate(model_fp16)
        print(f'Original model (fp16) accuracy: {acc_fp16}') # ori: 0.6393606393606394

    elif args.exp == "base_naive_kv":
        from transformers.models.llama.modeling_llama import LlamaForCausalLM

        model_fp16 = LlamaForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.float16)
        model_fp16.to("cuda:0")
        #### naive
        model_w8a8 = quantize_model_kv(model_fp16,n_bits=args.n_bits,weight_quant=args.weight_quant, act_quant=args.act_quant)
        acc_w8a8 = evaluator.evaluate(model_w8a8)
        print(f'base_naive_kv model (int8) accuracy: {acc_w8a8}') # naive: 0.6093906093906094
    elif args.exp == "base_ours_kv":
        print("Using base ours kv")
        from transformers.models.llama.modeling_llama import LlamaForCausalLM

        model_fp16 = LlamaForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.float16)
        model_fp16.to("cuda:0")
        #### ours
        model_w8a8 = quantize_model_mpo_kv(model_fp16,n_bits=args.n_bits,weight_quant=args.weight_quant, act_quant=args.act_quant)
        acc_w8a8 = evaluator.evaluate(model_w8a8)
        print(f'base_ours_kv model (int8) accuracy: {acc_w8a8}') # naive: 0.6093906093906094
    elif args.exp == "base_bitsandbytes_kv":
        print("Using base bitsandbytes kv")
        from transformers.models.llama.modeling_llama import LlamaForCausalLM
        from transformers.integrations import get_keys_to_not_convert
        _llama_names = [
            lambda l: f"model.layers.{l}.self_attn.q_proj",
            lambda l: f"model.layers.{l}.self_attn.o_proj",
            lambda l: f"model.layers.{l}.post_attention_layernorm",
            lambda l: f"model.layers.{l}.mlp.gate_proj",
            lambda l: f"model.layers.{l}.mlp.up_proj",
            lambda l: f"model.layers.{l}.mlp.down_proj",
            lambda l: f"model.layers.{l}.input_layernorm",
        ]
        names_filter = ['lm_head','model.embed_tokens']
        for layer_id in range(32): # TODO: set the proper num_hidden_layers
            for module_i in _llama_names:
                name_module = module_i(layer_id)
                names_filter.append(name_module)

        bitsandbytes_args = {'load_in_8bit': args.n_bits == 8,
                             'load_in_4bit': args.n_bits == 4,
                             'device_map':"auto",
                             "llm_int8_skip_modules":names_filter}
        model_int = LlamaForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.float16, **bitsandbytes_args)

        acc_w8a8 = evaluator.evaluate(model_int)
        print(f'base_bitsandbytes_kv model (int{args.n_bits}) accuracy: {acc_w8a8}') # naive: 0.6093906093906094