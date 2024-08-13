import torch
import sys
sys.path.append("/home/liupeiyu/smoothquant")
import argparse

from transformers.models.opt.modeling_opt import OPTAttention, OPTDecoderLayer, OPTForCausalLM
from transformers import GPT2Tokenizer
from smoothquant.smooth import smooth_lm,smooth_lm_kv
from smoothquant.fake_quant import W8A8LinearMPO,W8A8Linear
from datasets import load_dataset,load_from_disk
from tqdm import tqdm

DATA_DICT = {
    'lambada': "/mnt/liupeiyu/nlp_data/lambada_val",
    'cnn_dailymail': "/mnt/liupeiyu/nlp_data/cnn_dailymail",
}
def get_parameter_number(net):
    '''
    :param net: model class
    :return: params statistics
    '''
    total_num = sum(p.numel() for p in net.parameters())/1000/1000
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)/1000/1000
    return {'Total(M)': total_num, 'Trainable(M)': trainable_num}
def quantize_model_mpo(model, weight_quant='per_tensor', act_quant='per_tensor', quantize_bmm_input=True,n_bits=8):
    for name, m in model.model.named_modules():
        if isinstance(m, OPTDecoderLayer):
            pass
            m.fc1 = W8A8LinearMPO.from_float(m.fc1, weight_quant=weight_quant, act_quant=act_quant,n_bits=n_bits)
            m.fc2 = W8A8LinearMPO.from_float(m.fc2, weight_quant=weight_quant, act_quant=act_quant,n_bits=n_bits)
        elif isinstance(m, OPTAttention):
            # Her we simulate quantizing BMM inputs by quantizing the output of q_proj, k_proj, v_proj
            m.q_proj = W8A8LinearMPO.from_float(
                m.q_proj, weight_quant=weight_quant, act_quant=act_quant, quantize_output=quantize_bmm_input,n_bits=n_bits)
            m.k_proj = W8A8LinearMPO.from_float(
                m.k_proj, weight_quant=weight_quant, act_quant=act_quant, quantize_output=quantize_bmm_input,n_bits=n_bits)
            m.v_proj = W8A8LinearMPO.from_float(
                m.v_proj, weight_quant=weight_quant, act_quant=act_quant, quantize_output=quantize_bmm_input,n_bits=n_bits)
            m.out_proj = W8A8LinearMPO.from_float(m.out_proj, weight_quant=weight_quant, act_quant=act_quant,n_bits=n_bits)
    return model

def quantize_model(model, weight_quant='per_tensor', act_quant='per_tensor', quantize_bmm_input=True,n_bits=8):
    for name, m in model.model.named_modules():
        if isinstance(m, OPTDecoderLayer):
            m.fc1 = W8A8Linear.from_float(m.fc1, weight_quant=weight_quant, act_quant=act_quant,n_bits=n_bits)
            m.fc2 = W8A8Linear.from_float(m.fc2, weight_quant=weight_quant, act_quant=act_quant,n_bits=n_bits)
        elif isinstance(m, OPTAttention):
            # Her we simulate quantizing BMM inputs by quantizing the output of q_proj, k_proj, v_proj
            m.q_proj = W8A8Linear.from_float(
                m.q_proj, weight_quant=weight_quant, act_quant=act_quant, quantize_output=quantize_bmm_input,n_bits=n_bits)
            m.k_proj = W8A8Linear.from_float(
                m.k_proj, weight_quant=weight_quant, act_quant=act_quant, quantize_output=quantize_bmm_input,n_bits=n_bits)
            m.v_proj = W8A8Linear.from_float(
                m.v_proj, weight_quant=weight_quant, act_quant=act_quant, quantize_output=quantize_bmm_input,n_bits=n_bits)
            m.out_proj = W8A8Linear.from_float(m.out_proj, weight_quant=weight_quant, act_quant=act_quant,n_bits=n_bits)
    return model
def quantize_model_mpo_kv(model, weight_quant='per_tensor', act_quant='per_tensor', quantize_bmm_input=True,n_bits=8):
    for name, m in model.model.named_modules():
        if isinstance(m, OPTAttention):
            m.k_proj = W8A8LinearMPO.from_float(
                m.k_proj, weight_quant=weight_quant, act_quant=act_quant, quantize_output=quantize_bmm_input,n_bits=n_bits)
            m.v_proj = W8A8LinearMPO.from_float(
                m.v_proj, weight_quant=weight_quant, act_quant=act_quant, quantize_output=quantize_bmm_input,n_bits=n_bits)
    return model
def quantize_model_kv(model, weight_quant='per_tensor', act_quant='per_tensor', quantize_bmm_input=True,n_bits=8):
    for name, m in model.model.named_modules():
        if isinstance(m, OPTAttention):
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
            if 'text' in examples:
                example = self.tokenizer(examples['text'])
            else:
                example = self.tokenizer(examples['article'])
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
    parser.add_argument("--bitsandbytes_args", type=str, default="") # choose from ['load_int_8bit','load_in_4bit']


    args = parser.parse_args()
    MODEL_PATH = args.model

    tokenizer = GPT2Tokenizer.from_pretrained(MODEL_PATH)
    dataset = load_from_disk("/mnt/liupeiyu/nlp_data/lambada_val")
    evaluator = Evaluator(dataset, tokenizer, 'cuda', args.limit)
    if args.exp == "fp16":
        #### fp16
        model_fp16 = OPTForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.float16)
        model_fp16.to("cuda:0")
        acc_fp16 = evaluator.evaluate(model_fp16)
        print(f'Original model (fp16) accuracy: {acc_fp16}') # ori: 0.6393606393606394

    elif args.exp == "base_ours_kv":
        print("Using base ours kv")
        from transformers.models.opt.modeling_opt import OPTForCausalLM

        model_fp16 = OPTForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.float16)
        model_fp16.to("cuda:0")
        #### ours
        model_w8a8 = quantize_model_mpo_kv(model_fp16,n_bits=args.n_bits,weight_quant=args.weight_quant, act_quant=args.act_quant)
        acc_w8a8 = evaluator.evaluate(model_w8a8)
        print(f'base_ours_kv model (int{args.n_bits}) accuracy: {acc_w8a8}') # naive: 0.6093906093906094
    elif args.exp == "base_naive_kv":
        from transformers.models.opt.modeling_opt import OPTForCausalLM

        model_fp16 = OPTForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.float16)
        model_fp16.to("cuda:0")
        #### naive
        model_w8a8 = quantize_model_kv(model_fp16,n_bits=args.n_bits,weight_quant=args.weight_quant, act_quant=args.act_quant)
        acc_w8a8 = evaluator.evaluate(model_w8a8)
        print(f'base_naive_kv model (int{args.n_bits}) accuracy: {acc_w8a8}') # naive: 0.6093906093906094
    elif args.exp == "base_bitsandbytes_kv":
        print("Using base ours kv")
        from transformers.models.opt.modeling_opt import OPTForCausalLM
        
        _opt_names = [
            lambda l: f"model.decoder.layers.{l}.self_attn.q_proj",
            lambda l: f"model.decoder.layers.{l}.self_attn.out_proj",
            lambda l: f"model.decoder.layers.{l}.fc1",
            lambda l: f"model.decoder.layers.{l}.fc2",
            lambda l: f"model.decoder.layers.{l}.self_attn_layer_norm",
            lambda l: f"model.decoder.layers.{l}.final_layer_norm",
        ]
        names_filter = ['lm_head','model.decoder.embed_tokens']
        for layer_id in range(48): # TODO: set the proper num_hidden_layers
            for module_i in _opt_names:
                name_module = module_i(layer_id)
                names_filter.append(name_module)

        print(f"# names filter: {len(names_filter)}")
        bitsandbytes_args = {'load_in_8bit': args.n_bits == 8,
                             'load_in_4bit': args.n_bits == 4,
                             'device_map':"auto",
                             "llm_int8_skip_modules":names_filter}
        
        model_int = OPTForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.float16, **bitsandbytes_args)
        #### ours
        # model_w8a8 = quantize_model_mpo_kv(model_fp16,n_bits=args.n_bits,weight_quant=args.weight_quant, act_quant=args.act_quant)
        acc_w8a8 = evaluator.evaluate(model_int)
        print(f'base_bitsandbytes_kv model (int{args.n_bits}) accuracy: {acc_w8a8}') # naive: 0.6093906093906094
    elif args.exp == "smoothquant":
        from transformers.models.opt.modeling_opt import OPTForCausalLM

        model_fp16 = OPTForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.float16)
        model_fp16.to("cuda:0")
        if '6.7' in MODEL_PATH:
            act_scales = torch.load('/home/liupeiyu/smoothquant/act_scales/opt-6.7b.pt')
        else:
            act_scales = torch.load('/home/liupeiyu/smoothquant/act_scales/opt-1.3b.pt')
        smooth_lm_kv(model_fp16, act_scales, 0.5)
        # smooth_lm(model_fp16, act_scales, 0.5)
        model_smoothquant_w8a8 = quantize_model_kv(model_fp16,n_bits=args.n_bits,weight_quant=args.weight_quant, act_quant=args.act_quant)
        acc_smoothquant_w8a8 = evaluator.evaluate(model_smoothquant_w8a8)
        print(f'base_naive_kv model (int{args.n_bits}) accuracy: {acc_smoothquant_w8a8}') # naive: 0.6093906093906094
    elif args.exp == "naive_kv": # 这个不能说明kvcache的量化，因为只生成下一个单词，这个时候不会用到kvcache，已经修改
        raise NotImplementedError
        from modeling_opt import OPTForCausalLM

        model_fp16 = OPTForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.float16)
        model_fp16.to("cuda:0")
        acc_w8a8 = evaluator.evaluate(model_fp16)
        print(f'smart cache naive_kv model (int{args.n_bits}) accuracy: {acc_w8a8}') # naive: 0.6093906093906094
    else:
        raise NotImplementedError
        ### baseline
        model = OPTForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.float16)
        model.to("cuda:0")
        assert args.scales != ""
        act_scales = torch.load(f'../act_scales/{args.scales}.pt')
        smooth_lm(model, act_scales, 0.5)
        model_smoothquant_w8a8 = quantize_model(model,n_bits=args.n_bits,weight_quant='per_channel', act_quant='per_token')
        acc_smoothquant_w8a8 = evaluator.evaluate(model_smoothquant_w8a8)
        print(f'SmoothQuant W8A8 quantized model accuracy: {acc_smoothquant_w8a8}')