import argparse
from types import MethodType
import json
import torch
from vllm import LLM, SamplingParams
import os
from utils import data_construct
import torch.nn.functional as F
import random
from jinja2 import Template
from transformers import AutoTokenizer
import pickle
from tqdm import tqdm 


data_path = "/home/user/project/task-specific-neuron/natural-instructions-master/tasks/"
file_list = [
    "task190_snli_classification.json", "task227_clariq_classification.json", 
    "task075_squad1.1_answer_generation.json", "task1645_medical_question_pair_dataset_text_classification.json",
    "task566_circa_classification.json", "task379_agnews_topic_classification.json", 
    "task195_sentiment140_classification.json", "task391_causal_relationship.json"
]


parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", type=str, default="/nfs20t/user/llm/llama-7b-hf-chat")
parser.add_argument("-s", "--shot", type=str, default=5)
parser.add_argument("-t", "--task", type=str, default="task274_overruling_legal_classification.json")
parser.add_argument("-d", "--device", type=str, default="6")
parser.add_argument("-md", "--mod", type=str, default="GV_trace")
parser.add_argument("-ma", "--mask", type=str, default="over_zero")
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.device


task = args.task.split("_")[0]

args = parser.parse_args()
basemodel = args.model.split('/')[-1]
if args.mask!="non-language":
    if args.mod !="LAE":
        args.activation_mask = f"/home/user/project/task-specific-neuron/activation_mask/{task}/activation_{args.mod}_{task}_pth"
    else:
        args.activation_mask = f"/home/user/project/task-specific-neuron/activation_mask/{task}/activation_{args.mod}_{task}_{args.mask}_pth"
else:
    args.activation_mask=""


data_path = os.path.join(data_path, args.task)
with open(data_path, "r") as f:
    data = json.load(f)
    instruction = data["Definition"]
    instance = data["Instances"][:6500]
    data_number = len(instance)
    train, test = instance[:data_number//2], instance[data_number//2:]
    train_message = data_construct(train, instruction, shot=args.shot)
    test_message = data_construct(test, instruction, shot=args.shot)


tokenizer = AutoTokenizer.from_pretrained(args.model)


train_file = f'/home/user/project/task-specific-neuron/data_token/{task}/train_{str(args.shot)}.pkl'
if os.path.exists(train_file):
    with open(train_file, 'rb') as f:
        data = pickle.load(f)
        train_token = data["inputs"]
else:
    train_token = []
    progress_bar = tqdm(total=len(train_message), desc='Train Processing data')
    for i in range(len(train_message)):
        progress_bar.update(1)
        message = train_message[i]
        template_str = tokenizer.default_chat_template
        template = Template(template_str)
        bos_token = ""
        eos_token = ""
        result = template.render(messages=message, bos_token=bos_token, eos_token=eos_token)
        train_token.append(tokenizer.encode(result))
    os.makedirs(f'/home/user/project/task-specific-neuron/data_token/{task}', exist_ok=True)
    with open(train_file, 'wb') as f:
        pickle.dump({"inputs": train_token}, f)


test_file = f'/home/user/project/task-specific-neuron/data_token/{task}/test_{str(args.shot)}.pkl'
if os.path.exists(test_file):
    with open(test_file, 'rb') as f:
        data = pickle.load(f)
        test_token = data["inputs"]
        labels = data["labels"]
else:
    test_token = []
    labels = []
    progress_bar = tqdm(total=len(test_message), desc='Test Processing data')
    for i in range(len(test_message)):
        progress_bar.update(1)
        message = test_message[i]
        prompt, output = message[:-1], message[-1]
        template_str = tokenizer.default_chat_template
        template = Template(template_str)
        bos_token = ""
        eos_token = ""
        result = template.render(messages=prompt, bos_token=bos_token, eos_token=eos_token)
        test_token.append(tokenizer.encode(result))
        labels.append(output["content"])
    progress_bar.close()
    os.makedirs(f'/home/user/project/task-specific-neuron/data_token/{task}', exist_ok=True)
    with open(test_file, 'wb') as f:
        pickle.dump({"inputs": test_token, "labels": labels}, f)


is_llama = bool(args.model.lower().find('llama') >= 0)


model = LLM(model=args.model, tensor_parallel_size=torch.cuda.device_count(), enforce_eager=True)

if args.activation_mask:
    # activation_masks = [torch.load(args.activation_mask)[0]]
    activation_masks = torch.load(args.activation_mask)

else:
    activation_masks = [None]

for activation_mask, mask_lang in zip(activation_masks, [args.mask]):
    if activation_mask:
        def factory(mask):

            def llama_forward(self, x):
                gate_up, _ = self.gate_up_proj(x)  # b, l, 2i
                i = gate_up.size(-1)
                activation = F.silu(gate_up[:, :, : i // 2])
                
                # mask_tensor = torch.zeros(activation.size(-1), device=activation.device)
                # mask_tensor[mask] = 1


                mask_tensor = torch.ones(activation.size(-1), device=activation.device)
                mask_tensor[mask] = 0

                activation *= mask_tensor
                x = activation * gate_up[:, :, i // 2:]

                x, _ = self.down_proj(x)
                return x

            return llama_forward


        for i, layer_mask in enumerate(activation_mask):
            obj = model.llm_engine.driver_worker.model_runner.model.model.layers[i].mlp
            obj.forward = MethodType(factory(layer_mask.to('cuda').type(torch.int64)), obj)

test_token=test_token
outputs = model.generate(prompt_token_ids=test_token, sampling_params=SamplingParams(max_tokens=20,temperature=0,top_p=1,stop="[INST]"))
output_folder = f"results/{basemodel}/{task}/"
os.makedirs(output_folder, exist_ok=True)


if activation_mask:
    # output_file = f"{output_folder}/perturb_{args.mod}_{args.shot}.jsonl"
    output_file = f"{output_folder}/perturb_{args.mod}_{args.shot}.jsonl"
else:
    output_file = f"{output_folder}/{task}-{args.shot}-shot-1-token-chat.jsonl"

with open(output_file, "w", encoding="utf8") as f:

    data_output = [{"prompt": tokenizer.decode(output.prompt_token_ids),
                    "pred": output.outputs[0].text,
                    "label": l}
                   for l, output in zip(labels, outputs)]

    corect = 0
    for j in data_output:
        j["pred"] = j["pred"].strip()
        # if j["pred"] in j["label"] or  j["label"] in j["pred"]:
        if j["pred"].lower() == j["label"].lower():
            corect+=1
        f.write(json.dumps(j,ensure_ascii=False) + "\n")
    f.write(json.dumps({"num":str(len(data_output)),"correct":corect,"acc":corect/len(data_output)},ensure_ascii=False) + "\n")
    print(f"num:{str(len(data_output))}, right:{str(corect)}, acc:{str(corect/len(data_output))}")

with open(f"new_result1111111.jsonl","a") as f:
    f.write("----------------------------\n")
    f.write(f"option:{task},mod:{args.mod},task:{args.mask}\n")
    f.write(json.dumps({"num":str(len(data_output)),"correct":corect,"acc":corect/len(data_output)},ensure_ascii=False) + "\n")