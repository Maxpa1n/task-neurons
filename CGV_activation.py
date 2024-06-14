from transformers import LlamaForCausalLM, AutoTokenizer
import torch
from types import MethodType
import os
import torch.nn.functional as F
from utils import data_construct,find_all_sublists
from transformers import AutoTokenizer
import pickle
from tqdm import tqdm 
import os
import random
from transformers import AutoTokenizer
# from utils.instructions import INSTRUCTIONS
from jinja2 import Template
import pickle
import numpy as np
from jinja2 import Template
import argparse
import  torch.nn as nn
import json
parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", type=str, default="/nfs20t/songran/llm/llama-7b-hf-chat")
parser.add_argument("-t","--task", type=str, default="task274_overruling_legal_classification.json")
parser.add_argument("-d","--device",type=str, default="1")
parser.add_argument("-st","--shot",type=int, default=5)
parser.add_argument("-md", "--mod", type=str, default="GV_trace_latest_up")
args = parser.parse_args()
basemodel = args.model.split('/')[-1]

sub_squence = {
    "[INST]":[518, 25580, 29962],
    "[\INST]":[518, 29914, 25580, 29962],
    "<<SYS>>":[3532, 14816, 29903, 6778],
    "<<\SYS>>":[529, 829, 14816, 29903, 6778]
}
sub_squence_list = [[518, 25580, 29962],[518, 29914, 25580, 29962],[3532, 14816, 29903, 6778],[529, 829, 14816, 29903, 6778]]

os.environ["CUDA_VISIBLE_DEVICES"] = args.device
data_path = "/home/songran/project/task-specific-neuron/natural-instructions-master/tasks/"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# 获取任务名称
task = args.task.split("_")[0]

# 加载数据并拆分为训练集和测试集
data_path = os.path.join(data_path, args.task)
with open(data_path, "r") as f:
    data = json.load(f)
    instruction = data["Definition"]
    instance = data["Instances"]
    data_number = len(instance)
    train, test = instance[:data_number//2], instance[data_number//2:]
    train_message = data_construct(train, instruction, shot=args.shot)
    test_message = data_construct(test, instruction, shot=args.shot)

# 加载tokenizer
tokenizer = AutoTokenizer.from_pretrained(args.model)

# 处理训练数据并保存token
if "trace"  not in args.mod:
    train_file = f'/home/songran/project/task-specific-neuron/data_token/{task}/train_{str(args.shot)}.pkl'
else:
    train_file = f'/home/songran/project/task-specific-neuron/data_token/{task}/train_trace_{str(args.shot)}.pkl'
if os.path.exists(train_file):
    with open(train_file, 'rb') as f:
        data = pickle.load(f)
        train_token = data["inputs"]
        if "indexs" in list(data.keys()):
            indexs = data["indexs"]
else:
    train_token = []
    indexs = []
    progress_bar = tqdm(total=len(train_message), desc='Train Processing data')
    for i in range(len(train_message)):
        progress_bar.update(1)
        message = train_message[i]
        template_str = tokenizer.default_chat_template
        template = Template(template_str)
        bos_token = ""
        eos_token = ""
        result = template.render(messages=message, bos_token=bos_token, eos_token=eos_token).replace("<spe>"," ")
        if "trace" in args.mod:
            input_ids = tokenizer.encode(result)
            index_start = find_all_sublists(input_ids,sub_squence_list[0])
            index_1 = find_all_sublists(input_ids,sub_squence_list[1])
            track_index = index_start+index_1
            print(len(index_start),len(index_1))
            lat_list = [item for sublist in track_index for item in sublist]
            indexs.append(sorted(lat_list))
        train_token.append(tokenizer.encode(result))
    os.makedirs(f'/home/songran/project/task-specific-neuron/data_token/{task}', exist_ok=True)
    with open(train_file, 'wb') as f:
        pickle.dump({"inputs": train_token,"indexs":indexs}, f)

# 处理测试数据并保存token
test_file = f'/home/songran/project/task-specific-neuron/data_token/{task}/test_{str(args.shot)}.pkl'
if os.path.exists(test_file):
    with open(test_file, 'rb') as f:
        data = pickle.load(f)
        test_token = data["inputs"]
        labels = data["labels"]
else:
    test_token = []
    labels = []
    progress_bar = tqdm(total=len(train_message), desc='Test Processing data')
    for i in range(len(train_message)):
        progress_bar.update(1)
        message = train_message[i]
        prompt, output = message[:-1], message[1]
        template_str = tokenizer.default_chat_template
        template = Template(template_str)
        bos_token = ""
        eos_token = ""
        result = template.render(messages=prompt, bos_token=bos_token, eos_token=eos_token)
        test_token.append(tokenizer.encode(result))
        labels.append(output["content"])
    progress_bar.close()
    os.makedirs(f'/home/songran/project/task-specific-neuron/data_token/{task}', exist_ok=True)
    with open(test_file, 'wb') as f:
        pickle.dump({"inputs": test_token, "labels": labels}, f)


model = LlamaForCausalLM.from_pretrained(args.model,torch_dtype=torch.bfloat16)
model.to(device)
model.train()
criterion = nn.CrossEntropyLoss(reduction="none")
out_data = [[0]*11008]*32
ss=0
progress_bar = tqdm(total=len(train_token), desc='Getting data')
for input_ids,index in zip(train_token, indexs):
    progress_bar.update(1)
    # print(len(input_ids))
    if len(input_ids)>1300:
        ss+=1
        # print(ss)
        continue
    input_index = [i-1 for i in index]
    label_token = [input_ids[i] for i in index]

    input_ids = torch.tensor(input_ids,dtype=torch.int64).unsqueeze(0).to(device)
    label_token = torch.tensor(label_token,dtype=torch.int64).to(device)

    output = model(input_ids)
    loss1 = criterion(output.logits[0, input_index[:28], :], label_token[:28])
    loss2 = criterion(output.logits[0, input_index[28:], :], label_token[28:])

    # 计算平均损失
    # loss = loss1.mean()*0.0001 + loss2.mean()
    loss = loss1.mean() + loss2.mean()
    model.zero_grad()

    loss.backward()
    # print(loss.item())
    for name, param in model.named_parameters():
        if param.grad is not None and "up_proj" in name:
            # print(f'name: {name}, grad:\n{param.grad.shape}')
            layer = int(name.split(".")[2])
            grad = torch.sum(param.grad,dim=1).cpu().tolist()
            out_data[layer] =  [abs(a) + b for a, b in zip(grad, out_data[layer])]


with open(f"/home/songran/project/task-specific-neuron/matrix/{task}/{args.mod}.json","w") as f:
    json.dump(out_data,f)