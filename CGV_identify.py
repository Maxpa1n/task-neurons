import torch
import json

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-t","--task", type=str, default="task274_overruling_legal_classification.json")
parser.add_argument("-md", "--mod", type=str, default="GV_trace")
args = parser.parse_args()

task = args.task.split("_")[0]

path = f"/home/user/project/task-specific-neuron/matrix/{task}/{args.mod}.json"

with open(path,"r") as f:
    data = json.load(f)


matrix = torch.tensor(data)
layers,number = matrix.shape
flattened_matrix = matrix.view(-1)


num_elements = flattened_matrix.numel()
top_k = int(num_elements * 0.05)


top_values, top_indices = torch.topk(flattened_matrix, top_k)

rows = top_indices // matrix.size(1)
cols = top_indices % matrix.size(1)


top_indices_2d = torch.stack([
    top_indices // number,  
    top_indices % number   
], dim=1)
# original_indices = torch.stack((rows, cols), dim=1)



output = [[[] for i in range(layers)]]
for i in top_indices_2d:
    l,c = i
    output[0][l].append(c.item())

save_output = [[]]
q = 0
for j in output[0]:
    print(len(j))
    q+=len(j)
    # if len(j)==0:
    #     save_output[0].append(torch.tensor(0))
    # else:
    save_output[0].append(torch.tensor(j).type(torch.int64))
# print(output)
print(q)
torch.save(save_output,f"/home/user/project/task-specific-neuron/activation_mask/{task}/activation_{args.mod}_{task}_pth")
