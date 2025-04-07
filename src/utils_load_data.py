# %%
import json
import torch
import einops

BASE_DIR = "../data"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_grad_enabled(False)
folder = BASE_DIR

def load_res_data(index, group_size=4, groups_to_load=2):
    file_path = f"{folder}/res_tensors/res_data_{index:03d}.pt"
    data = torch.load(file_path, map_location="cpu", weights_only=False)
    data = torch.cat(data, dim=2)  # Concatenate list of tensors
    data = data.squeeze(0)
    data = data[1:, :, :]  # Remove first layer
    data = einops.rearrange(data, 'layers samples dim -> samples layers dim')
    data = einops.rearrange(data, 'samples (layersg g) dim -> samples layersg (g dim)', g=group_size)
    data = einops.rearrange(data[:, -groups_to_load:], 'samples layersg gdim -> samples (layersg gdim)')
    return data.float()

def load_embeds(index):
    file_path = f"{folder}/sonar_embeds/embeds_{index:03d}.pt"
    data = torch.load(file_path, map_location="cpu", weights_only=False)
    data = torch.cat(data, dim=0)
    return data.float()


from itertools import chain
def flatten(nested_list):
    return list(chain.from_iterable(nested_list))

def load_split_paragraphs(index):
    file_path = f"{folder}/split_paragraphs/paragraphs_{index:03d}.json"
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def load_split_paragraphs_inputs(index):
    file_path = f"{folder}/split_input_paragraphs/paragraphs_{index:03d}.json"
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def load_paragraphs():
    file_path = "/workspace/SPAR/gen-dataset/new_split_dataset.jsonl"
    with open(file_path, 'r') as f:
        data = [json.loads(line) for line in f]
    splits = flatten([x["split"][1:] for x in data])
    return splits

def load_paragraphs_with_context(zipped=True):
    file_path = "/workspace/SPAR/gen-dataset/new_split_dataset.jsonl"
    with open(file_path, 'r') as f:
        data = [json.loads(line) for line in f]

    inputs  = flatten([x["split"][:-1] for x in data])
    outputs = flatten([x["split"][1:] for x in data])
    if zipped:
        return list(zip(inputs, outputs))
    else:
        return inputs, outputs

def load_full_contexts():
    file_path = "/workspace/SPAR/gen-dataset/new_split_dataset.jsonl"
    with open(file_path, 'r') as f:
        data = [json.loads(line) for line in f]

    contexts = []
    for item in data:
        tokens = item["split"][:-1]  # Exclude the final element
        # Build cumulative contexts by joining token prefixes with "".join()
        for i in range(1, len(tokens) + 1):
            context = "".join(tokens[:i])
            contexts.append(context)
    return contexts

if __name__ == "__main__":
    print(load_res_data(0).shape)
    print(load_embeds(0).shape)
    print(len(load_paragraphs()))
    print(len(load_split_paragraphs(0)))
    print(len(load_split_paragraphs_inputs(0)))

    assert load_split_paragraphs(0)[0] == load_split_paragraphs_inputs(0)[1]
# %%
