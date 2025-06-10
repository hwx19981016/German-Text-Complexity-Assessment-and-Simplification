# captum_process.py

import torch
import numpy as np
import re
from captum.attr import LayerConductance
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if torch.__version__ >= '1.7.0':
    norm_fn = torch.linalg.norm
else:
    norm_fn = torch.norm

def construct_input_ref_pair(text, tokenizer, ref_token_id, sep_token_id, cls_token_id):
    text_ids = tokenizer.encode(text, add_special_tokens=False)
    input_ids = [cls_token_id] + text_ids + [sep_token_id]
    ref_input_ids = [cls_token_id] + [ref_token_id] * len(text_ids) + [sep_token_id]
    return torch.tensor([input_ids], device=device), torch.tensor([ref_input_ids], device=device), len(text_ids)

def construct_input_ref_token_type_pair(input_ids, sep_ind=0):
    seq_len = input_ids.size(1)
    token_type_ids = torch.tensor([[0 if i <= sep_ind else 1 for i in range(seq_len)]], device=device)
    ref_token_type_ids = torch.zeros_like(token_type_ids, device=device)
    return token_type_ids, ref_token_type_ids

def construct_input_ref_pos_id_pair(input_ids):
    seq_length = input_ids.size(1)
    position_ids = torch.arange(seq_length, dtype=torch.long, device=device)
    ref_position_ids = torch.zeros(seq_length, dtype=torch.long, device=device)
    position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
    ref_position_ids = ref_position_ids.unsqueeze(0).expand_as(input_ids)
    return position_ids, ref_position_ids

def construct_attention_mask(input_ids):
    return torch.ones_like(input_ids)

def summarize_attributions(attributions):
    attributions = attributions.sum(dim=-1).squeeze(0)
    attributions = attributions / norm_fn(attributions)
    return attributions

def topFromValue(xticklabels, values):
    merged_words = []
    merged_values = []
    temp_word = ""
    temp_value = 0
    for word, value in zip(xticklabels, values):
        if word.startswith('##'):
            temp_word += word[2:]
            temp_value += value
        else:
            if temp_word:
                merged_words.append(temp_word)
                merged_values.append(temp_value)
            temp_word = word
            temp_value = value
    if temp_word:
        merged_words.append(temp_word)
        merged_values.append(temp_value)
    filtered = [(w, v) for w, v in zip(merged_words, merged_values) if re.search(r'[\w\u4e00-\u9fff]', w)]
    if not filtered:
        return [], []
    merged_words, merged_values = zip(*filtered)
    top_k = max(3, int(len(merged_values) * 0.3))
    top_five_indices = sorted(range(len(merged_values)), key=lambda i: merged_values[i], reverse=True)[:top_k]
    top_five_words = [merged_words[i] for i in top_five_indices]
    top_five_values = [merged_values[i] for i in top_five_indices]
    return top_five_words, top_five_values

def captumProcess(text, tokenizer, model, ref_token_id, sep_token_id, cls_token_id):
    input_ids, ref_input_ids, sep_id = construct_input_ref_pair(text, tokenizer, ref_token_id, sep_token_id, cls_token_id)
    token_type_ids, ref_token_type_ids = construct_input_ref_token_type_pair(input_ids, sep_id)
    position_ids, ref_position_ids = construct_input_ref_pos_id_pair(input_ids)
    attention_mask = construct_attention_mask(input_ids)
    indices = input_ids[0].detach().tolist()
    all_tokens = tokenizer.convert_ids_to_tokens(indices)

    def squad_pos_forward_func2(input_emb, attention_mask=None, position=0):
        pred = model(inputs_embeds=input_emb, attention_mask=attention_mask)
        pred = pred[position]
        return pred.max(1).values

    def construct_whole_bert_embeddings(input_ids, ref_input_ids):
        input_embeddings = model.bert.embeddings(input_ids, token_type_ids=token_type_ids, position_ids=position_ids)
        ref_input_embeddings = model.bert.embeddings(ref_input_ids, token_type_ids=ref_token_type_ids, position_ids=ref_position_ids)
        return input_embeddings, ref_input_embeddings

    input_embeddings, ref_input_embeddings = construct_whole_bert_embeddings(input_ids, ref_input_ids)

    layer_attrs_start = []
    for i in range(model.config.num_hidden_layers):
        lc = LayerConductance(squad_pos_forward_func2, model.bert.encoder.layer[i])
        layer_attributions_start = lc.attribute(inputs=input_embeddings, baselines=ref_input_embeddings)
        layer_attrs_start.append(summarize_attributions(layer_attributions_start).cpu().detach().tolist())

    values = np.mean(np.array(layer_attrs_start), axis=0)
    xticklabels = all_tokens
    top_words, top_values = topFromValue(xticklabels, values)
    return top_words, top_values
