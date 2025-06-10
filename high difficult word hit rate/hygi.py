import hashlib
import os
import sys
from os.path import realpath, join, dirname
import re
import numpy as np
import pandas as pd
import tensorflow as tf
import torch
from keras.layers import Dense, Input,InputLayer
from keras.models import Sequential
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.python.keras.models import load_model
import torch
from torch import nn, optim
from sklearn.preprocessing import StandardScaler
import numpy as np
import os
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    TrainingArguments,
    AutoModelForSequenceClassification,
    set_seed,
)
sys.path.insert(0, realpath(join(dirname('captum_t.py'), '..')))

from util.helpers import (
    compute_metrics,
    load_dataset_with_features, get_hugging_face_name, TCCDataset, RegressionTrainer,
    compute_metrics_for_regression, OptimizedESCallback
)
import json
with open("difficult_words.json", "r", encoding="utf-8") as f:
    difficult_dict = json.load(f)

    
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


MODEL_NAME = 'gbert'



gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
EXPERIMENT_NAME = f'ensemble_{MODEL_NAME}'
EXPERIMENT_DIR = f'../autodl-tmp/cache/{EXPERIMENT_NAME}'


def custom_forward(inputs):
    preds = predict(inputs)
    print(preds)
    return preds

def predict(inputs):
    #print('model(inputs): ', model(inputs))
    return model(inputs)[0]
def construct_input_ref_pair(text, ref_token_id, sep_token_id, cls_token_id):
    text_ids = tokenizer.encode(text, add_special_tokens=False)
    # construct input token ids
    input_ids = [cls_token_id] + text_ids + [sep_token_id]
    # construct reference token ids 
    ref_input_ids = [cls_token_id] + [ref_token_id] * len(text_ids) + [sep_token_id]

    return torch.tensor([input_ids], device=device), torch.tensor([ref_input_ids], device=device), len(text_ids)

def construct_input_ref_token_type_pair(input_ids, sep_ind=0):
    seq_len = input_ids.size(1)
    token_type_ids = torch.tensor([[0 if i <= sep_ind else 1 for i in range(seq_len)]], device=device)
    ref_token_type_ids = torch.zeros_like(token_type_ids, device=device)# * -1
    return token_type_ids, ref_token_type_ids

def construct_input_ref_pos_id_pair(input_ids):
    seq_length = input_ids.size(1)
    position_ids = torch.arange(seq_length, dtype=torch.long, device=device)
    # we could potentially also use random permutation with `torch.randperm(seq_length, device=device)`
    ref_position_ids = torch.zeros(seq_length, dtype=torch.long, device=device)

    position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
    ref_position_ids = ref_position_ids.unsqueeze(0).expand_as(input_ids)
    return position_ids, ref_position_ids
    
def construct_attention_mask(input_ids):
    return torch.ones_like(input_ids)


tokenizer = AutoTokenizer.from_pretrained(get_hugging_face_name(MODEL_NAME))
ref_token_id = tokenizer.pad_token_id # A token used for generating token reference
sep_token_id = tokenizer.sep_token_id # A token used as a separator between question and text and it is also added to the end of the text.
cls_token_id = tokenizer.cls_token_id # A token used for prepending to the concatenated question-text word sequence

from transformers import BertTokenizer, BertForSequenceClassification, BertConfig
from captum.attr import visualization as viz
from captum.attr import IntegratedGradients, LayerConductance, LayerIntegratedGradients
from captum.attr import configure_interpretable_embedding_layer, remove_interpretable_embedding_layer
import torch
import matplotlib.pyplot as plt
import captum
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = AutoModelForSequenceClassification.from_pretrained(
                f'{EXPERIMENT_DIR}/models/{MODEL_NAME}/3221b14132ce1811b5cd73c4108382bd480e30138c2fadf6908e413349c00d49_gbert-large', local_files_only=True, num_labels=1
            ).to(device)

lig = LayerIntegratedGradients(custom_forward, model.bert.embeddings)

text =  "Kochbücher propagierten ab 1651 die „Haute Cuisine“, die von den ersten Restaurants fortan adaptiert wurde." #"The movie was one of those amazing movies"#"The movie was one of those amazing movies you can not forget"
##saved_act = None
def save_act(module, inp, out):
  #global saved_act
  saved_act = out
  return saved_act

hook = model.bert.embeddings.register_forward_hook(save_act)

def construct_whole_bert_embeddings(input_ids, ref_input_ids, \
                                    token_type_ids=None, ref_token_type_ids=None, \
                                    position_ids=None, ref_position_ids=None):
    input_embeddings = model.bert.embeddings(input_ids, token_type_ids=token_type_ids, position_ids=position_ids)
    ref_input_embeddings = model.bert.embeddings(ref_input_ids, token_type_ids=ref_token_type_ids, position_ids=ref_position_ids)
    
    return input_embeddings, ref_input_embeddings
def squad_pos_forward_func2(input_emb, attention_mask=None, position=0):
    pred = model(inputs_embeds=input_emb, attention_mask=attention_mask, )
    pred = pred[position]
    return pred.max(1).values
#def summarize_attributions(attributions):
#    attributions = attributions.sum(dim=-1).squeeze(0)
#    attributions = attributions / torch.norm(attributions)
#    return attributions
if torch.__version__ >= '1.7.0':
    norm_fn = torch.linalg.norm
else:
    norm_fn = torch.norm
def summarize_attributions(attributions):
    attributions = attributions.sum(dim=-1).squeeze(0)
    attributions = attributions / norm_fn(attributions)
    return attributions
def topFromValue(xticklabels, values):
    # 合并token. '##'开头的单词
    # return top words and values.
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
                temp_word = ""
                temp_value = 0
            temp_word = word
            temp_value = value
    
    if temp_word:
        merged_words.append(temp_word)
        merged_values.append(temp_value)

    filtered = [(w, v) for w, v in zip(merged_words, merged_values)
                if re.search(r'[\w\u4e00-\u9fff]', w)]  # Filtering symbol

    if not filtered: 
        return [], []

    merged_words, merged_values = zip(*filtered)
    
    
    top_k = max(3, int(len(merged_values) * 0.3))  # At least three, but the number of difficult words is determined based on 30% of the total number of words.
    
    top_five_indices = sorted(range(len(merged_values)), key=lambda i: merged_values[i], reverse=True)[:top_k]
    top_five_words = [merged_words[i] for i in top_five_indices]
    top_five_values = [merged_values[i] for i in top_five_indices]
    
    #print("Top words :", top_five_words)
    #print("contribution values:", top_five_values)
    return top_five_words, top_five_values

def captumProcess(text,tokenizer, model, ref_token_id, sep_token_id, cls_token_id):
    # 对text进行embedding
    input_ids, ref_input_ids, sep_id = construct_input_ref_pair(text, ref_token_id, sep_token_id, cls_token_id)
    token_type_ids, ref_token_type_ids = construct_input_ref_token_type_pair(input_ids, sep_id)
    position_ids, ref_position_ids = construct_input_ref_pos_id_pair(input_ids)
    attention_mask = construct_attention_mask(input_ids)
    
    indices = input_ids[0].detach().tolist()
    all_tokens = tokenizer.convert_ids_to_tokens(indices)
    layer_attrs_start = []
    layer_attrs_end = []
    
    
    
    layer_attrs_start_dist = []
    layer_attrs_end_dist = []
    
    input_embeddings, ref_input_embeddings = construct_whole_bert_embeddings(input_ids, ref_input_ids, \
                                             token_type_ids=token_type_ids, ref_token_type_ids=ref_token_type_ids, \
                                             position_ids=position_ids, ref_position_ids=ref_position_ids)
    
    for i in range(model.config.num_hidden_layers):
        lc = LayerConductance(squad_pos_forward_func2, model.bert.encoder.layer[i])
        layer_attributions_start = lc.attribute(inputs=input_embeddings, baselines=ref_input_embeddings)
        layer_attributions_end = lc.attribute(inputs=input_embeddings, baselines=ref_input_embeddings)
        layer_attrs_start.append(summarize_attributions(layer_attributions_start).cpu().detach().tolist())
        layer_attrs_end.append(summarize_attributions(layer_attributions_end).cpu().detach().tolist())
    values = np.mean(np.array(layer_attrs_start),axis=0) #
    xticklabels=all_tokens
    top_words, top_values = topFromValue(xticklabels, values)
    #print(top_words)
    return top_words, top_values

    


def batch_hit_percent(sentences, tokenizer, model, ref_token_id, sep_token_id, cls_token_id, difficult_dict):
    hit_percent_list = []
    for text in tqdm(sentences):
        try:
            top_words, _ = captumProcess(text, tokenizer, model, ref_token_id, sep_token_id, cls_token_id)
            if not top_words:
                continue
            hit = sum(1 for word in top_words if difficult_dict.get(word, 0) == 1)
            hit_percent = hit / len(top_words)
            hit_percent_list.append(hit_percent)
        except Exception as e:
            continue
    if hit_percent_list:
        avg_hit = np.mean(hit_percent_list)
        #print(f"Avg Hit Percent (difficulty word focus rate): {avg_hit:.3f}")
        return avg_hit
    else:
        print("No valid samples for hit percent.")
        return 0.0

import json
from easse.sari import corpus_sari
# from easse.samsa import sentence_samsa
from easse.fkgl import corpus_fkgl
from easse.bleu import corpus_bleu
from easse.quality_estimation import corpus_quality_estimation
from easse.compression import corpus_f1_token
#from eassede.easse.textstat_metrics import corpus_fre
from sacrebleu.metrics import BLEU
from rouge_score import rouge_scorer
from sacremoses import MosesTokenizer
import language_tool_python
from textstat import flesch_reading_ease
import torch
import nltk
import evaluate
import numpy as np

nltk.download('punkt_tab')

# Load metrics
metric_bertscore = evaluate.load("bertscore")
metric_bleu = evaluate.load("bleu")
metric_rouge = evaluate.load("rouge")

tool = language_tool_python.LanguageTool("de", remote_server='http://localhost:8081')
#tool = language_tool_python.LanguageTool('de-DE', host='localhost', port=8081)
tokenizerG = MosesTokenizer(lang="de")


def calculate_ppl(text, model, tokenizer):
    model = model.cpu()
    encodings = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**encodings, labels=encodings["input_ids"])
    loss = outputs.loss
    return torch.exp(loss).item()

'''
def grammar_check(texts):
    scores = []
    for text in texts:
        matches = tool.check(text)
        scores.append(len(matches))
    return np.mean(scores)
'''
def grammar_check(texts, max_errors=0):
    scores = []
    keep_mask = []
    for text in texts:
        matches = tool.check(text)
        num_errors = len(matches)
        scores.append(num_errors)
        keep_mask.append(num_errors <= max_errors)
    #    print(num_errors <= max_errors, num_errors, max_errors)
    #print(keep_mask)
    return np.mean(scores), keep_mask


def calculate_simplification(texts):
    lengths = [len(text.split()) for text in texts]
    readability = [flesch_reading_ease(text) for text in texts]
    return np.mean(lengths), np.mean(readability)


# read data
with open('output3.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

orig_sents = data["originals"]
refs_sents = data["references"]
sys_sents = data["predictions"]


import os
import numpy as np
import json

def wholeTest(orig_sents, refs_sents, sys_sents, save_path="results/eval.json", max_errors=20):
    # Step 1: Filter by grammar
    grammar_errors, keep_mask = grammar_check(sys_sents, max_errors=max_errors)
    
    filtered_sys_sents = [s for s, keep in zip(sys_sents, keep_mask) if keep]
    filtered_orig_sents = [s for s, keep in zip(orig_sents, keep_mask) if keep]
    filtered_refs_sents = [s for s, keep in zip(refs_sents, keep_mask) if keep]

    #remove_chars = 'Einfachere Version:'
    #filtered_refs_sents = []
    #for s in filtered_refs_sent:
    #    for ch in remove_chars:
    #        s = s.replace(ch, '')
    #    filtered_refs_sents.append(s)
    #    print(s)
    print(f"\n==== Evaluation Report ====")
    print(f"Filtered {len(sys_sents) - len(filtered_sys_sents)} sentences due to grammar errors.")

    if len(filtered_sys_sents) == 0:
        print("No valid sentences after filtering. Evaluation skipped.")
        return

    results = {}
    results["grammar_errors_avg"] = grammar_errors

    # Ensure references are in nested format
    formatted_refs = [[r] for r in filtered_refs_sents]

    # BERTScore
    bertscore = metric_bertscore.compute(
        predictions=filtered_sys_sents,
        references=filtered_refs_sents,
        model_type="bert-base-multilingual-cased",
        lang="de",
        batch_size=8
    )
    results["bertscore_f1"] = float(np.mean(bertscore['f1']))
    print(f"BERTScore (F1): {results['bertscore_f1']:.3f}")

    # BLEU
    bleu_score = metric_bleu.compute(predictions=filtered_sys_sents, references=formatted_refs)
    results["bleu"] = bleu_score['bleu']
    print(f"BLEU Score: {results['bleu']:.3f}")

    # ROUGE
    rouge_score = metric_rouge.compute(predictions=filtered_sys_sents, references=filtered_refs_sents, rouge_types=["rougeL"])
    results["rougeL"] = rouge_score["rougeL"]
    print(f"ROUGE-L Score: {results['rougeL']:.3f}")

    # Readability
    avg_length, avg_readability = calculate_simplification(filtered_sys_sents)
    results["avg_sentence_length"] = avg_length
    results["flesch_reading_score"] = avg_readability
    print(f"Average sentence length: {avg_length:.2f}")
    print(f"Flesch Reading score: {avg_readability:.2f}")

    # Difficulty dictionary hit rate
    avg_hit = batch_hit_percent(filtered_sys_sents, tokenizer, model, ref_token_id, sep_token_id, cls_token_id, difficult_dict)
    results["difficulty_hit_percent"] = avg_hit
    print("Difficulty Hit Percent score (the lower the better):", avg_hit)

    # SARI
    sari_score = corpus_sari(orig_sents=filtered_orig_sents, sys_sents=filtered_sys_sents, refs_sents=[filtered_refs_sents])
    results["sari"] = sari_score
    print("SARI score (the higher the better):", sari_score)

    # FKGL
    results["fkgl_original"] = corpus_fkgl(filtered_orig_sents)
    results["fkgl_output"] = corpus_fkgl(filtered_sys_sents)
    print(f"Original FKGL: {results['fkgl_original']:.2f}")
    print(f"Output FKGL: {results['fkgl_output']:.2f}")

    # Quality estimation
    results["quality_estimation"] = corpus_quality_estimation(orig_sentences=filtered_orig_sents, sys_sentences=filtered_sys_sents)
    print("Quality estimation:", results["quality_estimation"])

    # Word reduction
    original_lengths = [len(sent.split()) for sent in filtered_orig_sents]
    simplified_lengths = [len(sent.split()) for sent in filtered_sys_sents]
    length_diffs = [orig - simp for orig, simp in zip(original_lengths, simplified_lengths)]
    avg_reduction = np.mean(length_diffs)
    avg_reduction_percent = np.mean([diff / orig if orig > 0 else 0 for diff, orig in zip(length_diffs, original_lengths)]) * 100

    results["avg_word_reduction"] = avg_reduction
    results["avg_word_reduction_percent"] = avg_reduction_percent
    print(f"Average word reduction per sentence: {avg_reduction:.2f} words")
    print(f"Average word reduction percentage: {avg_reduction_percent:.2f}%")

    # Save results
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Results saved to {save_path}")

# untrained result.
print('train free model result:\n')
wholeTest(orig_sents, refs_sents, sys_sents)

with open('raw_output.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# normal trained result
raw_orig_sents = data["originals"]
raw_refs_sents = data["references"]
raw_sys_sents = data["predictions"]

print('normal trained model result:\n')
wholeTest(raw_orig_sents, raw_refs_sents, raw_sys_sents)

with open('VIBoutput_D6(好点了.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

#with open('VIBoutput_D9.json', 'r', encoding='utf-8') as f:
#    data = json.load(f)

# VIB result
raw_orig_sents = data["originals"]
raw_refs_sents = data["references"]
raw_sys_sents = data["predictions"]

print('VIB trained model result:\n')
wholeTest(raw_orig_sents, raw_refs_sents, raw_sys_sents)


