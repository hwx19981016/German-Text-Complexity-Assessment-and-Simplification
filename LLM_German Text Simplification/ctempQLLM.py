import torch
import json
import random
import evaluate
from datasets import load_dataset, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
from tqdm import tqdm
from sacremoses import MosesTokenizer
from comet import download_model, load_from_checkpoint

# keep this token safe please. It's for loading the model and tokenizer.
access_token = "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"

# load the data
dataset = load_dataset("json", data_files="german_simplification.jsonl")




def format_data(sample):
    return {
        "text": sample['input'],  
        "labels": sample["output"],  # label
    }

dataset = dataset.map(format_data)


# train_test_split
dataset = dataset["train"].train_test_split(test_size=0.1, seed=42)
train_dataset = dataset["train"]
test_dataset = dataset["test"]


# load tokenizer          meta-llama/Llama-3.2-3B-Instruct
#tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct")
#tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct",cache_dir='autodl-tmp', token=access_token)
#tokenizer = AutoTokenizer.from_pretrained("benjamin/gerpt2",cache_dir='autodl-tmp', token=access_token)
tokenizer = AutoTokenizer.from_pretrained("ai-forever/mGPT",cache_dir='autodl-tmp', token=access_token)


tokenizer.add_special_tokens({'pad_token': '[PAD]'})
tokenizer.pad_token = tokenizer.eos_token
def tokenize_function(samples):
    tokenized = tokenizer(samples["text"], padding="max_length", truncation=True, max_length=256)
    with tokenizer.as_target_tokenizer():  # 处理 target
        samples["labels"] = ['Einfachere Version:'+x for x in samples["labels"]]
        tokenized["labels"] = tokenizer(samples["labels"], padding="max_length", truncation=True, max_length=256)["input_ids"]
    return tokenized

train_dataset = train_dataset.map(tokenize_function, batched=True)
test_dataset = test_dataset.map(tokenize_function, batched=True)

# load LLM and make it qlora. Consturct the prompt.

#model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-3B-Instruct", torch_dtype=torch.float16, device_map="auto")
#model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B-Instruct", torch_dtype=torch.float16, device_map="auto",cache_dir='autodl-tmp', token=access_token)
#model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-3B-Instruct", torch_dtype=torch.float16, device_map="auto",cache_dir='autodl-tmp', token=access_token)
#model = AutoModelForCausalLM.from_pretrained("benjamin/gerpt2", torch_dtype=torch.float16, device_map="auto",cache_dir='autodl-tmp', token=access_token)
model = AutoModelForCausalLM.from_pretrained("ai-forever/mGPT", torch_dtype=torch.float16, device_map="auto",cache_dir='autodl-tmp', token=access_token)


for name, module in model.named_modules():
    if "attn" in name.lower():
        print(name)

raw_predictions = []
raw_originals = []
raw_references = []
for sample in tqdm(test_dataset, desc="Generating raw outputs"):
    original_sentence = sample["text"].strip()
    prompt = (
        "Bitte geben Sie nur die vereinfachte Version des Satzes aus.\n"
        "1. Die vereinfachte Version muss grammatikalisch korrekt sein.\n"
        "2. Der Inhalt des Satzes darf nicht verändert werden, sondern nur vereinfacht werden.\n"
        "3. Vermeiden Sie Wiederholungen oder unverständlichen Text.\n\n"
        "4. Keine weiteren Erklärunge"
        "zwei Beispiel:\n"
        "Original: Bei den Mönchsorden regelten genaue Vorschriften die Benutzung und Verwahrung der verwendeten Rasiermesser.\n"  # 添加样例
        "Einfachere Version: Bei den Mönchsorden gab es genaue Vorschriften. So wurde genau geregelt, wie Rasiermesser benutzt und verwahrt werden sollten.\n"
        "Original: Angesichts der Veränderung der Altersstruktur und des Anstiegs der Lebenserwartung in vielen Ländern rückt inzwischen auch die Betreuung und Pflege älterer oder pflegebedürftiger Angehöriger stärker in den Mittelpunkt des Interesses, auch der Politik.\n"  # 添加样例
        "Einfachere Version: In vielen Ländert verändert sich die Altersstruktur, die Lebenserwartung steigt. Deshalb sind Betreuung und Pflege älterer oder pflegebedürftiger Angehöriger in den Mittelpunkt des Interesses gerückt, auch der Politik..\n"
        f"Original: {original_sentence}\nEinfachere Version:"
    )
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    
    output = model.generate(
        **inputs,
        max_new_tokens=40,
        temperature=1.2,
        repetition_penalty=1.2,
        eos_token_id=tokenizer.eos_token_id,
    )
    
    raw_text = tokenizer.decode(output[0], skip_special_tokens=True).strip()
    label_text = tokenizer.decode(sample["labels"], skip_special_tokens=True).strip()
    if "Einfachere Version:" in raw_text:
        raw_text = raw_text.split("Einfachere Version:")[-1].strip()
    raw_predictions.append(raw_text)
    raw_originals.append(original_sentence)
    raw_references.append(label_text)  # BLEU 需要 [[参考答案]]

raw_data = {
    "originals": raw_originals,
    "references": raw_references,
    "predictions": raw_predictions
}

with open("raw_output.json", "w") as f:
    json.dump(raw_data, f, ensure_ascii=False, indent=4)

print("The untrained output was generated successfully！")


# Low Rank 
# qwen2.5
lora_config = LoraConfig(
    r=16,  # the scope of training parameter
    lora_alpha=24,  # The influence degree of training 
    #target_modules=["q_proj", "v_proj", "k_proj", "o_proj"], #moudle to train
    target_modules=["c_attn","c_proj"],
    lora_dropout=0.1,
    bias="none"
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# trainer 
training_args = TrainingArguments(
    output_dir="qwen_german_simplified",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=8,
    num_train_epochs=12,
    #eval_strategy="epoch",  # old version transformers to apply evaluation_strategy
    #save_strategy="epoch",
    logging_steps=10,
    fp16=True,
    save_total_limit=2,
    push_to_hub=False,
    greater_is_better=False,  # 关键！指定越小越好
)

import json
with open("difficult_words.json", "r", encoding="utf-8") as f:
    difficult_dict = json.load(f)

# float16 float32
# float pytorch16
# 

# Important! This is the regular train loss. 
'''
class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        outputs = model(**{k: v for k, v in inputs.items() if k != "labels"})
        logits = outputs.logits
        labels = inputs.get("labels")

        if labels is not None:
            loss_fct = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
            loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
        else:
            loss = torch.tensor(0.0, device=logits.device)

        return (loss, outputs) if return_outputs else loss
'''
import torch.nn.functional as F
'''
class CustomTrainer(Trainer):
    def __init__(self, *args, difficult_dict=None, tokenizer=None, gamma=0.5, **kwargs):
        super().__init__(*args, **kwargs)
        self.tokenizer_ref = tokenizer  
        self.gamma = gamma

        # === 构建困难词的 token 序列 ===
        self.difficult_token_seqs = []
        if difficult_dict is not None:
            for word, is_difficult in difficult_dict.items():
                if is_difficult:
                    token_ids = self.tokenizer_ref.encode(word, add_special_tokens=False)
                    if len(token_ids) > 0:
                        self.difficult_token_seqs.append(token_ids)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        outputs = model(**{k: v for k, v in inputs.items() if k != "labels"})
        logits = outputs.logits  # [B, T, V]
        labels = inputs.get("labels")

        if labels is None:
            return torch.tensor(0.0, device=logits.device)

        loss_fct = torch.nn.CrossEntropyLoss(ignore_index=self.tokenizer_ref.pad_token_id)
        ce_loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))

        log_probs = F.log_softmax(logits, dim=-1)      # [B, T, V]
        pred_ids = torch.argmax(log_probs, dim=-1)     # [B, T]

        difficulty_loss = torch.tensor(0.0, device=logits.device)
        matched_spans = 0

        for token_seq in self.difficult_token_seqs:
            token_seq_tensor = torch.tensor(token_seq, device=logits.device)
            token_len = len(token_seq)
            if token_len > pred_ids.size(1):
                continue
            window = pred_ids.unfold(dimension=1, size=token_len, step=1)  # [B, T-L+1, L]
            matches = (window == token_seq_tensor).all(dim=-1)             # [B, T-L+1]

            for b in range(matches.size(0)):
                for t in torch.where(matches[b])[0]:
                    matched_ids = token_seq_tensor
                    matched_log_probs = log_probs[b, t:t+token_len, matched_ids]  # [L, L]
                    difficulty_loss += -matched_log_probs.diagonal().sum()
                    matched_spans += 1

        if matched_spans > 0:
            difficulty_loss = difficulty_loss / matched_spans

        # === 总损失 ===
        total_loss = ce_loss + self.gamma * difficulty_loss
        return (total_loss, outputs) if return_outputs else total_loss
'''
'''
# Important! This is the VIB train loss. 
class CustomTrainer(Trainer):
    def __init__(self, *args, difficult_dict=None, tokenizer=None, gamma=2, beta=0.05, **kwargs):
        super().__init__(*args, **kwargs)
        self.difficult_dict = difficult_dict  # 词典，{token_id: 0/1} 标记难词
        self.tokenizer = tokenizer
        self.gamma = gamma  # 难词惩罚权重
        self.beta = beta    # VIB KL权重
    def get_difficulty_token_mask(self, token_ids):
        # 先把所有难词拆成token id序列，汇总所有子token id
        difficult_words = [word for word, flag in self.difficult_dict.items() if flag == 1]
        difficult_token_ids = []
        for word in difficult_words:
            ids = tokenizer.encode(word, add_special_tokens=False)
            difficult_token_ids.extend(ids)
        difficult_token_ids = list(set(difficult_token_ids))
        difficult_token_ids_tensor = torch.tensor(difficult_token_ids, device=token_ids.device)
    
        # token_ids 是 [B, T] 的tensor，判断每个token是否是难词子token之一
        mask = torch.isin(token_ids, difficult_token_ids_tensor).float()  # [B, T]
    
        return mask

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        outputs = model(**{k: v for k, v in inputs.items() if k != "labels"})
        logits = outputs.logits
        labels = inputs.get("labels")

        if labels is not None:
            loss_fct = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
            ce_loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))  # 交叉熵损失

            # --- 变分信息瓶颈（VIB）部分 ---
            mu = logits.mean(dim=-1)  # 计算均值
            std = torch.nn.functional.softplus(logits.std(dim=-1))  # 计算标准差，softplus 确保正值
            kl_loss = -0.5 * torch.sum(1 + torch.log(std ** 2) - mu ** 2 - std ** 2)  # KL 散度
            kl_loss = kl_loss / logits.size(0)  # 归一化
            
            # 下面是改的地方：先解码，然后获取difficulty_word的位置，
            with torch.no_grad():
                pred_ids = logits.argmax(dim=-1)  # [B, T]
            difficulty_mask = self.get_difficulty_token_mask(pred_ids)  # [B, T]
            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
            pred_log_prob = log_probs.gather(-1, pred_ids.unsqueeze(-1)).squeeze(-1)  # [B, T]

            # 这里是惩罚难词预测的 log-prob，难词越大，损失越大
            difficulty_loss = (pred_log_prob * difficulty_mask).sum() / (difficulty_mask.sum() + 1e-6)

            # 总损失
            #beta = 0.1  # 超参数：调整 VIB 影响力
            loss = ce_loss + self.beta * kl_loss + self.gamma * difficulty_loss # 组合交叉熵 + VIB 正则 + 困难词惩罚

            # loss = ce_loss + beta * kl_loss   # 组合交叉熵 + VIB 正则
        else:
            loss = torch.tensor(0.0, device=logits.device)

        return (loss, outputs) if return_outputs else loss
'''

from tqdm import tqdm
import torch.nn.functional as F
class CustomTrainer(Trainer):
    def __init__(self, *args, difficult_dict=None, tokenizer=None, gamma=1.0, beta=0.5, **kwargs):
        super().__init__(*args, **kwargs)
        self.tokenizer_ref = tokenizer  # 替代 self.tokenizer，消除了 deprecated warning
        self.gamma = gamma
        self.beta = beta

        # === 构建困难词 token 序列列表 ===
        self.difficult_token_seqs = []
        if difficult_dict is not None:
            for word, is_difficult in difficult_dict.items():
                if is_difficult:
                    token_ids = self.tokenizer_ref.encode(word, add_special_tokens=False)
                    if len(token_ids) > 0:
                        self.difficult_token_seqs.append(token_ids)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        outputs = model(**{k: v for k, v in inputs.items() if k != "labels"})
        logits = outputs.logits  # [B, T, V]
        labels = inputs.get("labels")

        if labels is None:
            return torch.tensor(0.0, device=logits.device)

        # === CrossEntropy Loss ===
        loss_fct = torch.nn.CrossEntropyLoss(ignore_index=self.tokenizer_ref.pad_token_id)
        ce_loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))

        # === VIB KL Loss ===
        mu = logits.mean(dim=-1)  # [B, T]
        std = F.softplus(logits.std(dim=-1))  # [B, T]
        kl_loss = -0.5 * torch.sum(1 + torch.log(std ** 2 + 1e-8) - mu ** 2 - std ** 2)
        kl_loss = kl_loss / logits.size(0)  # 按 batch 归一化

        
        log_probs = F.log_softmax(logits, dim=-1)  # [B, T, V]
        pred_ids = torch.argmax(log_probs, dim=-1)  # [B, T]

        difficulty_loss = torch.tensor(0.0, device=logits.device)
        matched_spans = 0

        for token_seq in self.difficult_token_seqs:
            token_seq_tensor = torch.tensor(token_seq, device=logits.device)
            token_len = len(token_seq)
            if token_len > pred_ids.size(1):
                continue

            window = pred_ids.unfold(dimension=1, size=token_len, step=1)  # [B, T-L+1, L]
            matches = (window == token_seq_tensor).all(dim=-1)  # [B, T-L+1]

            for b in range(matches.size(0)):
                for t in torch.where(matches[b])[0]:
                    matched_ids = token_seq_tensor  # [L]
                    matched_log_probs = log_probs[b, t:t+token_len, matched_ids]  # [L, L]
                    difficulty_loss += -matched_log_probs.diagonal().sum()
                    matched_spans += 1

        if matched_spans > 0:
            difficulty_loss = difficulty_loss / matched_spans

        total_loss = 0.6 * ce_loss + self.beta * kl_loss + self.gamma * difficulty_loss
        return (total_loss, outputs) if return_outputs else total_loss

import numpy as np

def difficulty_hit_percent(sentences, tokenizer, model, ref_token_id, sep_token_id, cls_token_id, difficult_dict):
    from captum_process import captumProcess
    hit_percent_list = []
    for text in sentences:
        try:
            top_words, _ = captumProcess(text, tokenizer, model, ref_token_id, sep_token_id, cls_token_id)
            if not top_words:
                continue
            hit = sum(1 for word in top_words if difficult_dict.get(word, 0) == 1)
            hit_percent = hit / len(top_words)
            hit_percent_list.append(hit_percent)
        except Exception:
            continue
    if hit_percent_list:
        avg_hit = np.mean(hit_percent_list)
        return avg_hit
    else:
        return 0.0
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    # 根据 logits 转成句子，这里假设有 decode 函数
    preds = tokenizer.batch_decode(logits.argmax(axis=-1), skip_special_tokens=True)
    difficulty_hit = difficulty_hit_percent(
        preds, tokenizer, model,
        ref_token_id=tokenizer.pad_token_id,
        sep_token_id=tokenizer.sep_token_id,
        cls_token_id=tokenizer.cls_token_id,
        difficult_dict=difficult_dict
    )
    return {"difficulty_hit_percent": difficulty_hit}
trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    difficult_dict=difficult_dict
)

# start train
trainer.train()

# save model
trainer.save_model("qwen_german_simplified_lora")

def simplify_sentence(sentence):
    prompt = (
        "Bitte geben Sie nur die vereinfachte Version des Satzes aus.\n"
        "1. Die vereinfachte Version muss grammatikalisch korrekt sein.\n"
        "2. Der Inhalt des Satzes darf nicht verändert werden, sondern nur vereinfacht werden.\n"
        "3. Vermeiden Sie Wiederholungen oder unverständlichen Text.\n\n"
        "4. Keine weiteren Erklärunge"
        "zwei Beispiel:\n"
        "Original: Bei den Mönchsorden regelten genaue Vorschriften die Benutzung und Verwahrung der verwendeten Rasiermesser.\n"  # 添加样例
        "Einfachere Version: Bei den Mönchsorden gab es genaue Vorschriften. So wurde genau geregelt, wie Rasiermesser benutzt und verwahrt werden sollten.\n"
        "Original: Angesichts der Veränderung der Altersstruktur und des Anstiegs der Lebenserwartung in vielen Ländern rückt inzwischen auch die Betreuung und Pflege älterer oder pflegebedürftiger Angehöriger stärker in den Mittelpunkt des Interesses, auch der Politik.\n"  # 添加样例
        "Einfachere Version: In vielen Ländert verändert sich die Altersstruktur, die Lebenserwartung steigt. Deshalb sind Betreuung und Pflege älterer oder pflegebedürftiger Angehöriger in den Mittelpunkt des Interesses gerückt, auch der Politik..\n"
        f"Original: {sentence}\nEinfachere Version:"
    )

    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    output = model.generate(
        **inputs,
        max_new_tokens=40,  
        temperature=0.2, # under this temperature, model performent better.
        repetition_penalty=1.2,  
        eos_token_id=tokenizer.eos_token_id,
    )

    simplified_text = tokenizer.decode(output[0], skip_special_tokens=True).strip()

    # Only extract the content after "Einfachere Version:" to avoid useless words.
    if "Einfachere Version:" in simplified_text:
        simplified_text = simplified_text.split("Einfachere Version:")[-1].strip()

    return simplified_text


# sample test
example_sentence = "Die Erde umkreist die Sonne in einer elliptischen Bahn."

print("Simplified:", simplify_sentence(example_sentence))



# total test
metric_bleu = evaluate.load("bleu")
metric_rouge = evaluate.load("rouge")

references = []
predictions = []
originals = []
for sample in tqdm(test_dataset):
    original_sentence = sample["text"].strip()
    
    prediction = simplify_sentence(original_sentence)

    label_text = tokenizer.decode(sample["labels"], skip_special_tokens=True).strip()
    originals.append(original_sentence)
    references.append(label_text)  # BLEU 需要 [[参考答案]]
    predictions.append(prediction)



import numpy as np
print(f'Label is {references}')
print(f'prediction is {predictions}')


import json

# save result
data = {
    "originals": originals,
    "references": references,
    "predictions": predictions
}

#with open("VIBoutput.json", "w") as f:
#    json.dump(data, f, ensure_ascii=False, indent=4)
with open("output1.json", "w") as f:
    json.dump(data, f, ensure_ascii=False, indent=4)
print("saved")



# more test about semantic similarity
# Semantic Similarity: BERTScore: Measures meaning preservation using contextual embeddings
# Surface-Level Similarity: BLEU、ROUGE
# Grammaticality: LanguageTool Error Count: Number of detected grammatical errors (German-specific)
# Readability: Flesch Reading Ease: Scores text difficulty (higher = easier); Average Sentence Length: In words (shorter = typically more readable);
# Fluency: Perplexity: Measures how well the language model predicts the text (lower = more fluent)

metric_bertscore = evaluate.load("bertscore")
bertscore = metric_bertscore.compute(
    predictions=predictions,
    references=references,
    model_type="bert-base-multilingual-cased",  
    lang="de",
    batch_size=8
)
raw_ertscore = metric_bertscore.compute(
    predictions=raw_predictions,
    references=raw_references,
    model_type="bert-base-multilingual-cased",  
    lang="de",
    batch_size=8
)
print(f"BERTScore (F1): {np.mean(bertscore['f1']):.3f}")
print(f"RAW BERTScore (F1): {np.mean(raw_ertscore['f1']):.3f}")
#bleu_score = metric_bleu.compute(predictions=predictions, references=references)
#rouge_score = metric_rouge.compute(predictions=predictions, references=references)

tokenizerG = MosesTokenizer(lang="de")
bleu_score = metric_bleu.compute(
    predictions=[tokenizerG.tokenize(pred, return_str=True) for pred in predictions],
    references=[[tokenizerG.tokenize(ref, return_str=True)] for ref in references],
    smooth=True
)
rouge_score = metric_rouge.compute(
    predictions=[tokenizerG.tokenize(pred, return_str=True) for pred in predictions],
    references=[tokenizerG.tokenize(ref, return_str=True) for ref in references],
    rouge_types=["rougeL", "rougeLsum"]
)

import language_tool_python

tool = language_tool_python.LanguageTool("de")  # 德语检查

def grammar_check(texts):
    scores = []
    for text in texts:
        matches = tool.check(text)  # 语法错误数量
        scores.append(len(matches))
    print(f'The number of grammatical errors: {scores}')
    return np.mean(scores)

grammar_errors = grammar_check(predictions)
raw_grammar_errors = grammar_check(raw_predictions)
print(f"The average number of grammatical errors: {grammar_errors:.2f}（the lower, the better）")
print(f"The average number of untrained grammatical errors: {raw_grammar_errors:.2f}（the lower, the better）")


from textstat import flesch_reading_ease
import nltk

def calculate_simplification(texts):
    lengths = [len(text.split()) for text in texts]  # 词数
    readability = [flesch_reading_ease(text) for text in texts]  # 可读性评分
    print(f'readability: {readability}')
    return np.mean(lengths), np.mean(readability)

avg_length, avg_readability = calculate_simplification(predictions)
raw_avg_length, raw_avg_readability = calculate_simplification(raw_predictions)
print(f"Average sentence length: {avg_length:.2f} ")
print(f"Flesch Reading readability score: {avg_readability:.2f}（The higher, the easier to read）")
print(f"Untrained average sentence length: {raw_avg_length:.2f} ")
print(f"Untrained Flesch reading readability score: {raw_avg_readability:.2f}（The higher, the easier to read）")







from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch


def calculate_ppl(texts, model, tokenizer):
    model=model.cpu()
    encodings = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**encodings, labels=encodings["input_ids"])
    loss = outputs.loss
    perplexity = torch.exp(loss)
    return perplexity.item()

fluency_scores = [calculate_ppl(text, model, tokenizer) for text in predictions]
raw_fluency_scores = [calculate_ppl(text, model, tokenizer) for text in raw_predictions]
print(f"Perplexity: {np.mean(fluency_scores):.3f}")
print(f"Perplexity: {np.mean(raw_fluency_scores):.3f}")




print(f'bertscore:{bertscore}')
#print(f"BLEU Score: {bleu_score}")
#print(f"ROUGE Score: {rouge_score}")

