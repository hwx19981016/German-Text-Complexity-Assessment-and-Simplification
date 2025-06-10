import json
from easse.sari import corpus_sari
#from easse.samsa import sentence_samsa
from easse.fkgl import corpus_fkgl
from easse.bleu import corpus_bleu
from easse.quality_estimation import corpus_quality_estimation
from easse.compression import corpus_f1_token
from eassede.easse.textstat_metrics import corpus_fre
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

tool = language_tool_python.LanguageTool("de")
tokenizerG = MosesTokenizer(lang="de")
def calculate_ppl(text, model, tokenizer):
    model = model.cpu()
    encodings = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**encodings, labels=encodings["input_ids"])
    loss = outputs.loss
    return torch.exp(loss).item()


def grammar_check(texts):
    scores = []
    for text in texts:
        matches = tool.check(text)
        scores.append(len(matches))
    return np.mean(scores)


def calculate_simplification(texts):
    lengths = [len(text.split()) for text in texts]
    readability = [flesch_reading_ease(text) for text in texts]
    return np.mean(lengths), np.mean(readability)




# read data
with open('output1.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

orig_sents = data["originals"]
refs_sents = data["references"]
sys_sents = data["predictions"]

def wholeTest(orig_sents, refs_sents, sys_sents):
    #print(refs_sents)
    #print('11412421421421')
    #refs_sents = [[sent] for sent in refs_sents]
    #print(refs_sents)
    #print(f'refs是{refs_sents}')
    #print(f'sys是{sys_sents}')
    #print(f'orig是{orig_sents}')
    # SARI score

    # BERTScore
    bertscore = metric_bertscore.compute(
        predictions=sys_sents,
        references=refs_sents,
        model_type="bert-base-multilingual-cased",
        lang="de",
        batch_size=8
    )
    print(f"BERTScore (F1): {np.mean(bertscore['f1']):.3f}")

    # BLEU
    bleu_score = metric_bleu.compute(predictions=sys_sents, references=[[r] for r in refs_sents])
    print(f"BLEU Score: {bleu_score['bleu']:.3f}")

    # ROUGE
    rouge_score = metric_rouge.compute(predictions=sys_sents, references=refs_sents, rouge_types=["rougeL"])
    print(f"ROUGE-L Score: {rouge_score['rougeL']:.3f}")

    # Grammar error
    grammar_errors = grammar_check(sys_sents)
    print(f"Avg Grammar Errors: {grammar_errors:.2f}")

    # Readability and sentence length
    avg_length, avg_readability = calculate_simplification(sys_sents)
    print(f"Average sentence length: {avg_length:.2f}")
    print(f"Flesch Reading score: {avg_readability:.2f}")

        
    sari_score = corpus_sari(orig_sents=orig_sents, sys_sents=sys_sents, refs_sents=[refs_sents])
    
    print("SARI score(The simplified score, the larger the better）:", sari_score)
    
    #samsa_score = corpus_samsa(orig_sents=orig_sents, sys_sents=sys_sents)
    #print("SAMSA score:", samsa_score)
    
    fkgl_score = corpus_fkgl(orig_sents)
    
    print("Original FKGL score（The smaller, the simpler. The smallest is 5）:", fkgl_score)
    
    fkgl_score_refs_sents = corpus_fkgl(sys_sents)
    
    print("Output FKGL score（The smaller, the simpler. The smallest is 5）:", fkgl_score_refs_sents)
    
    
    #bleu_score = corpus_bleu(sys_sents=sys_sents,  refs_sents=[refs_sents])
    #print("BLEU score:", bleu_score)
    
    
    score_quality = corpus_quality_estimation(orig_sentences=orig_sents, sys_sentences=sys_sents)
    
    print("Corpus_quality_estimation（The comparison before and after simplification）:", score_quality)
    
    
    #f1_score = corpus_f1_token(sys_sents=sys_sents, refs_sents=refs_sents)
    
    #print("F1 score:", f1_score)
    
    
    fre_score = corpus_fre(orig_sents)
    
    print("Original FRE score（The larger, the easier to read）:", fre_score)


    fre_score_output = corpus_fre(sys_sents)
    
    print("OUTPUT FRE score（The larger, the easier to read）:", fre_score_output)

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


with open('VIBoutput.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# VIB result
raw_orig_sents = data["originals"]
raw_refs_sents = data["references"]
raw_sys_sents = data["predictions"]

print('VIB trained model result:\n')
wholeTest(raw_orig_sents, raw_refs_sents, raw_sys_sents)

    
