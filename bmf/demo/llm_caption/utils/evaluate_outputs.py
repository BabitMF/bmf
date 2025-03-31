from evaluate import load
import os
import matplotlib.pyplot as plt
from collections import defaultdict
import json
import pickle
import sys

MODELS = ["Deepseek_VL2", "Deepseek_Janus_1b", "Deepseek_Janus_7b", "Qwen2_VL", "Qwen2_5_VL_3b", "Qwen2_5_VL_7b", "LLaVA_One_Vision", "LLaVA_Next_Video"]
DIRS = ["test", "test1", "test2", "test3", "test4"]

def get_truth():
    ground_truth = None
    try:
        with open("ground_truth.txt", "r") as file:
            ground_truth = file.readlines()
    except FileNotFoundError:
        print("ground_truth.txt not found, exiting")
    return ground_truth

def read_json(file_path):
    if os.path.getsize(file_path) == 0:
        return None
    with open(file_path, "r") as file:
        data = json.load(file)
    return data

def create_default_dict():
    return defaultdict(list)

def create_default_batch():
    return defaultdict(create_default_dict)

def process_results(ground_truth):
    bertscore = load("bertscore")
    bleu = load("bleu")
    rouge = load("rouge")
    meteor = load("meteor")
    # maps iteration count to a dictionary
    # this dictionary maps batch size, to another dictionary
    # this other dictionary maps model name to (bert score against summary, inference time per frame, total turnaround time)
    iteration_to_batch = defaultdict(create_default_batch)

    for iteration, dir in enumerate(DIRS):
        batch_to_model = create_default_batch()
        batch_count = 1
        while (prefix := os.path.join(dir, f"BATCH_{batch_count}")) and os.path.exists(prefix):
            model_to_results = defaultdict(list)
            for model in MODELS:
                # if there is an error file skip it
                if os.path.exists(os.path.join(prefix, model + ".log")):
                    model_to_results[model] = [None, None, None]
                    continue
                # get data
                data = read_json(os.path.join(prefix, model))
                # get all score's f score
                bert_score = bertscore.compute(predictions=[data["summary"]], references=ground_truth, lang="en")["f1"][0] if data["summary"] else None
                try:
                    bleu_score = bleu.compute(predictions=[data["summary"]], references=ground_truth)["bleu"] if data["summary"] else None
                except:
                    bleu_score = None
                rouge_score = rouge.compute(predictions=[data["summary"]], references=ground_truth)["rouge1"] if data["summary"] else None 
                meteor_score = meteor.compute(predictions=[data["summary"]], references=ground_truth)["meteor"] if data["summary"] else None 
                scores = [bert_score, bleu_score, rouge_score, meteor_score]

                model_to_results[model] = scores + [data["average_inference"], data["turnaround_time"]]
            batch_to_model[batch_count] = model_to_results
            batch_count += 1
        iteration_to_batch[iteration] = batch_to_model
    return iteration_to_batch

def create_figures(result):
    os.makedirs("figures", exist_ok=True)
    # bert score against batch size (filtered by model)
    # maps model name to (bert score, batch size)
    iteration_to_batch = defaultdict(lambda: defaultdict(list))
    for iteration, batch_results in result.items():
        model_to_score_and_batch_size = defaultdict(list)
        for batch_count, model_to_results in batch_results.items():
            for model, stats in model_to_results.items():
                model_to_score_and_batch_size[model].append(stats[0:4] + [batch_count])
        iteration_to_batch[iteration] = model_to_score_and_batch_size
    
    os.makedirs("figures/bert", exist_ok=True)
    for model in MODELS:
        plt.figure()
        for iteration in iteration_to_batch:
            filter = iteration_to_batch[iteration][model]
            bert_scores = []
            batch_sizes = []
            for scores in filter:
                if not scores[0]:
                    continue
                bert_scores.append(scores[0])
                batch_sizes.append(scores[4])
            plt.plot(batch_sizes, bert_scores, label=f"Iteration {iteration}")
        plt.xlabel("Batch Size")
        plt.ylabel("Score (F1)")
        plt.ylim([min(int(min(bert_scores)) - 1, 0), 1])
        plt.title(f"Scores vs Batch Size for {model}")
        plt.legend()
        plt.savefig(f"figures/bert/{model}.png")
        
    # bert score against batch size (highest bert score for each batch size)
        
    # inference time
    # total inference time

def main(args):
    if os.path.exists("result.pkl"):
        with open("result.pkl", "rb") as f:
            result = pickle.load(f)
    else:
        truth = get_truth()
        result = process_results(truth)
        with open("result.pkl", 'wb') as f:
            pickle.dump(result, f)
    create_figures(result)

if __name__ == "__main__":
    main(sys.argv)
