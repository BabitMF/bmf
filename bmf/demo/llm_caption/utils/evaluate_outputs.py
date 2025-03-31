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
                    model_to_results[model] = [None, None, None, None]
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

def _create_figure1(iteration_to_batch):
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
        # If the top value is greater than 1, set it to 1
        ax = plt.gca()
        current_ylim = ax.get_ylim()
        if current_ylim[1] > 1:
            ax.set_ylim(bottom=current_ylim[0], top=1)
        plt.title(f"Scores vs Batch Size for {model}")
        plt.legend()
        plt.savefig(f"figures/bert/{model}.png")

def _create_figure2(iteration_to_batch):
    os.makedirs("figures/all_scores", exist_ok=True)
    for model in MODELS:
        # maps batch size to score types that maps to scores
        batch_size_to_score_type = defaultdict(lambda: defaultdict(list))
        for iteration in iteration_to_batch:
            filter = iteration_to_batch[iteration][model]
            for score in filter:
                if any(s == None for s in score):
                    continue
                size = score[4]
                batch_size_to_score_type[size]['bert_score'].append(score[0])
                batch_size_to_score_type[size]['bleu_score'].append(score[1])
                batch_size_to_score_type[size]['rouge_score'].append(score[2])
                batch_size_to_score_type[size]['meteor_score'].append(score[3])

        score_type_to_average = defaultdict(list)
        for batch_size, scores in batch_size_to_score_type.items():
            for score_type, raw in scores.items():
                average_score = sum(raw) / len(raw)
                score_type_to_average[score_type].append((batch_size, average_score))

        plt.figure()

        for score_type, average in score_type_to_average.items():
            sorted_scores = sorted(average, key=lambda x: x[0])
            batch_sizes = [x[0] for x in sorted_scores]
            averages = [x[1] for x in sorted_scores]
            plt.plot(batch_sizes, averages, label=score_type)

        plt.xlabel("Batch Size")
        plt.ylabel("Average Score (F1)")
        plt.title(f"Average Scores vs Batch Size for {model}")
        plt.legend()
        plt.xticks(range(int(min(batch_sizes)), int(max(batch_sizes)) + 1))
        plt.savefig(f"figures/all_scores/{model}_average.png")


def create_figures(result):
    # prepare results
    os.makedirs("figures", exist_ok=True)
    iteration_to_batch = defaultdict(lambda: defaultdict(list))
    for iteration, batch_results in result.items():
        model_to_score_and_batch_size = defaultdict(list)
        for batch_count, model_to_results in batch_results.items():
            for model, stats in model_to_results.items():
                model_to_score_and_batch_size[model].append(stats[:4] + [batch_count])
        iteration_to_batch[iteration] = model_to_score_and_batch_size
    
    # bert score against batch size (filtered by model)
    # maps model name to (bert score, batch size)
    _create_figure1(iteration_to_batch)
        
    # all scores against batch size (filtered by model) and averaged across all iterations
    _create_figure2(iteration_to_batch)

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
