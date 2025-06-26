from evaluate import load
import os
import matplotlib.pyplot as plt
from collections import defaultdict
import json
import pickle
import sys

MASTER = {
    "hugging_face": {
        "MODELS": ["Deepseek_VL2", "Deepseek_Janus_1b", "Deepseek_Janus_7b", "Qwen2_VL", "Qwen2_5_VL_3b", "Qwen2_5_VL_7b", "LLaVA_One_Vision", "LLaVA_Next_Video"],
        "DIRS": ["test", "test1", "test2", "test3", "test4"],
    },
    "vllm": {
        "MODELS": ["Deepseek_VL2", "Qwen2_VL", "Qwen2_5_VL_3b", "Qwen2_5_VL_7b", "LLaVA_One_Vision"],
        "DIRS": ["vllm"] 
    }
}

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
    result = []
    for backend, dic in MASTER.items():
        # maps iteration count to a dictionary
        # this dictionary maps batch size, to another dictionary
        # this other dictionary maps model name to (bert score against summary, inference time per frame, total turnaround time)
        iteration_to_batch = defaultdict(create_default_batch)
        for iteration, dir in enumerate(dic["DIRS"]):
            batch_to_model = create_default_batch()
            batch_count = 1
            while (prefix := os.path.join(dir, f"BATCH_{batch_count}")) and os.path.exists(prefix):
                # if backend == "vllm" and batch_count == 19:
                #     break
                model_to_results = defaultdict(list)
                for model in dic["MODELS"]:
                    # if there is an error file skip it
                    if os.path.exists(os.path.join(prefix, model + ".log")):
                        model_to_results[model] = [None] * 6
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
        result.append(iteration_to_batch)
    return result

def _prep_figure1_2(result):
    os.makedirs("figures", exist_ok=True)
    iteration_to_batch = defaultdict(lambda: defaultdict(list))
    for iteration, batch_results in result.items():
        model_to_score_and_batch_size = defaultdict(list)
        for batch_count, model_to_results in batch_results.items():
            for model, stats in model_to_results.items():
                model_to_score_and_batch_size[model].append(stats[:4] + [batch_count])
        iteration_to_batch[iteration] = model_to_score_and_batch_size
    return iteration_to_batch

def _prep_figure3_4_5_6(result):
    model_to_batch_size_to_avgs = defaultdict(lambda: defaultdict(list))
    model_to_batch_size_to_turnaround = defaultdict(lambda: defaultdict(list))
    model_to_batch_size_to_rouge = defaultdict(lambda: defaultdict(list))
    model_to_batch_size_to_meteor = defaultdict(lambda: defaultdict(list))
    for iteration, batch_to_model in result.items():
        for batch_size, model_to_results in batch_to_model.items():
            for model, results in model_to_results.items():
                rouge_score = results[2]
                meteor_score = results[3]
                avg_inf = results[-2]
                turnaround = results[-1]
                if any(x is None for x in [rouge_score, meteor_score, avg_inf, turnaround]):
                    continue
                model_to_batch_size_to_avgs[model][batch_size].append(avg_inf)
                model_to_batch_size_to_turnaround[model][batch_size].append(turnaround)
                model_to_batch_size_to_rouge[model][batch_size].append(rouge_score)
                model_to_batch_size_to_meteor[model][batch_size].append(meteor_score)
    return model_to_batch_size_to_avgs, \
            model_to_batch_size_to_turnaround, \
            model_to_batch_size_to_rouge, \
            model_to_batch_size_to_meteor

def _create_figure1(iteration_to_batch, prefix):
    os.makedirs(f"figures/bert_{prefix}", exist_ok=True)
    for model in MASTER[prefix]["MODELS"]:
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
        plt.title(f"Scores vs Batch Size for {model} ({prefix})")
        plt.legend()
        plt.savefig(f"figures/bert_{prefix}/{model}.png")
        plt.close()

def _create_figure2(iteration_to_batch, prefix):
    os.makedirs(f"figures/all_scores_{prefix}", exist_ok=True)
    for model in MASTER[prefix]["MODELS"]:
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
        plt.title(f"Average Scores vs Batch Size for {model} ({prefix})")
        plt.legend()
        plt.xticks(range(int(min(batch_sizes)), int(max(batch_sizes)) + 1))
        plt.savefig(f"figures/all_scores_{prefix}/{model}.png")
        plt.close()

def _create_figure3(model_to_batch_size_to_avgs, prefix):
    os.makedirs(f"figures/average_inference_{prefix}", exist_ok=True)
    for model, batch_size_to_avgs in model_to_batch_size_to_avgs.items():
        plt.figure()
        input = []
        for batch_size, averages in batch_size_to_avgs.items():
            average_of_averages = sum(averages) / len(averages)
            input.append((batch_size, average_of_averages))
        input.sort()
        plt.plot([x[0] for x in input], [x[1] for x in input])
        plt.xlabel("Batch Size")
        plt.xticks(range(input[0][0], input[-1][0] + 1))
        plt.ylabel("Average Inference Time per Frame")
        plt.title(f"Average Inference Time against Batch Size for {model} ({prefix})")
        plt.savefig(f"figures/average_inference_{prefix}/{model}.png")
        plt.close()
        
def _create_figure3_combined(model_to_batch_size_to_avgs, prefix):
    os.makedirs(f"figures/average_inference_combined_{prefix}", exist_ok=True)
    plt.figure()
    for model, batch_size_to_avgs in model_to_batch_size_to_avgs.items():
        input = []
        for batch_size, averages in batch_size_to_avgs.items():
            average_of_averages = sum(averages) / len(averages)
            input.append((batch_size, average_of_averages))
        input.sort()
        plt.plot([x[0] for x in input], [x[1] for x in input], label=model)
    plt.xlabel("Batch Size")
    plt.ylabel("Average Inference Time per Frame")
    ax = plt.gca()
    current_xlim = ax.get_xlim()
    if current_xlim[1] > 60:
        ax.set_xlim(right=60)
    plt.title(f"Average Inference Time against Batch Size (combined {prefix})")
    plt.legend()
    plt.savefig(f"figures/average_inference_combined_{prefix}/combined.png")
    plt.close()
        
def _create_figure4(model_to_batch_size_to_turnaround, prefix):
    os.makedirs(f"figures/turnaround_time_{prefix}", exist_ok=True)
    for model, batch_size_to_avgs in model_to_batch_size_to_turnaround.items():
        plt.figure()
        input = []
        for batch_size, averages in batch_size_to_avgs.items():
            average_turnaround = sum(averages) / len(averages)
            input.append((batch_size, average_turnaround))
        input.sort()
        plt.plot([x[0] for x in input], [x[1] for x in input])
        plt.xlabel("Batch Size")
        plt.xticks(range(input[0][0], input[-1][0] + 1))
        plt.ylabel("Average Turnaround Time")
        plt.title(f"Average Turnaround Time against Batch Size for {model} ({prefix})")
        plt.savefig(f"figures/turnaround_time_{prefix}/{model}.png")
        plt.close()
        
def _create_figure4_combined(model_to_batch_size_to_turnaround, prefix):
    os.makedirs(f"figures/turnaround_time_combined_{prefix}", exist_ok=True)
    plt.figure()
    for model, batch_size_to_turnaround in model_to_batch_size_to_turnaround.items():
        input = []
        for batch_size, averages in batch_size_to_turnaround.items():
            average_turnaround = sum(averages) / len(averages)
            input.append((batch_size, average_turnaround))
        input.sort()
        plt.plot([x[0] for x in input], [x[1] for x in input], label=model)
    plt.xlabel("Batch Size")
    plt.ylabel("Average Turnaround Time")
    ax = plt.gca()
    current_xlim = ax.get_xlim()
    if current_xlim[1] > 60:
        ax.set_xlim(right=60)
    plt.title(f"Average Turnaround Time against Batch Size ({prefix})")
    plt.legend()
    plt.savefig(f"figures/turnaround_time_combined_{prefix}/combined.png")
    plt.close()

def _create_figure5_combined(model_to_batch_size_to_rouge, prefix):
    os.makedirs(f"figures/rouge_combined_{prefix}", exist_ok=True)
    plt.figure()
    for model, batch_size_to_rouge in model_to_batch_size_to_rouge.items():
        input = []
        for batch_size, scores in batch_size_to_rouge.items():
            average_rouge = sum(scores) / len(scores)
            input.append((batch_size, average_rouge))
        input.sort()
        plt.plot([x[0] for x in input], [x[1] for x in input], label=model)
    plt.xlabel("Batch Size")
    plt.ylabel("Average ROUGE score")
    ax = plt.gca()
    current_xlim = ax.get_xlim()
    if current_xlim[1] > 60:
        ax.set_xlim(right=60)
    plt.title(f"Average ROUGE score against Batch Size ({prefix})")
    plt.legend()
    plt.savefig(f"figures/rouge_combined_{prefix}/combined.png")
    plt.close()

def _create_figure6_combined(model_to_batch_size_to_meteor, prefix):
    os.makedirs(f"figures/meteor_combined_{prefix}", exist_ok=True)
    plt.figure()
    for model, batch_size_to_meteor in model_to_batch_size_to_meteor.items():
        input = []
        for batch_size, scores in batch_size_to_meteor.items():
            average_rouge = sum(scores) / len(scores)
            input.append((batch_size, average_rouge))
        input.sort()
        plt.plot([x[0] for x in input], [x[1] for x in input], label=model)
    plt.xlabel("Batch Size")
    plt.ylabel("Average METEOR score")
    ax = plt.gca()
    current_xlim = ax.get_xlim()
    if current_xlim[1] > 60:
        ax.set_xlim(right=60)
    plt.title(f"Average METEOR score against Batch Size ({prefix})")
    plt.legend()
    plt.savefig(f"figures/meteor_combined_{prefix}/combined.png")
    plt.close()

def _create_figure7(model_to_batch_size_to_avgs_hf, model_to_batch_size_to_avgs_vllm):
    os.makedirs(f"figures/hf_vs_vllm", exist_ok=True)
    plt.figure()
    # only plot models that are common to both
    common_models = set(model_to_batch_size_to_avgs_hf) & set(model_to_batch_size_to_avgs_vllm)
    for model_to_batch_size_to_avgs, line_style in [(model_to_batch_size_to_avgs_hf, '-'), (model_to_batch_size_to_avgs_vllm, '--')]:
        for model in common_models:
            batch_size_to_avgs = model_to_batch_size_to_avgs[model]
            input = []
            for batch_size, averages in batch_size_to_avgs.items():
                average_of_averages = sum(averages) / len(averages)
                input.append((batch_size, average_of_averages))
            input.sort()

            import itertools
            # Get consistent colours from matplotlib's default color cycle
            color_cycle = itertools.cycle(plt.rcParams["axes.prop_cycle"].by_key()["color"])
            # Assign one unique colour per model
            model_to_color = {model: next(color_cycle) for model in sorted(common_models)}
            plt.plot(
                [x[0] for x in input],
                [x[1] for x in input],
                label=f"{model} ({'HF' if model_to_batch_size_to_avgs is model_to_batch_size_to_avgs_hf else 'vLLM'})",
                linestyle=line_style,
                color=model_to_color[model]  # consistent colour per model
            )

    plt.xlabel("Batch Size")
    plt.ylabel("Average Inference Time per Frame")
    ax = plt.gca()
    current_xlim = ax.get_xlim()
    if current_xlim[1] > 60:
        ax.set_xlim(right=60)
    plt.title(f"Average Inference Time against Batch Size (HF vs vLLM)")
    plt.legend()
    plt.savefig(f"figures/hf_vs_vllm/combined.png")
    plt.close()

def create_figures(result, prefix):
    # prepare results for figure 1 and 2
    iteration_to_batch = _prep_figure1_2(result)
    # bert score against batch size (filtered by model)
    # maps model name to (bert score, batch size)
    _create_figure1(iteration_to_batch, prefix)
    # all scores against batch size (filtered by model) and averaged across all iterations
    _create_figure2(iteration_to_batch, prefix)

    # prepare results for figure 3,4,5,6
    model_to_batch_size_to_avgs, \
    model_to_batch_size_to_turnaround, \
    model_to_batch_size_to_rouge, \
    model_to_batch_size_to_meteor = _prep_figure3_4_5_6(result)
    # average inference time per frame against batch size
    # filtered by model
    _create_figure3(model_to_batch_size_to_avgs, prefix)
    # combined
    _create_figure3_combined(model_to_batch_size_to_avgs, prefix)
    # total inference time (turnaround time) against batch size (filtered by model)
    # filtered by model
    _create_figure4(model_to_batch_size_to_turnaround, prefix)
    # combined
    _create_figure4_combined(model_to_batch_size_to_turnaround, prefix)

    # rouge score against batch size (all models shown)
    _create_figure5_combined(model_to_batch_size_to_rouge, prefix)
    # meteor score against batch size (all models shown)
    _create_figure6_combined(model_to_batch_size_to_meteor, prefix)

def main(args):
    if os.path.exists("result.pkl"):
        with open("result.pkl", "rb") as f:
            result = pickle.load(f)
    else:
        truth = get_truth()
        result = process_results(truth)
        with open("result.pkl", 'wb') as f:
            pickle.dump(result, f)
    create_figures(result[0], "hugging_face")
    create_figures(result[1], "vllm")

    # inference time against batch size (filtered by common models) and both hf and vlm are shown
    _create_figure7(*[_prep_figure3_4_5_6(r)[0] for r in result])

if __name__ == "__main__":
    main(sys.argv)
