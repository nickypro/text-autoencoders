# %%
import os
import json
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import plotly.graph_objects as go
from scipy import stats
from sklearn.metrics import cohen_kappa_score


def load_result(result: str | dict) -> dict:
    if isinstance(result, str):
        return json.loads(result)
    else:
        return result

def get_scores(data_list: list[dict]) -> list[dict]:
    # Convert string JSON to dict if needed
    scores = []
    for idx, item in enumerate(data_list):
        result = load_result(item['result'])
        scores.append(result["scoring"])
    return scores


def process_scores(data_list: list[dict], metric: str) -> list[float]:
    # Convert string JSON to dict if needed
    scores = []
    for item in get_scores(data_list):
        score = item[metric]
        # if item["coherence"] < 2:
        #     continue
        # if score == -1:
        #     continue
        scores.append(score)
    return scores


# Loading data
##############

def run_references_match_check(data_dicts: dict, verbose=False):
    references = {}
    for data_type, data_dict in data_dicts.items():
        for index, item in data_dict.items():
            references[int(index)] = item["reference"]
        break

    for data_type, data_dict in data_dicts.items():
        for index, item in data_dict.items():
            if item["reference"] != references[int(index)]:
                if verbose:
                    print(f"{data_type} {index} {item['reference']} != {references[int(index)]}")

def run_short_indices_check(data_dicts, verbose=False):
    get_len = lambda x, key: int(np.mean([len(v[key]) for v in x.values()]))

    a = data_dicts["regenerated"]
    b = data_dicts["cheat-10"]
    indices_a = [item["index"] for item in a.values()]
    indices_b = [item["index"] for item in b.values()]
    short_indices = set(indices_a) - set(indices_b)
    if verbose:
        print("Num short indices:", len(short_indices))
        for idx in sorted(short_indices)[:10]:
            print(idx, {"short": a[str(idx)]["reference"]})

def get_data_dict_with_common_indices(data_dicts, compare_len=True, verbose=False):
    if compare_len:
        # check number of items in each data_dict before
        len_before = [len(v) for v in data_dicts.values()]

        # save average length of reference and comparison before
        mean_lens = {}
        get_len = lambda x, key: int(np.mean([len(v[key]) for v in x.values()]))
        for type, data_dict in data_dicts.items():
            mean_lens[type] = (get_len(data_dict, "reference"), get_len(data_dict, "comparison"))


    data_dicts = data_dicts.copy()
    indices = {}
    for k, v in data_dicts.items():
        indices[k] = [item["index"] for item in v.values()]

    common_indices = None
    for k, v in indices.items():
        if common_indices is None:
            common_indices = set(v)
        common_indices.intersection_update(v)

    for ref_type, data_dict in data_dicts.items():
        data_dicts[ref_type] = {str(k): v for k, v in data_dict.items() if int(k) in common_indices}


    if compare_len:
        len_after = [len(v) for v in data_dicts.values()]
        for idx, (type, data_dict) in enumerate(data_dicts.items()):
            mean_len_curr = (get_len(data_dict, "reference"), get_len(data_dict, "comparison"))
            if verbose:
                print(f"{type:12}"
                      f" Ref comp: {mean_lens[ref_type]} -> {mean_len_curr}"
                      f" Len: {len_before[idx]:5} -> {len_after[idx]:5}")
    return data_dicts

def load_rubric_results(file_path="processed_rubrics/all_data_dicts.json",
        indices_intersection=False,
        check_short_indices=False,
        check_references_match=False,
        verbose=False,
    ):
    with open(file_path, "r") as f:
        data_dicts = json.load(f)


    if check_short_indices:
        verbose and print("Checking short indices")
        run_short_indices_check(data_dicts, verbose=verbose)

    if indices_intersection:
        verbose and print("Getting data_dicts with common indices")
        data_dicts = get_data_dict_with_common_indices(data_dicts, verbose=verbose)

    if check_references_match:
        verbose and print("Checking references match")
        assert indices_intersection, "check_references_match requires indices_intersection"
        run_references_match_check(data_dicts, verbose=verbose)
    return data_dicts

def calculate_score_proportions(scores, cumulative=False):
    total = len(scores)
    if total == 0:
        return []

    # Get unique possible scores and sort them
    unique_scores = sorted(set(scores))
    proportions = []

    if cumulative:
        # Calculate proportion >= each score
        for threshold in unique_scores:
            count = sum(1 for score in scores if score >= threshold)
            proportions.append(count / total)
    else:
        # Calculate proportion = each score
        for score in unique_scores:
            count = sum(1 for s in scores if s == score)
            proportions.append(count / total)

    return proportions

def plot_score_proportions(data_dicts, metric, output_image=None):
    plt.figure(figsize=(12, 6))

    # Process each comparison type
    for label, data_dict in data_dicts.items():
        # Load and process data
        data_list = list(data_dict.values())
        scores = process_scores(data_list, metric)

        proportions = calculate_score_proportions(scores)

        # Get unique scores for x-axis
        unique_scores = sorted(set(scores))

        # Plot as lines
        # Plot stacked bars for each score threshold
        # Create a base color for this label using a consistent mapping
        label_index = list(data_dicts.keys()).index(label)
        base_color = plt.cm.Pastel1(label_index / len(data_dicts))

        bottom = 0
        for i, prop in reversed(list(enumerate(proportions))):
            # Darken the base color based on score level
            darkness = 1 - (i/len(proportions))
            color = tuple(c * darkness for c in base_color[:3]) + (base_color[3],)

            bar = plt.bar([label], [prop], bottom=bottom,
                   label=f'{label} (≥{unique_scores[i]})',
                   color=color)
            plt.text(bar[0].get_x() + bar[0].get_width()/2, bottom + prop/2,
                    str(unique_scores[i]),
                    ha='center', va='center')
            bottom += prop

    plt.xlabel(f'{metric} Score Threshold')
    plt.ylabel('Proportion >= Score')
    plt.title(f'Cumulative Score Distribution for {metric.capitalize()} ({unique_scores[0]} to {unique_scores[-1]})')
    plt.grid(True, alpha=0.3)
    # plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()

    if output_image:
        plt.savefig(output_image, dpi=300, bbox_inches='tight')
    plt.show()

def get_examples(data_dict, metric, shuffle=True, limit=10):
    examples = defaultdict(list)
    for index, item in data_dict.items():
        score = load_result(item["result"])["scoring"][metric]
        examples[score].append(item)
    if shuffle:
        for score in examples:
            random.shuffle(examples[score])
    if limit is not None:
        for score in examples:
            examples[score] = examples[score][:limit]
    return examples

def plot_score_proportions_interactive(data_dicts, metric):
    fig = go.Figure()

    for label, data_dict in data_dicts.items():
        data_list = list(data_dict.values())
        scores = process_scores(data_list, metric)
        proportions = calculate_score_proportions(scores)
        unique_scores = sorted(set(scores))
        examples = get_examples(data_dict, metric, limit=5) # examples[score]["reference"]

        label_index = list(data_dicts.keys()).index(label)

        bottom = 0
        for i, prop in reversed(list(enumerate(proportions))):
            #base_color = f'rgba({label_index * 50 % 255}, {label_index * 80 %
            #255}, {label_index * 110 % 255}, 0.6)'
            # base_color = f'rgba({label_index * 50 % 255}, {label_index * 80 % 255}, {label_index * 110 % 255}, {(i+1)/len(proportions)})'
            base_color = f'hsla({label_index * 150 % 360}, 50%, 50%, {(i+1)/len(proportions)})'

            # Safely access examples
            hover_text = f"<br><b>Score:</b> {unique_scores[i]}"
            for example in examples[unique_scores[i]]:
                hover_text += (
                    f"<br><b>Reference:</b> {example['reference'][:80]}"
                    +f"<br><b>Comparison:</b> {example['comparison'][:80]}<br>"
                )

            fig.add_trace(go.Bar(
                x=[label],
                y=[prop],
                base=bottom,
                name=f'{label} (≥{unique_scores[i]})',
                marker_color=base_color,
                hoverinfo='text',
                hovertext=hover_text,
                text=f'{unique_scores[i]}',  # Add number in middle
                textposition='inside',  # Position text in middle of bar
                textfont=dict(size=10),  # Set consistent font size
            ))
            bottom += prop

    fig.update_layout(
        title=f'Cumulative Score Distribution for {metric.capitalize()} ({unique_scores[0]} to {unique_scores[-1]})',
        xaxis_title=f'{metric} Score Threshold',
        yaxis_title='Proportion >= Score',
        barmode='stack'
    )

    # hide legend
    fig.update_layout(showlegend=False)

    # save png to ./figures
    fig.write_image(f"./figures/score_distribution_{metric}.png")

    fig.show(renderer="notebook_connected")

if __name__ == "__main__":

    data_dicts = load_rubric_results(
        file_path="processed_rubrics/all_data_dicts.json",
        indices_intersection=True,
        check_short_indices=True,
        check_references_match=True,
        verbose=True,
    )

    metrics = ["complexity", "coherence", "structure", "subject", "entities", "details", "terminology", "tone", "length"]

    print("Loaded data_dicts:", list(data_dicts.keys()))

# %%
