from networkx import Graph
import numpy as np
import csv


def get_score_correlation(graph:Graph, score_file:str="./datasets/pa_scores_2017.csv"):
    node_scores = {}
    with open(score_file, "r") as data_file:
        scores_csv = csv.reader(data_file)
        next(scores_csv)
        for line in scores_csv:
            provider = int(line[5].strip())
            # quality=24, pi=47, ia=75, cost=85, mip=20
            if provider in graph.nodes:
                final = (line[20].strip())
                quality = (line[24].strip())
                pi = (line[47].strip())
                ia = (line[75].strip())
                cost = (line[85].strip())
                node_scores[provider] = [final, quality, pi, ia, cost]
                converted = []
                for score in node_scores[provider]:
                    if score:
                        converted.append(float(score))
                    else:
                        converted.append(0)
                node_scores[provider] = converted

    correlation_dict = {}
    for i, score_type in enumerate(["mips", "quality", "pi", "ia", "cost"]):
        correlation_dict[score_type] = [[[], [], []], []]
        for node in node_scores:
            p_total = graph.nodes[node]["pair_total"]
            b_total = graph.nodes[node]["beneficiary_total"]
            s_total = graph.nodes[node]["same_total"]

            correlation_dict[score_type][0][0].append(p_total)
            correlation_dict[score_type][0][1].append(b_total)
            correlation_dict[score_type][0][2].append(s_total)

            correlation_dict[score_type][1].append(node_scores[node][i])

    for score_type in correlation_dict:
        for total_version, total_type in zip(correlation_dict[score_type][0], ["pair", "bene", "same"]):
            x = total_version
            y = correlation_dict[score_type][1]
            correlation_matrix = np.corrcoef(x, y)
            correlation_coefficient = correlation_matrix[0, 1]
            print(f"{score_type} correlation for {total_type}: {correlation_coefficient}")