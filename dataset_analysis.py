from networkx import Graph
import networkx as nx
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

def row_check(file:str):
    rows = 0
    with open(file, "r", encoding="utf-8") as row_file:
        row_reader = csv.reader(row_file)
        for row in row_reader:
            rows += 1

    return rows

def find_best_graph_specialties(graph:Graph, n=10):
    spec_nums = {}
    for node in graph:
        for specialty in graph.nodes[node]["specialties"]:
            if specialty in spec_nums:
                spec_nums[specialty] += 1
            else:
                spec_nums[specialty] = 1

    ordered_specs = sorted(list(spec_nums.items()), key=lambda item: item[1], reverse=True)

    # suggested top 10
    # [('207R00000X', 44657), ('207Q00000X', 31036), ('207RC0000X', 14335), ('207P00000X', 13557),
    # ('363A00000X', 11946), ('2085R0202X', 10374), ('363LF0000X', 10303), ('208600000X', 9697),
    # ('367500000X', 9331), ('207X00000X', 8442)]
    return ordered_specs[:n]

def get_graph_information(graph:Graph, specialties_to_analyze:list=None, output_file="./results/graph_information.csv"):
    nodes = graph.nodes
    node_num = graph.number_of_nodes()
    edge_num = graph.number_of_edges()
    mean_degree = sum(dict(graph.degree()).values()) / graph.number_of_nodes()
    mean_specialties = 0

    for node in nodes:
        specialties = nodes[node]["specialties"]
        mean_specialties += len(specialties)

    mean_specialties /= node_num

    # per specialty processing
    if specialties_to_analyze:
        per_specialty = {specialty: {"node_num": 0,
                                     "edge_num": 0,
                                     "mean_degree": 0,
                                     "mean_specialties": 0}
                         for specialty in specialties_to_analyze}

        for node in nodes:
            for specialty in nodes[node]["specialties"]:
                if specialty in per_specialty:
                    per_specialty[specialty]["node_num"] += 1
                    per_specialty[specialty]["mean_specialties"] += len(nodes[node]["specialties"])
                    per_specialty[specialty]["mean_degree"] += graph.degree(node)

        for u, v in graph.edges:
            both_specialties = set(nodes[u]["specialties"] + nodes[v]["specialties"])
            for specialty in both_specialties:
                if specialty in per_specialty:
                    per_specialty[specialty]["edge_num"] += 1

        for specialty in per_specialty:
            per_specialty[specialty]["mean_degree"] /= per_specialty[specialty]["node_num"]
            per_specialty[specialty]["mean_specialties"] /= per_specialty[specialty]["node_num"]

        with open(output_file, "w") as output:
            output_writer = csv.writer(output)
            rows = [["Specialty", "Node Number", "Edge Number", "Avg Degree", "Avg Specialties"],
                    ["Entire Graph", node_num, edge_num, mean_degree, mean_specialties]]
            specialty_rows = [[specialty, per_specialty[specialty]["node_num"], per_specialty[specialty]["edge_num"],
                               per_specialty[specialty]["mean_degree"], per_specialty[specialty]["mean_specialties"]]
                              for specialty in per_specialty]

            rows.extend(specialty_rows)

            output_writer.writerows(rows)

    else:
        with open(output_file, "w") as output:
            output_writer = csv.writer(output)
            rows = [["Specialty", "Node Number", "Edge Number", "Avg Degree", "Avg Specialties"],
                    ["Entire Graph", node_num, edge_num, mean_degree, mean_specialties]]

            output_writer.writerows(rows)





if __name__ == "__main__":
    print(row_check("./datasets/specialty_2018.csv"))