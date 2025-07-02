from networkx import Graph
import networkx as nx
import numpy as np
import csv


def row_check(file:str):
    """
    check the number of rows in a csv file
    :param file: csv file to check
    :return: number of rows
    """
    rows = 0
    with open(file, "r", encoding="utf-8") as row_file:
        row_reader = csv.reader(row_file)
        for row in row_reader:
            rows += 1

    return rows


def find_best_graph_specialties(graph:Graph, n=10):
    """
    find the most common specialties in the graph
    :param graph: networkx graph with specialty node attribute
    :param n: number of specialties to return
    :return: top n most common specialties in a list
    """
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
    """
    get graph statistics
    :param graph: networkx graph
    :param specialties_to_analyze: list of specialties to include individually
    :param output_file: file to write results to, csv
    :return:
    """
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

        with open(output_file, "w", newline="") as output:
            output_writer = csv.writer(output)
            rows = [["Specialty", "Node Number", "Edge Number", "Avg Degree", "Avg Specialties"],
                    ["Entire Graph", node_num, edge_num, mean_degree, mean_specialties]]
            specialty_rows = [[specialty, per_specialty[specialty]["node_num"], per_specialty[specialty]["edge_num"],
                               per_specialty[specialty]["mean_degree"], per_specialty[specialty]["mean_specialties"]]
                              for specialty in per_specialty]

            rows.extend(specialty_rows)

            output_writer.writerows(rows)

    else:
        with open(output_file, "w", newline="") as output:
            output_writer = csv.writer(output)
            rows = [["Specialty", "Node Number", "Edge Number", "Avg Degree", "Avg Specialties"],
                    ["Entire Graph", node_num, edge_num, mean_degree, mean_specialties]]

            output_writer.writerows(rows)

    print(f"graph statistics saved to {output_file}")