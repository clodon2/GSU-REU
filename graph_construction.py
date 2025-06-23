import networkx as nx
import time
import csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps


class GraphBuilder:
    def __init__(self, primary_specialty_weight:float=2,
                 provider_data_file:str="./datasets/pa_data.txt",
                 specialty_data_file:str="./datasets/specialty_2015_reformatted.csv",
                 graph_data_file:str="./datasets/physician_graph.gexf"):
        """
        create provider graph manager
        :param primary_specialty_weight: the weight given to a provider's main specialty
        :param provider_data_file: data file to load provider edges from (nodes created also)
        :param specialty_data_file: data file to load specialty data from for providers
        :param graph_data_file: data file to save/load graph to
        """
        self.primary_specialty_weight = primary_specialty_weight
        self.provider_data_file = provider_data_file
        self.provider_specialty_data_file = specialty_data_file
        self.graph_data_file = graph_data_file
        self.graph = nx.Graph()
        self.coboundary_columns = 0

    def import_txt_data(self, rows:int=500):
        """
        add data from the provider txt dataset to self.graph
        :param rows: number of rows of data to add (edges)
        :return: None
        """
        print("importing provider data...")
        start = time.time()
        lines_read = 0
        with open(self.provider_data_file, "r") as data:
            for line in data:
                # line format:
                # npi1, npi2, pair count, beneficiary count, same day count
                lines_read += 1
                # extract data
                row_data = line.split(",")
                provider1 = int(row_data[0].strip())
                provider2 = int(row_data[1].strip())
                pairs = int(row_data[2].strip())
                benes = int(row_data[3].strip())
                sameday = int(row_data[4].strip())

                self.graph.add_edge(provider1, provider2, weight=pairs, beneficiaries=benes, same_day=sameday)

                # stop at however many rows
                if lines_read >= rows:
                    break

        end = time.time()

        print(f"{lines_read} edges added in {end - start}")

    def add_specialties_fast(self):
        """
        add specialties to all possible nodes, delete those without specialties
        :return:
        """
        print("adding specialties...")
        start = time.time()
        line_count = 0
        with open(self.provider_specialty_data_file, "r") as data:
            csv_reader = csv.reader(data)
            for line in csv_reader:
                line_count += 1
                provider = int(line[0])
                if self.graph.has_node(provider):
                    specialties = []
                    for sc in line[1:-1]:
                        if sc:
                            specialties.append(sc)

                    self.graph.nodes[provider]["specialties"] = specialties
                    self.graph.nodes[provider]["primary"] = line[-1]

        end = time.time()
        print(f"{line_count} in reformatted csv")
        print(f"finished in {end - start}")

        remove_nodes = []
        for node in self.graph.nodes:
            try:
                spec = self.graph.nodes[node]["specialties"]
                prim = self.graph.nodes[node]["primary"]
                if not spec:
                    remove_nodes.append(node)
            except:
                remove_nodes.append(node)
                print(f"no specialties for node {node}")

        for node in remove_nodes:
            self.graph.remove_node(node)

        print(f"{len(remove_nodes)} no specialty nodes removed")

    def build_graph(self, rows=999999999999999999, remove_unscored_nodes_file='',
                    remove_non_overlap_spec_file=''):
        """
        create graph structure for providers and add specialties
        :return: the graph
        """
        print("building graph...")
        start = time.time()
        self.import_txt_data(rows=rows)
        if remove_unscored_nodes_file:
            self.remove_unscored_nodes_new(remove_unscored_nodes_file)
        self.add_specialties_fast()
        if remove_non_overlap_spec_file:
            self.remove_non_overlap_specialties(remove_non_overlap_spec_file)
        self.remove_other_connections()
        self.sheaf_specialty_conversion()
        self.add_provider_totals()
        end = time.time()
        print(f"graph built in {end - start}")
        return self.graph

    def get_graph_stats(self):
        print("Nodes:", self.graph.number_of_nodes())
        print("Edges:", self.graph.number_of_edges())

    def save_graph(self):
        """
        save graph locally as graphml file, expensive
        :return:
        """
        print("writing graph...")
        nx.write_graphml(self.graph, self.graph_data_file)

    def load_graph(self):
        """
        update graph to match graph file
        :return:
        """
        print("importing graph...")
        self.graph = nx.read_graphml(self.graph_data_file)

    def sheaf_specialty_conversion(self):
        """
        add an array to node vertices that stores numerical conversion of specialty list
        :return:
        """
        print("adding numerical specialty vectors...")
        start = time.time()
        for node in self.graph.nodes:
            num_specialties = []
            for spec in self.graph.nodes[node]["specialties"]:
                # weight the primary specialty more
                if spec == self.graph.nodes[node]["primary"]:
                    num_specialties.append(self.primary_specialty_weight)
                else:
                    num_specialties.append(1)

            # add to node
            self.graph.nodes[node]["sheaf_vector"] = np.array(num_specialties)
        end = time.time()
        print(f"specialty numerical finished in {end - start}")

    def add_provider_totals(self):
        """
        total values for all edges a node is connected to, store in node as attribute
        :return:
        """
        print("adding edge totals to providers...")
        start = time.time()
        col = 0
        for node in self.graph.nodes:
            # add indices to add in coboundary map later
            spec_num = len(self.graph.nodes[node]["specialties"])
            self.graph.nodes[node]["indices"] = list(range(col, col + spec_num))
            self.graph.nodes[node]["edge_indices"] = []
            # start at next free index
            col += spec_num + 1
            # start with one to avoid division by 0 in add_coboundary_matrices
            self.graph.nodes[node]["pair_total"] = 1
            self.graph.nodes[node]["beneficiary_total"] = 1
            self.graph.nodes[node]["same_total"] = 1
            for connection in self.graph[node]:
                pairs = self.graph[node][connection]["weight"]
                benes = self.graph[node][connection]["beneficiaries"]
                same_days = self.graph[node][connection]["same_day"]

                self.graph.nodes[node]["pair_total"] += pairs
                self.graph.nodes[node]["beneficiary_total"] += benes
                self.graph.nodes[node]["same_total"] += same_days

        self.coboundary_columns = col
        end = time.time()
        print(f"totals calculated in {end - start}")

    def remove_unscored_nodes(self, score_file_name):
        """
        removes nodes that do not have a score in the scoring dataset from the graph
        :param score_file_name: the score dataset filename
        :return:
        """
        print("removing unscored nodes...")
        valid_providers = set()
        with open(score_file_name, "r") as rank_file:
            rank_file = csv.reader(rank_file)
            next(rank_file)
            for line in rank_file:
                provider = int(line[0].strip())
                valid_providers.add(provider)

        unscored_nodes = [node for node in self.graph.nodes if node not in valid_providers]
        print(f"removed {len(unscored_nodes)} no score nodes")

        self.graph.remove_nodes_from(unscored_nodes)

    def remove_unscored_nodes_new(self, score_file_name,
                                  score_specialty_file="./datasets/specialty_2018_reformatted.csv"):
        """
        removes nodes that do not have a score in the scoring dataset from the graph
        :param score_file_name: the score dataset filename
        :return:
        """
        print("removing unscored nodes...")
        specialty_info = {}
        with open(score_specialty_file, "r") as spec_file:
            spec_file = csv.reader(spec_file)
            next(spec_file)
            for line in spec_file:
                npi = int(line[0].strip())
                primary = line[-1].strip()
                specialty_info[npi] = primary

        valid_providers = set()
        with open(score_file_name, "r") as rank_file:
            rank_file = csv.reader(rank_file)
            next(rank_file)
            for line in rank_file:
                provider = int(line[5].strip())
                # quality=24, pi=47, ia=75, cost=85, mip=20
                # 3, 4, 5, 6, 8 old
                primary_spec = specialty_info[npi]
                quality = line[24].strip()
                pi = line[47].strip()
                ia = line[75].strip()
                cost = line[85].strip()
                final = line[20].strip()
                scores = [quality, pi, ia, cost, final]
                converted = []
                for score in scores:
                    new_score = 0
                    if score:
                        new_score = float(score)
                    converted.append(new_score)
                # provider for new dataset: 5 old: 0
                if converted[0] and converted[1] and converted[2] and converted[3] and converted[4] and primary_spec:
                    valid_providers.add(provider)

        print(len(valid_providers))
        unscored_nodes = [node for node in self.graph.nodes if node not in valid_providers]
        print(f"removed {len(unscored_nodes)} no score nodes")

        self.graph.remove_nodes_from(unscored_nodes)

    def remove_small_connections(self):
        """
        removes any weakly connected components from the graph
        :return:
        """
        components = list(nx.connected_components(self.graph))
        for component in components:
            if len(component) <= (.05 * self.graph.number_of_nodes()):
                self.graph.remove_nodes_from(component)

    def remove_other_connections(self):
        num_nodes = self.graph.number_of_nodes()
        components = list(nx.connected_components(self.graph))
        for i, component in enumerate(components):
            if i > 0:
                self.graph.remove_nodes_from(component)

        print(len(components[0]), "cc")

        print(num_nodes - self.graph.number_of_nodes(), "unconnected nodes removed")

    def remove_non_overlap_specialties(self, specialty_file):
        print("removing non overlap nodes...")
        specialty_dict = {}
        with open(specialty_file, "r") as actual_specialty_info:
            actual_specialty_info = csv.reader(actual_specialty_info)
            next(actual_specialty_info)
            for row in actual_specialty_info:
                npi = int(row[0].strip())
                specialty_dict[npi] = []
                for sc in row[1:-1]:
                    if sc:
                        specialty_dict[npi].append(sc)

                specialty_dict[npi].append(row[-1])

        remove_nodes = []
        for npi in self.graph.nodes:
            if npi in specialty_dict:
                overlapping_specialties = set(specialty_dict[npi][:-1]) | set(self.graph.nodes[npi]["specialties"])
                if not overlapping_specialties or specialty_dict[npi][-1] != self.graph.nodes[npi]["primary"]:
                    remove_nodes.append(npi)
                else:
                    self.graph.nodes[npi]["specialties"] = overlapping_specialties
            else:
                remove_nodes.append(npi)

        self.graph.remove_nodes_from(remove_nodes)
        print(f"removed spec different {len(remove_nodes)} from graph")


    def draw_graph(self, edge_colors: bool = True, edge_labels: bool = True):
        """
        draw self.graph in a new window
        :param edge_colors: color edges based on intensity (darker larger weight)
        :param edge_labels: label weights of edges
        :return:
        """
        # best layouts
        # can look confusing
        pos = nx.spring_layout(self.graph, seed=7443, k=20)
        # probably the best but intensive
        # pos = nx.kamada_kawai_layout(self.graph)
        # circular (like a better spring)
        # pos = nx.forceatlas2_layout(self.graph)
        # good variety of distances but can look crowded
        pos = nx.fruchterman_reingold_layout(self.graph)

        if edge_colors:
            color_map = colormaps["Blues"]
            weights = [self.graph[u][v]["weight"] for u, v in self.graph.edges()]
            # scale color map to maximum weight in graph
            # min_weight = min(weights)
            max_weight = max(weights)
        else:
            color_map = None
            weights = None
            max_weight = 0

        if edge_labels:
            edge_weights = nx.get_edge_attributes(self.graph, "weight")
            nx.draw_networkx_edge_labels(self.graph, pos, edge_weights)

        nx.draw_networkx(self.graph, pos=pos, with_labels=True,
                         edge_color=weights, edge_cmap=color_map, edge_vmin=0, edge_vmax=max_weight)
        plt.show()