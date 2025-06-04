import networkx as nx
import matplotlib.pyplot as plt
from matplotlib import colormaps
import csv
import numpy as np
import scipy as sp
import time
from multiprocessing import Pool
from data_comparison import CompareData, EvaluationMethods


class ProviderConnections:
    def __init__(self, primary_specialty_weight:float=2, restriction_weights:list=[1, 1, 1],
                 provider_data_file:str="pa_data.txt",
                 specialty_data_file:str="specialty_reformatted.csv",
                 graph_data_file:str="physician_graph.gexf"):
        """
        create provider graph manager
        :param primary_specialty_weight: the weight given to a provider's main specialty
        :param restriction_weights: [pair count weight, beneficiary count weight, same day count weight]
        :param provider_data_file: data file to load provider edges from (nodes created also)
        :param specialty_data_file: data file to load specialty data from for providers
        :param graph_data_file: data file to save/load graph to
        """
        self.primary_specialty_weight = primary_specialty_weight
        self.restriction_weights = restriction_weights
        self.provider_data_file = provider_data_file
        self.provider_specialty_data_file = specialty_data_file
        self.graph_data_file = graph_data_file
        self.graph = nx.Graph()
        self.coboundary_columns = 0
        self.coboundary_map = None
        self.sheaf_laplacian = None
        self.rankings = {}

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
            except:
                remove_nodes.append(node)
                print(f"no specialties for node {node}")

        for node in remove_nodes:
            self.graph.remove_node(node)

        print(f"{len(remove_nodes)} no specialty nodes removed")

    def build_graph(self, rows=999999999999999999):
        """
        create graph structure for providers and add specialties
        :return: the graph
        """
        print("building graph...")
        start = time.time()
        self.import_txt_data(rows=rows)
        self.add_specialties_fast()
        self.sheaf_specialty_conversion()
        self.add_provider_totals()
        end = time.time()
        print(f"graph built in {end - start}")
        return self.graph

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

    def compute_coboundary_map(self):
        """
        add coboundary map to each edge, based on each provider's unique restriction map for each edge
        :return:
        """
        print("computing coboundary map...")
        start = time.time()
        nonzero_restrictions = []
        nzr_row_indices = []
        nzr_column_indices = []
        #coboundary_map = sp.sparse.lil_matrix((len(self.graph.edges), self.coboundary_columns), dtype=np.float64)
        for i, edge in enumerate(self.graph.edges):
            # get edge specific values
            edge_attr = self.graph.get_edge_data(edge[0], edge[1])
            edge_pairs = edge_attr["weight"]
            edge_benes = edge_attr["beneficiaries"]
            edge_same_days = edge_attr["same_day"]
            #edge_restrictions = []
            for provider in edge:
                # get restriction maps based on provider edge percentages
                pair_percentage = self.restriction_weights[0] * (edge_pairs / self.graph.nodes[provider]["pair_total"])
                bene_total = self.restriction_weights[1] * (edge_benes / self.graph.nodes[provider]["beneficiary_total"])
                same_day_total = self.restriction_weights[2] * (edge_same_days / self.graph.nodes[provider]["same_total"])
                restriction = np.array([pair_percentage, bene_total, same_day_total])

                # add info to array for sparse matrix conversion
                nonzero_restrictions.extend((self.graph.nodes[provider]["sheaf_vector"] * np.sum(restriction)).tolist())
                nzr_column_indices.extend(self.graph.nodes[provider]["indices"])
                for input_num in range(len(self.graph.nodes[provider]["indices"])):
                    nzr_row_indices.append(i)

        coboundary_map = sp.sparse.csr_matrix((nonzero_restrictions, (nzr_row_indices, nzr_column_indices)),
                                              shape=(len(self.graph.edges), self.coboundary_columns))

        self.coboundary_map = coboundary_map
        end = time.time()
        print(f"coboundary map found in {end - start}")

        return coboundary_map

    def compute_sheaf_laplacian(self):
        """
        compute sheaf laplacian (transposed coboundary map * og coboundary map)
        :return:
        """
        print("computing sheaf laplacian...")
        start = time.time()
        coboundary_map_t = self.coboundary_map.transpose()

        # multiply by transposition
        sheaf_lap = coboundary_map_t.dot(self.coboundary_map)

        self.sheaf_laplacian = sheaf_lap
        end = time.time()

        print(f"sheaf laplacian done in {end - start}")

        return sheaf_lap

    def compute_centrality(self, values_to_consider=5):
        """
        get centrality score for each provider
        :param values_to_consider: number of eigenvalues to include in the calculation, (number in most influential)
        :return:
        """
        print("computing sheaf laplacian energy...")
        start = time.time()
        eigenvalues, eigenvectors = sp.sparse.linalg.eigsh(self.sheaf_laplacian, k=values_to_consider, which="LM")
        sheaf_laplacian_energy = sum(val**2 for val in eigenvalues)
        print(f"sheaf laplacian energy", sheaf_laplacian_energy)
        print("computing sheaf laplacian centralities")
        for i, node in enumerate(self.graph.nodes):
            self.graph.nodes[node]["centralities"] = []
            print(f"node {i} of {len(self.graph.nodes)}")
            # for each specialty, get the centrality score
            for specialty, specialty_name in zip(self.graph.nodes[node]["indices"], self.graph.nodes[node]["specialties"]):
                # have to convert to lil matrix for specialty removal
                lil_sheaf_laplacian = self.sheaf_laplacian.tolil()
                # remove specialty (subtract this centrality by initial for centrality of speciality--impact
                for row in range(lil_sheaf_laplacian.shape[0]):
                    lil_sheaf_laplacian[row, specialty] = 0
                # convert back for matrix operations
                spec_removed_sheaf_laplacian = lil_sheaf_laplacian.tocsr()
                spec_removed_sheaf_laplacian.eliminate_zeros()
                # eigsh used because matrix is symmetric
                eigenvalues, eigenvectors = sp.sparse.linalg.eigsh(spec_removed_sheaf_laplacian, k=values_to_consider,
                                                                  which="LM")
                spec_sheaf_laplacian_energy = sum(val**2 for val in eigenvalues)
                # centrality (impact) for each specialty of each node
                centrality = (sheaf_laplacian_energy - spec_sheaf_laplacian_energy) / sheaf_laplacian_energy
                # can probs remove this storage
                self.graph.nodes[node]["centralities"].append(centrality)

                # add to rankings for specialty
                if specialty_name in self.rankings:
                    self.rankings[specialty_name][node] = centrality
                else:
                    self.rankings[specialty_name] = {}
                    self.rankings[specialty_name][node] = centrality

        end = time.time()
        print(f"energies found in {end - start}")

    def compute_centrality_multiprocessing_helper(self, sheaf_laplacian_energy, node, values_to_consider=5):
        node_centralities = []
        self.graph.nodes[node]["centralities"] = []
        # for each specialty, get the centrality score
        for specialty, specialty_name in zip(self.graph.nodes[node]["indices"], self.graph.nodes[node]["specialties"]):
            # have to convert to lil matrix for specialty removal
            lil_sheaf_laplacian = self.sheaf_laplacian.tolil()
            # remove specialty (subtract this centrality by initial for centrality of speciality--impact
            for row in range(lil_sheaf_laplacian.shape[0]):
                lil_sheaf_laplacian[row, specialty] = 0
            # convert back for matrix operations
            spec_removed_sheaf_laplacian = lil_sheaf_laplacian.tocsr()
            spec_removed_sheaf_laplacian.eliminate_zeros()
            # eigsh used because matrix is symmetric
            eigenvalues, eigenvectors = sp.sparse.linalg.eigsh(spec_removed_sheaf_laplacian, k=values_to_consider,
                                                              which="LM")
            spec_sheaf_laplacian_energy = sum(val**2 for val in eigenvalues)
            # centrality (impact) for each specialty of each node
            centrality = (sheaf_laplacian_energy - spec_sheaf_laplacian_energy) / sheaf_laplacian_energy
            node_centralities.append(centrality)

        return node, node_centralities

    def compute_centrality_multiprocessing(self, values_to_consider=None):
        """
        get centrality score for each provider
        :param values_to_consider: number of eigenvalues to include in the calculation, (number in most influential)
        :return:
        """
        if values_to_consider == None:
            values_to_consider = self.sheaf_laplacian.shape[0] - 1
        print("computing sheaf laplacian energy...")
        start = time.time()
        eigenvalues, eigenvectors = sp.sparse.linalg.eigsh(self.sheaf_laplacian, k=values_to_consider, which="LM")
        sheaf_laplacian_energy = sum(val**2 for val in eigenvalues)
        print(f"sheaf laplacian energy", sheaf_laplacian_energy)
        print("computing sheaf laplacian centralities")
        print(f"time estimation: {(time.time() - start) * len(self.graph.nodes)}")
        pool_args = []
        for node in self.graph.nodes:
            pool_args.append((sheaf_laplacian_energy, node, values_to_consider))
        with Pool(processes=8) as pool:
            results = pool.starmap(self.compute_centrality_multiprocessing_helper, pool_args)

        # add results to ranking dict
        for entry in results:
            node = entry[0]
            centralities = entry[1]
            for specialty, centrality in zip(self.graph.nodes[node]["specialties"], centralities):
                # add to rankings for specialty
                if specialty in self.rankings:
                    self.rankings[specialty][node] = centrality
                else:
                    self.rankings[specialty] = {}
                    self.rankings[specialty][node] = centrality

        end = time.time()
        print(f"energies found in {end - start}")

    def get_ranking(self):
        """
        get rankings of providers based on specialties
        :return:
        """
        compare = CompareData()
        sorted_rankings = {}
        for specialty in self.rankings:
            values = self.rankings[specialty]
            # reorder to see best provider
            sorted_rankings[specialty] = sorted(values.items(), key=lambda item: item[1], reverse=True)
        specs = compare.compare(self.graph, sorted_rankings, title="Sheaf Laplacian", show_lists=False)
        """
        for specialty in self.rankings:
            print(specialty)
            values = self.rankings[specialty]
            # reorder to see best provider
            sorted_rankings = sorted(values.items(), key=lambda item: item[1])
            np.set_printoptions(suppress=True)
            # round (useless rn)
            readable_rankings = []
            for t in sorted_rankings:
                readable_rankings.append((t[0], round(t[1], 8)))
            print(readable_rankings)
        """
        return specs

    def compute_all_give_rankings(self):
        """
        compute everything needed to get rankings and print them
        :return:
        """
        print("computing all for ranking...")
        self.compute_coboundary_map()
        self.compute_sheaf_laplacian()
        self.compute_centrality_multiprocessing(values_to_consider=None)
        specs = self.get_ranking()

        return specs

    def add_test_data(self):
        test_edges = [("v1", "v2"), ("v3", "v2"), ("v3", "v4"), ("v1", "v4")]

        test_matrix = [
            [-1, -2, 1, 0, 0, 0],
            [0, -2, 3. -1, 0, 0],
            [0, 0, 0, 3, -1, 1],
            [2, 0, 0, 0, -1, 0]
        ]

        self.coboundary_map = sp.sparse.csr_matrix(([-1, -2, 1, -2, 3, -1, 3, -1, 1, 2, -1],
                                                    ([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3], [0, 1, 2, 1, 2, 3, 3, 4, 5, 0, 4])),
                                              shape=(len(test_matrix), len(test_matrix[0])))

        print(self.coboundary_map)

        self.compute_sheaf_laplacian()

    def draw_graph(self, edge_colors:bool=True, edge_labels:bool=True):
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
        #pos = nx.kamada_kawai_layout(self.graph)
        # circular (like a better spring)
        #pos = nx.forceatlas2_layout(self.graph)
        # good variety of distances but can look crowded
        pos = nx.fruchterman_reingold_layout(self.graph)

        if edge_colors:
            color_map = colormaps["Blues"]
            weights = [self.graph[u][v]["weight"] for u, v in self.graph.edges()]
            # scale color map to maximum weight in graph
            #min_weight = min(weights)
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


if __name__ == "__main__":
    pc = ProviderConnections(primary_specialty_weight=2, restriction_weights=[.8, 1, 1])
    #pc.add_test_data()
    graph = pc.build_graph(rows=1_000)
    spec_names = pc.compute_all_give_rankings()
    ev = EvaluationMethods(graph)
    eval_specs_pr = ev.page_rank_all_specialties(spec_names)
    eval_specs_rl = ev.regular_laplacian(graph)
    eval_specs_sir = ev.SIR(graph, spec_names)
    eval_compare = CompareData()
    eval_compare.compare(ev.graph, eval_specs_pr, title="Page Ranking", show_lists=False)
    eval_compare.compare(ev.graph, eval_specs_rl, title="Regular Laplacian", show_lists=False)
    eval_compare.compare(ev.graph, eval_specs_sir, title="Susceptible-Infected-Recovered", show_lists=False)
    #pc.sheaf_laplacian()
    #pc.draw_graph(edge_colors=True, edge_labels=True)