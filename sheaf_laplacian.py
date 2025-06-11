import networkx as nx
import numpy as np
import scipy as sp
import time
from multiprocessing import Pool
from math import ceil


class SheafLaplacian:
    def __init__(self, graph:nx.Graph, coboundary_columns:int, restriction_weights:list=[1, 1, 1]):
        """
        create provider graph manager
        :param restriction_weights: [pair count weight, beneficiary count weight, same day count weight]
        """
        self.restriction_weights = restriction_weights
        self.graph = graph
        self.coboundary_columns = coboundary_columns
        self.original_coboundary = None
        self.coboundary_map = None
        self.sheaf_laplacian = None
        self.rankings = {}

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
        # coboundary_map = sp.sparse.lil_matrix((len(self.graph.edges), self.coboundary_columns), dtype=np.float64)
        for i, edge in enumerate(self.graph.edges):
            # get edge specific values
            edge_attr = self.graph.get_edge_data(edge[0], edge[1])
            edge_pairs = edge_attr["weight"]
            edge_benes = edge_attr["beneficiaries"]
            edge_same_days = edge_attr["same_day"]
            # edge_restrictions = []
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

                # add edges connected to indices for removal later
                self.graph.nodes[provider]["edge_indices"].append(i)

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

    def get_diagonals(self, csr_mx: sp.sparse.csr_matrix) -> tuple:
        main_diagonal = []
        upper_diagonal = []
        # Access the non-zero elements using row, column, and data
        row, col = csr_mx.nonzero()
        for i, j in zip(row, col):
            if i == j:
                main_diagonal.append((i, j, csr_mx[i, j]))  # Element at (i, i)
            elif i < j:
                upper_diagonal.append((i, j, csr_mx[i, j]))  # Element above the diagonal
        return main_diagonal, upper_diagonal

    def compute_sheaf_laplacian_energy(self, coboundary_map: sp.sparse.csr_matrix) -> float:
        return np.sum(coboundary_map.data ** 2)

    def compute_centralities_multiprocessing_helper(self, sheaf_laplacian_energy, node, i):
        print(f"node {i} of {len(self.graph.nodes)}")
        coboundary_map = self.original_coboundary.copy().tolil()
        node_centralities = []
        # for each specialty, get the centrality score
        for specialty, specialty_name in zip(self.graph.nodes[node]["indices"], self.graph.nodes[node]["specialties"]):
            coboundary_map[:, specialty] = 0
            for row in self.graph.nodes[node]["edge_indices"]:
                coboundary_map.rows[row] = []
                coboundary_map.data[row] = []
            print(len(self.coboundary_map.rows[0]), "node")
            coboundary_csr = coboundary_map.tocsr()
            self.sheaf_laplacian = coboundary_csr.transpose().dot(self.original_coboundary)
            spec_energy = self.compute_sheaf_laplacian_energy(self.sheaf_laplacian)
            # centrality (impact) for each specialty of each node
            centrality = (sheaf_laplacian_energy - spec_energy) / sheaf_laplacian_energy

            node_centralities.append(centrality)

        return node, node_centralities

    def compute_centralities_multiprocessing(self):
        print("computing sheaf laplacian energy...")
        start = time.time()
        sheaf_laplacian_energy = self.compute_sheaf_laplacian_energy(self.sheaf_laplacian)
        self.sheaf_laplacian.tocsc()
        print(f"sheaf laplacian energy", sheaf_laplacian_energy)
        print("computing sheaf laplacian centralities")
        self.original_coboundary = self.coboundary_map.copy()
        self.coboundary_map = self.coboundary_map.tolil()
        pool_args = []
        print("generating pool args")

        # divide up total work into groups to avoid pickling errors with large node number
        group_size = 50_000
        divisions = ceil(len(self.graph.nodes) / group_size)
        groups = [list(self.graph.nodes)[i * group_size:(i + 1) * group_size] for i in range(divisions)]

        # process groups, centralities
        results = []
        for group in groups:
            for i, node in enumerate(group):
                pool_args.append((sheaf_laplacian_energy, node, i))
            print("computing sheaf laplacian centralities")
            with Pool(processes=4) as pool:
                results.extend(pool.starmap(self.compute_centralities_multiprocessing_helper, pool_args))

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
        sorted_rankings = {}
        for specialty in self.rankings:
            values = self.rankings[specialty]
            # reorder to see best provider
            sorted_rankings[specialty] = sorted(values.items(), key=lambda item: item[1], reverse=True)
        return sorted_rankings

    def compute_all_give_rankings(self):
        """
        compute everything needed to get rankings and print them
        :return:
        """
        print("computing all for ranking...")
        self.compute_coboundary_map()
        self.compute_sheaf_laplacian()
        self.compute_centralities_multiprocessing()
        ranking = self.get_ranking()

        return ranking

    def add_test_data(self):
        test_edges = [("v1", "v2"), ("v3", "v2"), ("v3", "v4"), ("v1", "v4")]

        test_matrix = [
            [-1, -2, 1, 0, 0, 0],
            [0, -2, 3. - 1, 0, 0],
            [0, 0, 0, 3, -1, 1],
            [2, 0, 0, 0, -1, 0]
        ]

        self.coboundary_map = sp.sparse.csr_matrix(([-1, -2, 1, -2, 3, -1, 3, -1, 1, 2, -1],
                                                    ([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3], [0, 1, 2, 1, 2, 3, 3, 4, 5, 0, 4])),
                                                   shape=(len(test_matrix), len(test_matrix[0])))

        print(self.coboundary_map)

        self.compute_sheaf_laplacian()
