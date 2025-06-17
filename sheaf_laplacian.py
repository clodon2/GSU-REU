import networkx as nx
import numpy as np
import scipy as sp
import time
from multiprocessing import Pool
from math import ceil


class SheafLaplacian:
    def __init__(self, graph:nx.Graph, coboundary_columns:int, restriction_weights:list=[1, 1, 1], primary_specialty_weight:float=2):
        """
        create provider graph manager
        :param restriction_weights: [pair count weight, beneficiary count weight, same day count weight]
        """
        self.restriction_weights = restriction_weights
        self.primary_specialty_weight = primary_specialty_weight
        self.graph = graph
        self.coboundary_columns = coboundary_columns
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

                # check primery weight is correct
                """
                if self.graph.nodes[provider]["primary"]:
                    specialty_primary_index = self.graph.nodes[provider]["specialties"].index(self.graph.nodes[provider]["primary"])
                    if self.graph.nodes[provider]["sheaf_vector"][specialty_primary_index] != self.primary_specialty_weight:
                        self.graph.nodes[provider]["sheaf_vector"][specialty_primary_index] = self.primary_specialty_weight
                """
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

    def compute_centralities_multiprocessing_helper(self, sheaf_laplacian_energy, node, all_col):
        """
        each worker performs this function, finds all specialty centralities for a single node
        :param sheaf_laplacian_energy: energy of the sheaf laplacian without removing anything
        :param node: node to get specialty energies for
        :param i: node number, not needed
        :return: node id, centralities in order same as node specialties
        """
        # should be csr
        coboundary_csr = self.coboundary_map
        node_centralities = []
        indices = self.graph.nodes[node]["indices"]
        # for each specialty, get the centrality score
        for specialty_index in indices:
            include_cols = np.setdiff1d(all_col, [specialty_index])
            sub_coboundary = coboundary_csr[:, include_cols]

            # sheaf laplacian of coboundary w/ removed
            sheaf_laplacian = sub_coboundary.transpose().dot(sub_coboundary)
            spec_energy = np.sum(sheaf_laplacian.data ** 2)
            # centrality (impact) for each specialty of each node
            centrality = (sheaf_laplacian_energy - spec_energy) / sheaf_laplacian_energy
            node_centralities.append(centrality)

        return node, node_centralities

    def compute_centralities_multiprocessing(self):
        """
        calculate the centralities for every specialty of every node by removing the column of the specialty
        :return:
        """
        print("computing sheaf laplacian energy...")
        start = time.time()
        sheaf_laplacian_energy = np.sum(self.sheaf_laplacian.data ** 2)
        print(f"sheaf laplacian energy", sheaf_laplacian_energy)
        # convert coboundary map for column removal later
        self.coboundary_map = self.coboundary_map.tocsr()
        all_columns = np.arange(self.coboundary_map.shape[1])
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
                pool_args.append((sheaf_laplacian_energy, node, all_columns))
            print("computing sheaf laplacian centralities")
            start = time.time()
            with Pool(processes=10) as pool:
                results.extend(pool.starmap(self.compute_centralities_multiprocessing_helper, pool_args, chunksize=500))

        print(f"found in {time.time() - start}")
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
        return self.rankings

    def compute_centralities_multiprocessing_remove_whole_helper(self, sheaf_laplacian_energy, node, all_col, all_row):
        """
        each worker performs this function, finds all specialty centralities for a single node
        :param sheaf_laplacian_energy: energy of the sheaf laplacian without removing anything
        :param node: node to get specialty energies for
        :param i: node number, not needed
        :return: node id, centralities in order same as node specialties
        """
        # should be csr
        coboundary_csr = self.coboundary_map
        indices = self.graph.nodes[node]["indices"]
        edge_indices = self.graph.nodes[node]["edge_indices"]
        # for each node, get the centrality score
        include_cols = np.setdiff1d(all_col, indices)
        include_rows = np.setdiff1d(all_row, edge_indices)
        sub_coboundary = coboundary_csr[include_rows, :][:, include_cols]

        # sheaf laplacian of coboundary w/ removed
        sheaf_laplacian = sub_coboundary.transpose().dot(sub_coboundary)
        node_energy = np.sum(sheaf_laplacian.data ** 2)
        # centrality (impact) for each specialty of each node
        centrality = (sheaf_laplacian_energy - node_energy) / sheaf_laplacian_energy

        return node, centrality

    def compute_centralities_multiprocessing_remove_whole(self):
        """
        calculate the centrality of every node by removing the columns and edges
        :return:
        """
        print("computing sheaf laplacian energy...")
        start = time.time()
        sheaf_laplacian_energy = np.sum(self.sheaf_laplacian.data ** 2)
        print(f"sheaf laplacian energy", sheaf_laplacian_energy)
        # convert coboundary map for column removal later
        self.coboundary_map = self.coboundary_map.tocsr()
        all_columns = np.arange(self.coboundary_map.shape[1])
        all_rows = np.arange(self.coboundary_map.shape[0])
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
                pool_args.append((sheaf_laplacian_energy, node, all_columns, all_rows))
            print("computing sheaf laplacian centralities", groups.index(group))
            start = time.time()
            with Pool(processes=10) as pool:
                results.extend(pool.starmap(self.compute_centralities_multiprocessing_remove_whole_helper, pool_args,
                                            chunksize=500))

        print(f"found in {time.time() - start}")

        end = time.time()
        print(f"energies found in {end - start}")
        return results

    def compute_centralities_multiprocessing_faster(self, batch_size=10000):
        """Parallel computation of centrality per specialty component."""
        m, n = self.coboundary_map.shape
        all_cols = np.arange(n)
        E = np.sum(self.sheaf_laplacian.data ** 2)

        # Step 1: Build tasks
        tasks = []
        for node in self.graph.nodes():
            cols_for_node = self.graph.nodes[node]["indices"]
            specialties = self.graph.nodes[node]["specialties"]
            for local_idx, global_col in enumerate(cols_for_node):
                specialty = specialties[local_idx]
                tasks.append((node, global_col, specialty, all_cols, E))

        print(f"Total component tasks: {len(tasks)}")

        # Step 2: Parallel process
        specialty_to_centrality = {}
        num_workers = 4

        for i in range(0, len(tasks), batch_size):
            print("batch", i)
            batch = tasks[i:i + batch_size]

            with Pool(processes=num_workers, initializer=init_worker, initargs=(self.coboundary_map,)) as pool:
                results = pool.imap(compute_component_centrality, batch, chunksize=500)

            for specialty, (node, score) in results:
                if specialty not in specialty_to_centrality:
                    specialty_to_centrality[specialty] = []
                specialty_to_centrality[specialty].append((node, score))

        # Step 3: Sort by descending centrality
        for specialty in specialty_to_centrality:
            specialty_to_centrality[specialty].sort(key=lambda x: x[1], reverse=True)

        self.rankings = specialty_to_centrality
        return self.rankings

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


_global_coboundary_csr = None
def init_worker(coboundary_csr):
    global _global_coboundary_csr
    _global_coboundary_csr = coboundary_csr
    
def compute_component_centrality(task):
    node, global_col, specialty, all_cols, E = task
    print(node, "running")
    delta = _global_coboundary_csr
    try:
        keep_cols = np.setdiff1d(all_cols, [global_col])
        new_delta = delta[:, keep_cols]
        new_L = new_delta.transpose().dot(new_delta)
        new_E = np.sum(new_L.data ** 2)
        centrality_value = (E - new_E) / E
        return (specialty, (node, centrality_value))
    except Exception as e:
        return (specialty, (node, 0.0))
