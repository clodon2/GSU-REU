import networkx as nx
import numpy as np
import scipy as sp
import time
from multiprocessing import Pool, shared_memory
from math import ceil
from tqdm import tqdm
import sys


_shared_coboundary = None
def init_worker(shared_data):
    global _shared_coboundary
    _shared_coboundary = {
        "data": shared_memory.SharedMemory(name=shared_data["data_name"]),
        "indices": shared_memory.SharedMemory(name=shared_data["indices_name"]),
        "indptr": shared_memory.SharedMemory(name=shared_data["indptr_name"]),
        "shape": shared_data["shape"],
        "dtype": shared_data["dtype"],
        "data_shape": shared_data["data_shape"],
        "indices_shape": shared_data["indices_shape"],
        "indptr_shape": shared_data["indptr_shape"]
    }

def create_shared_csc(matrix: sp.sparse.csc_matrix):
    """convert csc matrix to shared memory for less ram usage and quicker lookup"""
    shm_data = shared_memory.SharedMemory(create=True, size=matrix.data.nbytes)
    shm_indices = shared_memory.SharedMemory(create=True, size=matrix.indices.nbytes)
    shm_indptr = shared_memory.SharedMemory(create=True, size=matrix.indptr.nbytes)

    # copy data
    np.frombuffer(shm_data.buf, dtype=matrix.data.dtype)[:] = matrix.data
    np.frombuffer(shm_indices.buf, dtype=matrix.indices.dtype)[:] = matrix.indices
    np.frombuffer(shm_indptr.buf, dtype=matrix.indptr.dtype)[:] = matrix.indptr

    return {
        "data": shm_data,
        "indices": shm_indices,
        "indptr": shm_indptr,
        "shape": matrix.shape,
        "dtype": matrix.data.dtype,
        "data_shape": matrix.data.shape,
        "indices_shape": matrix.indices.shape,
        "indptr_shape": matrix.indptr.shape
    }

def load_shared_csc(meta):
    """reconstruct csc from shared"""
    data = np.ndarray(
        shape=meta["data_shape"],
        dtype=meta["dtype"],
        buffer=meta["data"].buf
    )
    indices = np.ndarray(
        shape=meta["indices_shape"],
        dtype=np.int32,
        buffer=meta["indices"].buf
    )
    indptr = np.ndarray(
        shape=meta["indptr_shape"],
        dtype=np.int32,
        buffer=meta["indptr"].buf
    )
    return sp.sparse.csc_matrix((data, indices, indptr), shape=meta["shape"])

def compute_centralities_multiprocessing_helper(args):
    """
    each worker performs this function, finds all specialty centralities for a single node
    :param args: contains entire sheaf laplacian energy, all column array and node data tuple in order
    :return: node id, centralities in order same as node specialties
    """
    # should be csc
    sheaf_laplacian_energy, node_data = args
    global _shared_coboundary
    coboundary_map = load_shared_csc(_shared_coboundary)
    node, index, specialty = node_data
    # for each specialty, get the centrality score
    # cut out specialty column with mask (efficient vs hstack or setdiff1d)
    keep = np.ones(coboundary_map.shape[1], dtype=bool)
    keep[index] = False
    sub_coboundary = coboundary_map[:, keep]
    """
    left = coboundary_map[:, :index]
    right = coboundary_map[:, index + 1:]
    sub_coboundary = sp.sparse.hstack([left, right], format='csc')
    """

    # sheaf laplacian of coboundary w/ removed
    sheaf_laplacian = sub_coboundary.transpose().dot(sub_coboundary)
    spec_energy = np.sum(sheaf_laplacian.data ** 2)
    # centrality (impact) for each specialty of each node
    centrality = (sheaf_laplacian_energy - spec_energy) / sheaf_laplacian_energy

    return node, centrality, specialty


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

    def compute_coboundary_map(self, include_edge_indices=False):
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
            for num, provider in enumerate(edge):
                if include_edge_indices:
                    self.graph.nodes[provider]["edge_indices"].append(i)
                # get restriction maps based on provider edge percentages
                pair_percentage = self.restriction_weights[0] * (edge_pairs)
                bene_total = self.restriction_weights[1] * (edge_benes)
                same_day_total = self.restriction_weights[2] * (edge_same_days)
                restriction = np.array([pair_percentage, bene_total, same_day_total])

                # check primary weight is correct
                if self.graph.nodes[provider]["primary"]:
                    specialty_primary_index = self.graph.nodes[provider]["specialties"].index(self.graph.nodes[provider]["primary"])
                    if self.graph.nodes[provider]["sheaf_vector"][specialty_primary_index] != self.primary_specialty_weight:
                        self.graph.nodes[provider]["sheaf_vector"][specialty_primary_index] = self.primary_specialty_weight
                # add info to array for sparse matrix conversion
                restriction_map = self.graph.nodes[provider]["sheaf_vector"] * np.sum(restriction) * self.graph.degree(provider) / 50
                # one restriction map is negative
                if num == 1:
                    restriction_map *= -1
                nonzero_restrictions.extend(restriction_map.tolist())
                nzr_column_indices.extend(self.graph.nodes[provider]["indices"])
                for input_num in range(len(self.graph.nodes[provider]["indices"])):
                    nzr_row_indices.append(i)

        coboundary_map = sp.sparse.csr_matrix((nonzero_restrictions, (nzr_row_indices, nzr_column_indices)),
                                              shape=(len(self.graph.edges), self.coboundary_columns))

        self.coboundary_map = coboundary_map
        end = time.time()
        print(f"coboundary map found in {end - start}")
        print(f"coboundary shape: {self.coboundary_map.shape}")

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

    def compute_centralities_multiprocessing(self, only_top_specialties:list=[]):
        """
        calculate the centralities for every specialty of every node by removing the column of the specialty
        :param only_top_specialties: list of specialties to get node centralities for
        :return: dict of specialty:ranking list of tuples (node, centrality)
        """
        print("computing sheaf laplacian energy...")
        start = time.time()
        self.rankings = {}
        sheaf_laplacian_energy = np.sum(self.sheaf_laplacian.data ** 2)
        print(f"sheaf laplacian energy", sheaf_laplacian_energy)
        # convert coboundary map for column removal later
        self.coboundary_map = self.coboundary_map.tocsc()
        print("generating pool args")

        print("setting up shared memory...")
        shared = create_shared_csc(self.coboundary_map)

        # Pass only memory names to pool
        shared_data_for_pool = {
            "data_name": shared["data"].name,
            "indices_name": shared["indices"].name,
            "indptr_name": shared["indptr"].name,
            "shape": shared["shape"],
            "dtype": str(shared["dtype"]),
            "data_shape": shared["data_shape"],
            "indices_shape": shared["indices_shape"],
            "indptr_shape": shared["indptr_shape"]
        }

        # divide up total work into groups to avoid pickling errors with large node number (deprecated for <500k nodes)
        group_size = 100_000
        divisions = ceil(len(self.graph.nodes) / group_size)
        groups = [list(self.graph.nodes)[i * group_size:(i + 1) * group_size] for i in range(divisions)]

        print(f"setup finished in {time.time() - start}")
        # process groups, centralities
        print("computing sheaf laplacian centralities")
        results = []
        start = time.time()
        for g, group in enumerate(groups):
            pool_args = []
            for node in group:
                for specialty, node_index in zip(self.graph.nodes[node]["specialties"], self.graph.nodes[node]["indices"]):
                    # if only need certain specialty centralities, only add those to process list
                    if only_top_specialties:
                        if specialty in only_top_specialties:
                            pool_args.append((sheaf_laplacian_energy, (node, node_index, specialty)))
                    else:
                        pool_args.append((sheaf_laplacian_energy, (node, node_index, specialty)))
            print(f"processing group {g+1} of {len(groups)} with {len(pool_args)} specialties of nodes")
            with Pool(processes=15, initializer=init_worker, initargs=(shared_data_for_pool, )) as pool:
                # use imap to give iterable to track results with tqdm
                results_iter = (pool.imap_unordered(compute_centralities_multiprocessing_helper, pool_args, chunksize=5))
                for result in tqdm(results_iter, total=len(pool_args), file=sys.stdout):
                    results.append(result)

        print("cleaning shared memory...")
        shared["data"].close()
        shared["data"].unlink()
        shared["indices"].close()
        shared["indices"].unlink()
        shared["indptr"].close()
        shared["indptr"].unlink()

        # add results to ranking dict
        for entry in results:
            node = entry[0]
            centrality = entry[1]
            specialty = entry[2]
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

    def compute_centralities_multiprocessing_remove_whole(self, only_top_specialties=[]):
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
        print("generating pool args")

        # divide up total work into groups to avoid pickling errors with large node number
        group_size = 50_000
        divisions = ceil(len(self.graph.nodes) / group_size)
        groups = [list(self.graph.nodes)[i * group_size:(i + 1) * group_size] for i in range(divisions)]

        # process groups, centralities
        results = []
        start = time.time()
        for group in groups:
            if only_top_specialties:
                for node in group:
                    keep_indices = []
                    # find indexes of specialties to keep
                    for i, specialty in enumerate(self.graph.nodes[node]["specialties"]):
                        if specialty in only_top_specialties:
                            keep_indices.append(i)
                    # modify indices and specialties together for consistency
                    new_specs = []
                    new_spec_names = []
                    for index in keep_indices:
                        new_specs.append(self.graph.nodes[node]["indices"][index])
                        new_spec_names.append(self.graph.nodes[node]["specialties"][index])
                    self.graph.nodes[node]["indices"] = new_specs
                    self.graph.nodes[node]["specialties"] = new_spec_names

            pool_args = [(sheaf_laplacian_energy, node, all_columns, all_rows) for node in group]
            print("computing sheaf laplacian centralities", groups.index(group))
            with Pool(processes=10) as pool:
                results.extend(pool.starmap(self.compute_centralities_multiprocessing_remove_whole_helper, pool_args,
                                            chunksize=500))

        end = time.time()
        print(f"energies found in {end - start}")
        return results

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

    def compute_all_give_rankings(self, only_top_specialties=[]):
        """
        compute everything needed to get rankings and print them
        :return:
        """
        print("computing all for ranking...")
        self.compute_coboundary_map()
        self.compute_sheaf_laplacian()
        self.compute_centralities_multiprocessing(only_top_specialties)
        ranking = self.get_ranking()

        return ranking

    def compute_all_give_rankings_whole_removal(self, only_top_specialties=[]):
        """
        compute everything needed to get rankings and print them
        :return:
        """
        print("computing all for ranking...")
        self.compute_coboundary_map(include_edge_indices=True)
        self.compute_sheaf_laplacian()
        self.compute_centralities_multiprocessing_remove_whole(only_top_specialties)
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
