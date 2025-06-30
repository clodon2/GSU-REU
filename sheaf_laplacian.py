import networkx as nx
import numpy as np
import scipy as sp
import time
from multiprocessing import Pool, shared_memory
from math import ceil
from tqdm import tqdm
import sys
import json


def compute_coboundary_map_multiprocessing_helper(args):
    i, edge, edge_attributes, u_vec, u_idx, u_deg, v_vec, v_idx, v_deg, restriction_weights = args
    node_data = {edge[0]: {"sheaf_vector": u_vec,
                           "indices": u_idx,
                           "degree": u_deg},
                 edge[1]: {"sheaf_vector": v_vec,
                           "indices": v_idx,
                           "degree": v_deg}
                 }
    nonzero_restrictions = []
    nzr_row_indices = []
    nzr_column_indices = []

    edge_pairs = edge_attributes["pairs"]
    edge_benes = edge_attributes["beneficiaries"]
    edge_same_days = edge_attributes["same_day"]
    for num, provider in enumerate(edge):
        # get restriction maps based on provider edge percentages
        pair_percentage = restriction_weights[0] * (edge_pairs)
        bene_total = restriction_weights[1] * (edge_benes)
        same_day_total = restriction_weights[2] * (edge_same_days)
        restriction = np.array([pair_percentage, bene_total, same_day_total])

        # add info to array for sparse matrix conversion
        restriction_map = node_data[provider]["sheaf_vector"] * np.sum(restriction) / 20
        # one restriction map is negative
        if num == 1:
            restriction_map *= -1
        nonzero_restrictions.extend(restriction_map.tolist())
        nzr_column_indices.extend(node_data[provider]["indices"])
        nzr_row_indices.extend([i] * len(node_data[provider]["indices"]))

    return nzr_row_indices, nzr_column_indices, nonzero_restrictions


_shared_sheaf_laplacian = None
def init_worker(shared_data):
    global _shared_sheaf_laplacian
    _shared_sheaf_laplacian = {
        "data": shared_memory.SharedMemory(name=shared_data["data_name"]),
        "indices": shared_memory.SharedMemory(name=shared_data["indices_name"]),
        "indptr": shared_memory.SharedMemory(name=shared_data["indptr_name"]),
        "shape": shared_data["shape"],
        "dtype": shared_data["dtype"],
        "data_shape": shared_data["data_shape"],
        "indices_shape": shared_data["indices_shape"],
        "indptr_shape": shared_data["indptr_shape"]
    }

_shared_coboundary_map = None
def init_worker_edge(shared_data):
    global _shared_coboundary_map
    _shared_coboundary_map = {
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

def create_shared_csr(matrix: sp.sparse.csr_matrix):
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

def load_shared_csr(meta):
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
    return sp.sparse.csr_matrix((data, indices, indptr), shape=meta["shape"])

def compute_centralities_multiprocessing_helper(args):
    """
    each worker performs this function, finds all specialty centralities for a single node
    :param args: contains entire sheaf laplacian energy, all column array and node data tuple in order
    :return: node id, centralities in order same as node specialties
    """
    # should be csc
    sheaf_laplacian_energy, node_data = args
    global _shared_sheaf_laplacian
    sheaf_laplacian = load_shared_csc(_shared_sheaf_laplacian)
    node, index, specialty = node_data
    # for each specialty, get the centrality score
    # cut out specialty column
    sheaf_column = sheaf_laplacian[:, index]
    # value at row/column intersection only gets subtracted once
    intersection = sheaf_column[index, 0]

    spec_energy = np.sum(2*(sheaf_column.data ** 2)) - (intersection ** 2)
    # centrality (impact) for each specialty of each node
    centrality = spec_energy / sheaf_laplacian_energy

    return node, centrality, specialty

def compute_centralities_multiprocessing_helper_remove_whole(args):
    """
    each worker performs this function, finds all specialty centralities for a single node
    :param args: contains entire sheaf laplacian energy, all column array and node data tuple in order
    :return: node id, centralities in order same as node specialties
    """
    # should be csc
    sheaf_laplacian_energy, node_data = args
    global _shared_coboundary_map
    coboundary = load_shared_csr(_shared_coboundary_map)
    node, edge_indices, edge_mask, column_indices, column_mask = node_data
    edge_mask[:] = True
    edge_mask[edge_indices] = False
    column_mask[:] = True
    column_mask[column_indices] = False
    removed_coboundary = coboundary[edge_mask][:, column_mask]

    sheaf_lap = removed_coboundary.transpose().dot(removed_coboundary)

    spec_energy = np.sum((sheaf_lap.data ** 2))
    # centrality (impact) for each specialty of each node
    centrality = (sheaf_laplacian_energy - spec_energy) / sheaf_laplacian_energy

    return node, centrality


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

        nodes = self.graph.nodes
        for provider in self.graph.nodes:
            # check primary weight is correct
            primary = nodes[provider]["primary"]
            if primary:
                specialty_primary_index = nodes[provider]["specialties"].index(primary)
                if nodes[provider]["sheaf_vector"][specialty_primary_index] != self.primary_specialty_weight:
                    nodes[provider]["sheaf_vector"][specialty_primary_index] = self.primary_specialty_weight

        for i, edge in enumerate(self.graph.edges):
            # get edge specific values
            edge_attr = self.graph[edge[0]][edge[1]]
            edge_pairs = edge_attr["pairs"]
            edge_benes = edge_attr["beneficiaries"]
            edge_same_days = edge_attr["same_day"]
            # edge_restrictions = []
            for num, provider in enumerate(edge):
                if include_edge_indices:
                    nodes[provider]["edge_indices"].append(i)
                # get restriction maps based on provider edge percentages
                pair_percentage = self.restriction_weights[0] * (edge_pairs)
                bene_total = self.restriction_weights[1] * (edge_benes)
                same_day_total = self.restriction_weights[2] * (edge_same_days)
                restriction = np.array([pair_percentage, bene_total, same_day_total])

                # add info to array for sparse matrix conversion
                restriction_map = nodes[provider]["sheaf_vector"] * np.sum(restriction) * self.graph.degree(provider, weight="pairs") / 50
                # one restriction map is negative
                if num == 1:
                    restriction_map *= -1
                nonzero_restrictions.extend(restriction_map.tolist())
                nzr_column_indices.extend(nodes[provider]["indices"])
                nzr_row_indices.extend([i] * len(nodes[provider]["indices"]))

        coboundary_map = sp.sparse.csc_matrix((nonzero_restrictions, (nzr_row_indices, nzr_column_indices)),
                                              shape=(len(self.graph.edges), self.coboundary_columns))

        self.coboundary_map = coboundary_map
        end = time.time()
        print(f"coboundary map found in {end - start}")
        print(f"coboundary shape: {self.coboundary_map.shape}")

        return coboundary_map

    def compute_coboundary_map_multiprocessing(self, include_edge_indices=False):
        print("computing coboundary map multiprocessing")
        start = time.time()
        nodes = self.graph.nodes
        # update primary weights to match stored in class object
        for provider, provider_data in nodes.items():
            primary = provider_data["primary"]
            if primary:
                specialty_primary_index = provider_data["specialties"].index(primary)
                if provider_data["sheaf_vector"][specialty_primary_index] != self.primary_specialty_weight:
                    provider_data["sheaf_vector"][specialty_primary_index] = self.primary_specialty_weight

        # store edge data to pass
        edges = list(self.graph.edges(data=True))
        edge_data = []
        for i, (u, v, attr) in enumerate(edges):
            edge_data.append((
                i,
                (u, v),
                attr,
                nodes[u]["sheaf_vector"],
                nodes[u]["indices"],
                self.graph.degree(u, weight="pairs"),
                nodes[v]["sheaf_vector"],
                nodes[v]["indices"],
                self.graph.degree(v, weight="pairs"),
                self.restriction_weights
            ))

        results = []
        with Pool(processes=15) as pool:
            results_iter = pool.imap_unordered(compute_coboundary_map_multiprocessing_helper, edge_data, chunksize=20000)
            for result in tqdm(results_iter, total=len(edge_data), file=sys.stdout):
                results.append(result)

        all_rows = []
        all_columns = []
        all_nonzero = []
        for r, c, nz in results:
            all_rows.extend(r)
            all_columns.extend(c)
            all_nonzero.extend(nz)

        coboundary_map = sp.sparse.csc_matrix((all_nonzero, (all_rows, all_columns)),
                                              shape=(len(edges), self.coboundary_columns))
        self.coboundary_map = coboundary_map
        print(f"coboundary map found in {time.time() - start} with shape {coboundary_map.shape}")
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

        self.sheaf_laplacian = sheaf_lap.tocsc()
        end = time.time()

        print(f"sheaf laplacian done in {end - start}")
        print(sheaf_lap.shape, self.coboundary_map.shape)

        return sheaf_lap.tocsc()

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
        print("generating pool args")

        print("setting up shared memory...")
        shared = create_shared_csc(self.sheaf_laplacian)

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
        group_size = 1_000_000
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
            with Pool(processes=12, initializer=init_worker, initargs=(shared_data_for_pool, )) as pool:
                # use imap to give iterable to track results with tqdm
                results_iter = pool.imap_unordered(compute_centralities_multiprocessing_helper, pool_args, chunksize=500)
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
            node = int(entry[0])
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


    def compute_centralities_multiprocessing_remove_whole(self, only_top_specialties:list=[]):
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
        print("generating pool args")

        print("setting up shared memory...")
        csr_coboundary = self.coboundary_map.tocsr()
        shared = create_shared_csr(csr_coboundary)

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
        group_size = 1_000_000
        divisions = ceil(len(self.graph.nodes) / group_size)
        groups = [list(self.graph.nodes)[i * group_size:(i + 1) * group_size] for i in range(divisions)]

        print(f"setup finished in {time.time() - start}")
        # process groups, centralities
        print("computing sheaf laplacian centralities")
        results = []
        start = time.time()
        edge_mask = np.ones(csr_coboundary.shape[0], dtype=bool)
        column_mask = np.ones(csr_coboundary.shape[1], dtype=bool)
        for g, group in enumerate(groups):
            pool_args = []
            for node in group:
                # if only need certain specialty centralities, only add those to process list
                if only_top_specialties:
                    # check if any specialty from node in top specialties
                    if any(s in self.graph.nodes[node]["specialties"] for s in only_top_specialties):
                        edge_indices = self.graph.nodes[node]["edge_indices"]
                        node_indices = self.graph.nodes[node]['indices']
                        pool_args.append((sheaf_laplacian_energy, (node, edge_indices, edge_mask, node_indices, column_mask)))
                else:
                    edge_indices = self.graph.nodes[node]["edge_indices"]
                    node_indices = self.graph.nodes[node]['indices']
                    pool_args.append((sheaf_laplacian_energy, (node, edge_indices, edge_mask, node_indices, column_mask)))
            print(f"processing group {g+1} of {len(groups)} with {len(pool_args)} nodes")
            with Pool(processes=12, initializer=init_worker_edge, initargs=(shared_data_for_pool, )) as pool:
                # use imap to give iterable to track results with tqdm
                results_iter = pool.imap_unordered(compute_centralities_multiprocessing_helper_remove_whole, pool_args, chunksize=500)
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
        results_by_spec = {}
        for node, score in results:
            node = int(node)
            for specialty in self.graph.nodes[node]["specialties"]:
                if specialty in results_by_spec:
                    results_by_spec[specialty][node] = float(score)
                else:
                    results_by_spec[specialty] = {}
                    results_by_spec[specialty][node] = float(score)

        self.rankings = results_by_spec
        end = time.time()
        print(f"energies found in {end - start}")
        with open("removeWholeReserve.json", "w") as f:
            json.dump(self.rankings, f)
        return self.rankings

    def get_ranking(self):
        """
        get rankings of providers based on specialties
        :return:
        """
        sorted_rankings = {}
        try:
            for specialty in self.rankings:
                values = self.rankings[specialty]
                # reorder to see best provider
                sorted_rankings[specialty] = sorted(values.items(), key=lambda item: item[1], reverse=True)
            return sorted_rankings
        except:
            return self.rankings

    def compute_all_give_rankings(self, only_top_specialties=[]):
        """
        compute everything needed to get rankings and print them
        :return:
        """
        print("computing all for ranking...")
        self.compute_coboundary_map_multiprocessing()
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
