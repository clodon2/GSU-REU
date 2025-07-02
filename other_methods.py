from networkx import Graph
import networkx as nx
import networkit as nk
import networkit.centrality
import numpy as np
import scipy as sp
from multiprocessing import Pool, shared_memory
from scipy.sparse import csr_matrix
from tqdm import tqdm
from time import time
import sys


_shared_laplacian = None
def init_worker(shared_data):
    """
    initialize workers for laplacian pool
    :param shared_data: shared memory data
    :return:
    """
    global _shared_laplacian
    _shared_laplacian = {
        "data": shared_memory.SharedMemory(name=shared_data["data_name"]),
        "indices": shared_memory.SharedMemory(name=shared_data["indices_name"]),
        "indptr": shared_memory.SharedMemory(name=shared_data["indptr_name"]),
        "shape": shared_data["shape"],
        "dtype": np.dtype(shared_data["dtype"]),
        "indices_dtype": np.dtype(shared_data["indices_dtype"]),
        "indptr_dtype": np.dtype(shared_data["indptr_dtype"]),
        "data_shape": shared_data["data_shape"],
        "indices_shape": shared_data["indices_shape"],
        "indptr_shape": shared_data["indptr_shape"]
    }

def create_shared_csr(matrix: csr_matrix):
    """
    convert a csr matrix to shared memory storage
    :param matrix: networkx csr matrix
    :return:
    """
    shm_data = shared_memory.SharedMemory(create=True, size=matrix.data.nbytes)
    shm_indices = shared_memory.SharedMemory(create=True, size=matrix.indices.nbytes)
    shm_indptr = shared_memory.SharedMemory(create=True, size=matrix.indptr.nbytes)

    # Copy data into shared memory buffers
    np.frombuffer(shm_data.buf, dtype=matrix.data.dtype)[:] = matrix.data
    np.frombuffer(shm_indices.buf, dtype=matrix.indices.dtype)[:] = matrix.indices
    np.frombuffer(shm_indptr.buf, dtype=matrix.indptr.dtype)[:] = matrix.indptr

    return {
        "data": shm_data,
        "indices": shm_indices,
        "indptr": shm_indptr,
        "shape": matrix.shape,
        "dtype": matrix.data.dtype,
        "indices_dtype": matrix.indices.dtype,
        "indptr_dtype": matrix.indptr.dtype,
        "data_shape": matrix.data.shape,
        "indices_shape": matrix.indices.shape,
        "indptr_shape": matrix.indptr.shape
    }

def load_shared_csr(meta):
    """
    load a csr matrix from shared memory (reconstruct)
    :param meta: shared memory dict
    :return: csr matrix
    """
    data = np.ndarray(shape=meta["data_shape"], dtype=meta["dtype"], buffer=meta["data"].buf)
    indices = np.ndarray(shape=meta["indices_shape"], dtype=meta["indices_dtype"], buffer=meta["indices"].buf)
    indptr = np.ndarray(shape=meta["indptr_shape"], dtype=meta["indptr_dtype"], buffer=meta["indptr"].buf)
    return csr_matrix((data, indices, indptr), shape=meta["shape"])

def laplacian_centrality_helper(args):
    """
    multiprocessing function for each task to find laplacian centrality of a node
    :param args: see pool in main method, node_index, node, full_laplacian_energy, degree, neighbors
    :return: node, centrality
    """
    global _shared_laplacian
    laplacian = load_shared_csr(_shared_laplacian)
    node_index, node, full_laplacian_energy, degree, neighbors = args
    removed_row = laplacian._getrow(node_index).toarray().flatten()
    diagonal = removed_row[node_index]
    row_energy = np.sum(removed_row ** 2)
    E_sub = full_laplacian_energy - (2 * row_energy - (diagonal ** 2))
    correction = 0
    for neighbor in neighbors:
        neighbor_diagonal = laplacian[neighbor, neighbor]
        correction += 1 - (2 * neighbor_diagonal)

    final_energy = E_sub + correction
    centrality = (full_laplacian_energy - final_energy) / full_laplacian_energy
    return node, centrality


class EvaluationMethods:
    def __init__(self, graph):
        self.graph = graph
        self.laplacian = None

    def regular_laplacian(self, graph:Graph):
        """
        get the centralities for all nodes using graph laplacian
        :param graph: graph
        :return: dict[specialty] = [scores]
        """
        start = time()
        ranking = {}
        centralities = self.laplacian_centrality_multiprocessing(graph)
        for node in centralities:
            for specialty in graph.nodes[node]["specialties"]:
                if specialty in ranking:
                    ranking[specialty].append((node, centralities[node]))
                else:
                    ranking[specialty] = [(node, centralities[node])]
        print(f"regular laplacian centralities found in {time() - start}")
        return ranking

    def degree(self, graph:Graph):
        """
        get the centralities for all nodes using degree centrality
        WARNING: lots of identical scorings
        :param graph: networkx graph
        :return: dict[specialty] = [scores]
        """
        start = time()
        ranking = {}
        centralities = nx.degree_centrality(graph)
        for node in centralities:
            for specialty in graph.nodes[node]["specialties"]:
                if specialty in ranking:
                    ranking[specialty].append((node, centralities[node]))
                else:
                    ranking[specialty] = [(node, centralities[node])]
        print(f"regular laplacian centralities found in {time() - start}")
        return ranking

    def katz(self, graph:Graph):
        """
        get the centralities for all nodes using katz centrality
        :param graph: networkx graph
        :return: dict[specialty] = [scores]
        """
        start = time()
        ranking = {}
        nk_graph = nk.nxadapter.nx2nk(graph)
        centrality = nk.centrality.KatzCentrality(nk_graph).run()
        scores = centrality.scores()
        node_list = list(graph.nodes())  # index i corresponds to scores[i]
        centrality_dict = {node_list[i]: scores[i] for i in range(len(scores))}
        for node in centrality_dict:
            for specialty in graph.nodes[node]["specialties"]:
                if specialty in ranking:
                    ranking[specialty].append((node, centrality_dict[node]))
                else:
                    ranking[specialty] = [(node, centrality_dict[node])]

        print(f"katz centralities found in {time() - start}")
        return ranking

    def closeness(self, graph:Graph):
        """
        get the centralities for all nodes using closeness centrality
        WARNING: computationally expensive, takes ~5x regular laplacian time
        :param graph: networkx graph
        :return: dict[specialty] = [scores]
        """
        start = time()
        ranking = {}
        nk_graph = nk.nxadapter.nx2nk(graph)
        centrality = nk.centrality.Closeness(nk_graph, True, False).run()
        scores = centrality.scores()
        node_list = list(graph.nodes())  # index i corresponds to scores[i]
        closeness_dict = {node_list[i]: scores[i] for i in range(len(scores))}
        for node in closeness_dict:
            for specialty in graph.nodes[node]["specialties"]:
                if specialty in ranking:
                    ranking[specialty].append((node, closeness_dict[node]))
                else:
                    ranking[specialty] = [(node, closeness_dict[node])]
        print(f"closeness centralities found in {time() - start}")
        return ranking

    def betweenness(self, graph:Graph):
        """
        get the centralities for all nodes using betweenness centrality
        WARNING: very computationally expensive, likely greater than 8 hours to finish
        :param graph: networkx graph
        :return: dict[specialty] = [scores]
        """
        start = time()
        ranking = {}
        nk_graph = nk.nxadapter.nx2nk(graph)
        centrality = nk.centrality.ApproxBetweenness(nk_graph).run()
        scores = centrality.scores()
        node_list = list(graph.nodes())  # index i corresponds to scores[i]
        score_dict = {node_list[i]: scores[i] for i in range(len(scores))}
        for node in score_dict:
            for specialty in graph.nodes[node]["specialties"]:
                if specialty in ranking:
                    ranking[specialty].append((node, score_dict[node]))
                else:
                    ranking[specialty] = [(node, score_dict[node])]
        print(f"betweenness centralities found in {time() - start}")
        return ranking

    def page_rank(self, graph:Graph):
        """
        get the pagerank centralities for each node in a subgraph
        :param graph: networkx graph
        :return: dict[specialty] = [scores]
        """
        start = time()
        ranking = {}
        centralities = nx.pagerank(graph)
        for node in centralities:
            for specialty in graph.nodes[node]["specialties"]:
                if specialty in ranking:
                    ranking[specialty].append((node, centralities[node]))
                else:
                    ranking[specialty] = [(node, centralities[node])]
        print(f"page rank centralities found in {time() - start}")
        return ranking

    def laplacian_centrality_multiprocessing(self, subgraph:Graph, weight="weight"):
        """
        calculate laplacian centralities of a graph
        :param subgraph: networkx graph
        :param weight: edge attribute to use as weight in calculation
        :return:
        """
        # divide up total work into groups to avoid pickling errors with large node number
        print("calculating centralities for specialty")
        start = time()
        laplacian = nx.laplacian_matrix(subgraph, weight=weight).tocsr()
        full_laplacian_energy = np.sum(laplacian ** 2)
        nodelist = subgraph.nodes
        node_indices = {node:i for i, node in enumerate(nodelist)}

        shared = create_shared_csr(laplacian)

        # Pass only memory names to pool
        shared_data_for_pool = {
            "data_name": shared["data"].name,
            "indices_name": shared["indices"].name,
            "indptr_name": shared["indptr"].name,
            "shape": shared["shape"],
            "dtype": shared["dtype"].name,
            "indices_dtype": shared["indices_dtype"].name,
            "indptr_dtype": shared["indptr_dtype"].name,
            "data_shape": shared["data_shape"],
            "indices_shape": shared["indices_shape"],
            "indptr_shape": shared["indptr_shape"]
        }

        pool_args = []

        setup_time_start = time()
        # setup arguments for helper function calls
        for node, node_index in node_indices.items():
            degree = subgraph.degree(node)
            neighbors = [node_indices[n] for n in subgraph.neighbors(node)]
            pool_args.append((node_index, node, full_laplacian_energy, degree, neighbors))
        print(f"setup finished in {time() - setup_time_start}")

        results = []
        with Pool(processes=15, initializer=init_worker, initargs=(shared_data_for_pool,)) as pool:
            results_iter = pool.imap_unordered(laplacian_centrality_helper, pool_args, chunksize=1)
            for result in tqdm(results_iter, total=len(pool_args), file=sys.stdout):
                results.append(result)

        shared["data"].close()
        shared["data"].unlink()
        shared["indices"].close()
        shared["indices"].unlink()
        shared["indptr"].close()
        shared["indptr"].unlink()

        laplacian_centralities = {}
        print("processing results...")
        for result in results:
            node = result[0]
            lapl_cent = result[1]
            laplacian_centralities[node] = float(lapl_cent)

        print(f"regular laplacian centralities done in {time() - start}")
        return laplacian_centralities
