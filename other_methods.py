from networkx import Graph, pagerank
import networkx as nx
import numpy as np
import scipy as sp
from multiprocessing import Pool, shared_memory
from scipy.sparse import csr_matrix
from math import ceil
from tqdm import tqdm
from time import time
import sys


_shared_laplacian = None
def init_worker(shared_data):
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
    """Convert a CSR matrix to shared memory for multiprocessing."""
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
    data = np.ndarray(shape=meta["data_shape"], dtype=meta["dtype"], buffer=meta["data"].buf)
    indices = np.ndarray(shape=meta["indices_shape"], dtype=meta["indices_dtype"], buffer=meta["indices"].buf)
    indptr = np.ndarray(shape=meta["indptr_shape"], dtype=meta["indptr_dtype"], buffer=meta["indptr"].buf)
    return csr_matrix((data, indices, indptr), shape=meta["shape"])

def laplacian_centrality_helper(args):
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

    def subgraph_given_specialty(self, specialty):
        """
        get a subgraph of a graph including nodes with the same specialty
        :param specialty:
        :return:
        """
        nodes_of_interest = []
        for node in self.graph.nodes():
            if specialty in self.graph.nodes[node]["specialties"]:
                nodes_of_interest.append(node)
        subgraph = self.graph.subgraph(nodes_of_interest)
        return subgraph

    def page_rank(self, subgraph:Graph, alpha=0.85, personalization=None,
                  max_iter=100, tol=1e-06, nstart=None, weight='weight',dangling=None):
        """
        get the pagerank centralities for each node in a subgraph
        :param subgraph:
        :param alpha:
        :param personalization:
        :param max_iter:
        :param tol:
        :param nstart:
        :param weight:
        :param dangling:
        :return:
        """
        page_rank_scores = nx.pagerank(subgraph, alpha, personalization,
                                            max_iter, tol, nstart, weight, dangling)
        # Print the results
        """
        print("PageRank Scores(with edge weights):")
        for node, score in self.page_rank_scores.items():
            print(f"Node{node}: {score}")
        """

        return page_rank_scores

    def page_rank_all_specialties(self, specialties:list, alpha=0.85, personalization=None,
                  max_iter=100, tol=1e-06, nstart=None, weight='weight',dangling=None):
        """
        pagerank all nodesof all specialties
        :param specialties:
        :param alpha:
        :param personalization:
        :param max_iter:
        :param tol:
        :param nstart:
        :param weight:
        :param dangling:
        :return:
        """
        rankings = {}
        for specialty in specialties:
            spec_subgraph = self.subgraph_given_specialty(specialty)
            spec_pr = pagerank(spec_subgraph, alpha, personalization, max_iter, tol,nstart, weight, dangling)
            rankings[specialty] = spec_pr.items()

        return rankings

    def laplacian_centrality_multiprocessing(self, subgraph:Graph, weight="weight"):
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
        with Pool(processes=6, initializer=init_worker, initargs=(shared_data_for_pool,)) as pool:
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

        print(f"one regular laplacian specialty centrality set done in {time() - start}")
        return laplacian_centralities

    def regular_laplacian(self, specialties:list):
        """
        get the centralities for all nodes using graph laplacian
        :param specialties: list of specialty names to subgraph and find centralities
        :return: dict[specialty][ = [scores]
        """
        start = time()
        ranking = {}
        for s, specialty in enumerate(specialties):
            print(f"specialty {s}")
            centralities = self.laplacian_centrality_multiprocessing(self.subgraph_given_specialty(specialty))
            ranking[specialty] = list(centralities.items())
        print(f"regular laplacian centralities found in {time() - start}")
        return ranking

    def betweenness(self):
        """
        get betweenness centralities for each node in graph
        :return: dict[specialty][ = [scores]
        """
        ranking = {}
        centralities = nx.betweenness_centrality(self.graph)
        for node in centralities:
            centrality = centralities[node]
            # add to rankings for specialty
            for specialty_name in self.graph.nodes[node]["specialties"]:
                if specialty_name in ranking:
                    ranking[specialty_name][node] = centrality
                else:
                    ranking[specialty_name] = {}
                    ranking[specialty_name][node] = centrality

        for specialty in ranking:
            ranking[specialty] = list(ranking[specialty].items())

        return ranking

    def degrees(self, specialties:list):
        ranking = {}
        for specialty in specialties:
            centralities = nx.degree_centrality(self.subgraph_given_specialty(specialty))
            ranking[specialty] = list(centralities.items())

        return ranking

    def closeness(self):
        """
        get closeness centralities for each node in graph
        :return: dict[specialty][ = [scores]
        """
        ranking = {}
        centralities = nx.closeness_centrality(self.graph)
        for node in centralities:
            centrality = centralities[node]
            # add to rankings for specialty
            for specialty_name in self.graph.nodes[node]["specialties"]:
                if specialty_name in ranking:
                    ranking[specialty_name][node] = centrality
                else:
                    ranking[specialty_name] = {}
                    ranking[specialty_name][node] = centrality

        for specialty in ranking:
            ranking[specialty] = list(ranking[specialty].items())

        return ranking

    def load_centrality(self):
        """
        get load centralities for each node in graph
        :return: dict[specialty][ = [scores]
        """
        ranking = {}
        centralities = nx.load_centrality(self.graph)
        for node in centralities:
            centrality = centralities[node]
            # add to rankings for specialty
            for specialty_name in self.graph.nodes[node]["specialties"]:
                if specialty_name in ranking:
                    ranking[specialty_name][node] = centrality
                else:
                    ranking[specialty_name] = {}
                    ranking[specialty_name][node] = centrality

        for specialty in ranking:
            ranking[specialty] = list(ranking[specialty].items())

        return ranking

    def SIR_math(self, specialties:list):
        for specialty in specialties:
            total_population = [n for n, attr in self.graph.nodes(data=True) if attr.get('specialty') == specialty]
            # for all nodes in the specialty, calculate sir value
            # number of infected = S/N * p * k * D
            # S=susceptible N=total p=infectionChance k=contacts D=daysInfected I=infected y=recoveryRate=1/D
            # ds/dt=-(p*k*S*I)/N  di/dt=((p*k*S*I)/N)-yI  dr/dt=yI
            # ds/dt=susceptibleChange di/dt=infectedChange dr/dt=recoveredChange
            for node in total_population:
                pass

    def SIR_vectors(self, specialties:list, iterations=50):
        ranking = {}
        for specialty in specialties:
            specialty_nodes = [n for n, attr in self.graph.nodes(data=True) if specialty in attr.get('specialties')]

            for node_number, node in enumerate(specialty_nodes):
                print(f"{node_number} of {len(specialty_nodes)}")
                node_id = node
                population = nx.node_connected_component(self.graph, node_id)

                # population subgraph
                population_graph = self.graph.subgraph(population).copy()

                adjacency = nx.to_scipy_sparse_array(population_graph)

                # Get the nodes in population_graph
                population_nodes = list(population_graph.nodes)
                node_to_idx = {node: idx for idx, node in enumerate(population_nodes)}
                N = len(population_nodes)

                # Initialize states array for this subgraph
                states = np.zeros(N, dtype=np.int8)
                states[node_to_idx[node_id]] = 1
                infection_history = [states.copy()]

                infection_probability_mx = sp.sparse.dok_matrix((N, N))

                for u, v, data in population_graph.edges(data=True):
                    i = node_to_idx[u]
                    j = node_to_idx[v]
                    infection_prob = data.get("weight", 0.2)
                    infection_probability_mx[i, j] = infection_prob
                    infection_probability_mx[j, i] = infection_prob

                for time in range(iterations):
                    infected = (states == 1).astype(np.float32)

                    infected_prob = adjacency @ infected

                    susceptible = (states == 0)

                    rand_vals = np.random.rand(N)

                    infected_neighbors = (rand_vals < infected_prob) & susceptible

                    states[infected_neighbors] = 1

                centrality = len((states == 1).astype(np.float32))
                # add to rankings for specialty
                for specialty_name in self.graph.nodes[node_id]["specialties"]:
                    if specialty_name in ranking:
                        ranking[specialty_name][node] = centrality
                    else:
                        ranking[specialty_name] = {}
                        ranking[specialty_name][node] = centrality

        for specialty in ranking:
            ranking[specialty] = list(ranking[specialty].items())

        return ranking