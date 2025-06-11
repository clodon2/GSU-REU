from networkx import Graph, pagerank
import networkx as nx
import numpy as np
import scipy as sp


class EvaluationMethods:
    def __init__(self, graph):
        self.graph = graph
        self.laplacian = None

    def subgraph_given_specialty(self, specialty):
        nodes_of_interest = []
        for node in self.graph.nodes():
            if specialty in self.graph.nodes[node]["specialties"]:
                nodes_of_interest.append(node)
        subgraph = self.graph.subgraph(nodes_of_interest)
        return subgraph

    def page_rank(self, subgraph:Graph, alpha=0.85, personalization=None,
                  max_iter=100, tol=1e-06, nstart=None, weight='weight',dangling=None):
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
        rankings = {}
        for specialty in specialties:
            spec_subgraph = self.subgraph_given_specialty(specialty)
            spec_pr = pagerank(spec_subgraph, alpha, personalization, max_iter, tol,nstart, weight, dangling)
            rankings[specialty] = spec_pr.items()

        return rankings

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

    def compute_laplacian_energy(self, laplacian: sp.sparse.csr_matrix) -> float:
        main_diagonal, upper_diagonal = self.get_diagonals(laplacian)
        # Compute d
        diag = 0
        for _, _, val in main_diagonal:
            diag += val ** 2
        # Compute w_upp
        diag_upp = 0
        for _, _, val in upper_diagonal:
            diag_upp += val ** 2
        # Laplacian energy
        energy = diag + 2 * diag_upp
        return energy

    def regular_laplacian(self):
        # Build degree matrix
        node_list = list(self.graph.nodes())
        degrees = [self.graph.degree(n) for n in node_list]
        D = sp.sparse.diags(degrees)
        # Build the Adjency
        A = nx.to_scipy_sparse_array(self.graph, nodelist=node_list)
        # Regular Lapacian marix
        L = D - A

        L = sp.sparse.csr_matrix(L)
        laplacian_energy = self.compute_laplacian_energy(L)
        print(f"reg laplacian energy", laplacian_energy)
        print("computing laplacian centralities")
        ranking = {}
        for i, node in enumerate(self.graph.nodes):
            # Get nonzero row indices and values in node column
            col = i
            start_ptr = L.indptr[col]
            end_ptr = L.indptr[col + 1]
            row_indices = L.indices[start_ptr:end_ptr]
            values = L.data[start_ptr:end_ptr]
            subtract_total = 0
            for value, row in zip(values, row_indices):
                value = value ** 2
                if row != col:
                    value *= 2
                subtract_total += value
            spec_laplacian_energy = laplacian_energy - subtract_total
            # centrality (impact) of each node
            centrality = (laplacian_energy - spec_laplacian_energy) / laplacian_energy

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