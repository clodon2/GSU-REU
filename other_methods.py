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

    def regular_laplacian(self):
        ranking = {}
        centralities = nx.laplacian_centrality(self.graph)
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