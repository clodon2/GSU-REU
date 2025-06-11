from networkx import Graph, pagerank
import networkx as nx
import numpy as np
import scipy as sp
from multiprocessing import Pool
from scipy.sparse import csr_matrix
from math import ceil


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

    def laplacian_centrality_helper(self, i, node, full_energy, normalized):
        # remove row and col i from lap_matrix
        all_but_i = list(np.arange(self.laplacian.shape[0]))
        all_but_i.remove(i)
        A_2 = self.laplacian[all_but_i, :][:, all_but_i]

        # Adjust diagonal for removed row
        new_diag = self.laplacian.diagonal() - abs(self.laplacian[:, i])
        A_2.setdiag(new_diag[all_but_i])

        if len(all_but_i) > 0:  # catches degenerate case of single node
            new_energy = np.sum(A_2 ** 2)
        else:
            new_energy = 0.0

        lapl_cent = full_energy - new_energy
        if normalized:
            lapl_cent = lapl_cent / full_energy

        return node, lapl_cent

    def laplacian_centrality_multiprocessing(self, normalized=True, nodelist=None, weight="weight",
                                             walk_type=None, alpha=0.95):
        r"""FROM nx.algorithms.centrality.laplacian

        Compute the Laplacian centrality for nodes in the graph `G`.

        The Laplacian Centrality of a node ``i`` is measured by the drop in the
        Laplacian Energy after deleting node ``i`` from the graph. The Laplacian Energy
        is the sum of the squared eigenvalues of a graph's Laplacian matrix.

        .. math::

            C_L(u_i,G) = \frac{(\Delta E)_i}{E_L (G)} = \frac{E_L (G)-E_L (G_i)}{E_L (G)}

            E_L (G) = \sum_{i=0}^n \lambda_i^2

        Where $E_L (G)$ is the Laplacian energy of graph `G`,
        E_L (G_i) is the Laplacian energy of graph `G` after deleting node ``i``
        and $\lambda_i$ are the eigenvalues of `G`'s Laplacian matrix.
        This formula shows the normalized value. Without normalization,
        the numerator on the right side is returned.

        Parameters
        ----------
        G : graph
            A networkx graph

        normalized : bool (default = True)
            If True the centrality score is scaled so the sum over all nodes is 1.
            If False the centrality score for each node is the drop in Laplacian
            energy when that node is removed.

        nodelist : list, optional (default = None)
            The rows and columns are ordered according to the nodes in nodelist.
            If nodelist is None, then the ordering is produced by G.nodes().

        weight: string or None, optional (default=`weight`)
            Optional parameter `weight` to compute the Laplacian matrix.
            The edge data key used to compute each value in the matrix.
            If None, then each edge has weight 1.

        walk_type : string or None, optional (default=None)
            Optional parameter `walk_type` used when calling
            :func:`directed_laplacian_matrix <networkx.directed_laplacian_matrix>`.
            One of ``"random"``, ``"lazy"``, or ``"pagerank"``. If ``walk_type=None``
            (the default), then a value is selected according to the properties of `G`:
            - ``walk_type="random"`` if `G` is strongly connected and aperiodic
            - ``walk_type="lazy"`` if `G` is strongly connected but not aperiodic
            - ``walk_type="pagerank"`` for all other cases.

        alpha : real (default = 0.95)
            Optional parameter `alpha` used when calling
            :func:`directed_laplacian_matrix <networkx.directed_laplacian_matrix>`.
            (1 - alpha) is the teleportation probability used with pagerank.

        Returns
        -------
        nodes : dictionary
           Dictionary of nodes with Laplacian centrality as the value.

        Notes
        -----
        The algorithm is implemented based on [1]_ with an extension to directed graphs
        using the ``directed_laplacian_matrix`` function.

        Raises
        ------
        NetworkXPointlessConcept
            If the graph `G` is the null graph.
        ZeroDivisionError
            If the graph `G` has no edges (is empty) and normalization is requested.

        References
        ----------
        .. [1] Qi, X., Fuller, E., Wu, Q., Wu, Y., and Zhang, C.-Q. (2012).
           Laplacian centrality: A new centrality measure for weighted networks.
           Information Sciences, 194:240-253.
           https://math.wvu.edu/~cqzhang/Publication-files/my-paper/INS-2012-Laplacian-W.pdf

        See Also
        --------
        :func:`~networkx.linalg.laplacianmatrix.directed_laplacian_matrix`
        :func:`~networkx.linalg.laplacianmatrix.laplacian_matrix`
        """
        G = self.graph
        if len(G) == 0:
            raise nx.NetworkXPointlessConcept("null graph has no centrality defined")
        if G.size(weight=weight) == 0:
            if normalized:
                raise ZeroDivisionError("graph with no edges has zero full energy")
            return dict.fromkeys(G, 0)

        if nodelist is not None:
            nodeset = set(G.nbunch_iter(nodelist))
            if len(nodeset) != len(nodelist):
                raise nx.NetworkXError("nodelist has duplicate nodes or nodes not in G")
            nodes = nodelist + [n for n in G if n not in nodeset]
        else:
            nodelist = nodes = list(G)

        if G.is_directed():
            lap_matrix = nx.directed_laplacian_matrix(G, nodes, weight, walk_type, alpha)
        else:
            lap_matrix = nx.laplacian_matrix(G, nodes, weight)

        self.laplacian = lap_matrix

        full_energy = lap_matrix.power(2).sum()
        # calculate laplacian centrality
        laplace_centralities_dict = {}

        # divide up total work into groups to avoid pickling errors with large node number
        group_size = 50_000
        divisions = ceil(len(nodelist) / group_size)
        groups = [nodelist[i * group_size:(i + 1) * group_size] for i in range(divisions)]
        results = []
        for group in groups:
            pool_args = []
            for i, node in enumerate(group):
                print(i)
                i = nodelist.index(node)
                print(i)
                pool_args.append((i, node, full_energy, normalized))

            print(f"starting pool...")
            with Pool(processes=4) as pool:
                results.extend(pool.starmap(self.laplacian_centrality_helper, pool_args))

        print("processing results...")
        for result in results:
            node = result[0]
            lapl_cent = result[1]
            laplace_centralities_dict[node] = float(lapl_cent)

        return laplace_centralities_dict

    def regular_laplacian(self):
        ranking = {}
        #centralities = nx.laplacian_centrality(self.graph)
        centralities = self.laplacian_centrality_multiprocessing()
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