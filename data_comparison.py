import csv
from networkx import Graph, pagerank, non_edges
import networkx as nx
import numpy as np
import scipy as sp
from random import random, shuffle
from math import log2


class CompareData:
    def __init__(self, provider_ranking_file:str="pa_scores.csv",
                 taxonomy_info_file:str="taxonomy_info.csv"):
        self.provider_ranking_file = provider_ranking_file
        self.taxonomy_info_file = taxonomy_info_file
        self.provider_ranking = []
        self.provider_specialty_ranking = {}
        self.taxonomy_info = {}

    def import_provider_ranking(self):
        with open(self.provider_ranking_file, "r") as rank_file:
            rank_file = csv.reader(rank_file)
            next(rank_file)
            providers = {}
            for line in rank_file:
                provider = line[0].strip()
                # provider for new dataset: 5 old: 0
                # score for new dataset; 5 old: 11/12
                if line[11].strip() == '':
                    score = 0
                else:
                    score = float(line[11].strip())
                # if duplicate, ignore
                if provider not in providers:
                    self.provider_ranking.append((int(provider), score))
                    providers[provider] = score
                else:
                    if providers[provider] < score:
                        providers[provider] = score

    def import_taxonomy_info(self):
        with open(self.taxonomy_info_file, "r") as tax_file:
            tax_file = csv.reader(tax_file)
            next(tax_file)
            for line in tax_file:
                self.taxonomy_info[line[2].strip()] = line[3].strip()

    def add_provider_specialties(self, graph:Graph):
        for entry in self.provider_ranking:
            provider = int(entry[0])
            score = float(entry[1])
            if provider in graph.nodes:
                specialties = graph.nodes[provider]["specialties"]
                for specialty in specialties:
                    if specialty in self.provider_specialty_ranking:
                        self.provider_specialty_ranking[specialty].append(entry)
                    else:
                        self.provider_specialty_ranking[specialty] = []
                        self.provider_specialty_ranking[specialty].append(entry)

    def sort_scores(self):
        self.provider_ranking = sorted(self.provider_ranking, key=lambda item: item[1], reverse=True)
        for specialty in self.provider_specialty_ranking:
            values = self.provider_specialty_ranking[specialty]
            self.provider_specialty_ranking[specialty] = sorted(values, key=lambda item: item[1], reverse=True)

    def setup_evaluate(self, graph:Graph):
        self.import_provider_ranking()
        self.add_provider_specialties(graph)
        self.import_taxonomy_info()
        self.sort_scores()

    def save_results(self, file:str, row):
        print(f"saving results to {file}...")
        with open(file, "w", newline='') as save_file:
            write = csv.writer(save_file)
            if type(row) == list:
                write.writerows(row)
            else:
                write.writerow(row)

    def append_results(self, file:str, row):
        print(f"saving results to {file}...")
        with open(file, "a", newline='') as save_file:
            write = csv.writer(save_file)
            if type(row) == list:
                write.writerows(row)
            else:
                write.writerow(row)

    def evaluate_hits(self, trimmed_rankings:dict, n=15):
        f"""
        evaluate the hits@n for a given ranking
        :param trimmed_rankings: trimmed rankings for each top specialty
        :param n: get the hits within the top n scores
        :return: dict of specialties: spec : value_name: values, mean : mean hits
        """
        output = {}
        mean_hits_at_n = 0

        for specialty in trimmed_rankings:
            # calculate hits@n
            hits_at_n = 0

            # get trimmed rankings
            final_computed = trimmed_rankings[specialty]["final_computed"][:n]
            final_ranked = trimmed_rankings[specialty]["final_ranked"][:n]

            # check percentage of correct shared (if correct in computed top n, add to total)
            for i in final_computed:
                for j in final_ranked:
                    if i[0] == j[0]:
                        hits_at_n += 1
                        # there is something wrong here...should only count once unless duplicate npi
                        # without break, value increases over 1 for hits@ so need to check rest of code

            # check if in exact position
            """
            for i in range(hits_n):
                if final_ranked[i][0] == final_computed[i][0]:
                    hits_at_n += 1
            """

            # calculate percentage of correct in computed
            hits_at_n /= n
            # add to total of percentages for mean calculation later
            mean_hits_at_n += hits_at_n

            output[specialty] = hits_at_n

        # calculate mean hits over all specialties
        mean_hits_at_n = mean_hits_at_n / len(trimmed_rankings.keys())

        output["mean"] = mean_hits_at_n

        return output

    def evaluate_NDCG(self, trimmed_rankings:dict, n=15):
        output = {}
        mean_NDCG = 0

        for specialty in trimmed_rankings:
            final_computed = trimmed_rankings[specialty]["final_computed"][:n]
            final_ranked = trimmed_rankings[specialty]["final_ranked"][:n]

            final_computed_relevancy = []
            # distance calculated by subtracting index
            for i in range(len(final_computed)):
                for j in range(len(final_ranked)):
                    if final_ranked[j][0] == final_computed[i][0]:
                        # append distance
                        final_computed_relevancy.append(abs(j - i))
                        break

            if not final_computed_relevancy:
                output[specialty] = 0
                continue

            discounted_gain = 0
            for i, distance in enumerate(final_computed_relevancy):
                i += 1
                discounted_gain += ( 1 / (distance + 1) ) * log2(i + 1)

            ideal_discounted_gain = 0
            for i in range(len(final_computed_relevancy)):
                i += 1
                ideal_discounted_gain += log2(i + 1)

            normalized_discounted_gain = discounted_gain / ideal_discounted_gain
            output[specialty] = normalized_discounted_gain

        output["mean"] = sum(specialty_score[1] for specialty_score in output.items()) / len(trimmed_rankings)

        return output

    def trim_rankings(self, computed_ranking:dict, n):
        """
        trim rankings to top n specialties
        :param final_ranking: ranking from rank database
        :param computed_ranking: ranking computed
        :param n: number of top specialties to include
        :return: dict of specialty: dict of label:ranking
        """
        output = {}

        # only get results for top 5 specialties
        specialty_scores = []
        for specialty in self.provider_specialty_ranking:
            specialty_scores.append((specialty, len(self.provider_specialty_ranking[specialty])))

        specialty_scores = sorted(specialty_scores, key=lambda item: item[1], reverse=True)
        specialty_scores = specialty_scores[:n]
        top_specialties_names = [specialty_info[0] for specialty_info in specialty_scores]

        for specialty in top_specialties_names:
            output[specialty] = {}
            final_computed = []
            final_ranked = []
            # only eval for providers that are shared between computed and rank dataset
            for score in self.provider_specialty_ranking[specialty]:
                for calc_score in computed_ranking[specialty]:
                    if score[0] == calc_score[0]:
                        final_computed.append(calc_score)
                        final_ranked.append(score)
                        break

            # remove duplicates (move to dict creation probs)
            for score in final_ranked:
                count = final_ranked.count(score)
                for i in range(count - 1):
                    final_ranked.remove(score)

            for score in final_computed:
                count = final_computed.count(score)
                for i in range(count - 1):
                    final_computed.remove(score)

            # remove biases
            shuffle(final_computed)

            final_ranked = sorted(final_ranked, key=lambda item: item[1], reverse=True)
            final_computed = sorted(final_computed, key=lambda item: item[1], reverse=True)

            output[specialty]["final_computed"] = final_computed
            output[specialty]["final_ranked"] = final_ranked

        return output


    def evaluate_all_and_save(self, computed_ranking:dict, title="unknonwn",
                              save_unfiltered=True, hits_n=15, ndcg_n=15, top_specialties=5, save_type="new"):
        if save_unfiltered:
            print(f"Saving unfiltered results to ./results/results_unfiltered{title.strip()}.csv...")
            with open(f"./results/results_unfiltered{title.strip()}.csv", "w", newline='') as unfiltered:
                write = csv.writer(unfiltered)
                write.writerow(["key", "rankings"])
                for key in computed_ranking:
                    write.writerow([key, computed_ranking[key]])

        trimmed_rankings_by_specialty = self.trim_rankings(computed_ranking, top_specialties)

        # calculate evaluations
        evaluations = []
        if hits_n:
            hits_at_n = self.evaluate_hits(trimmed_rankings_by_specialty, hits_n)
            evaluations.append((hits_at_n, f"hits@{hits_n}"))
        if ndcg_n:
            ndcg_at_n = self.evaluate_NDCG(trimmed_rankings_by_specialty, ndcg_n)
            evaluations.append((ndcg_at_n, f"NDCG@{ndcg_n}"))

        # save evaluations to file
        save_file_name = f"./results/results{title.strip()}.csv"
        save_info = []
        save_header = ["eval", "specialty", "score"]
        save_info.append(save_header)

        eval = evaluations[0]
        for specialty in eval[0]:
            try:
                specialty_name = self.taxonomy_info[specialty]
            except:
                specialty_name = specialty
            save_row = [eval[1], specialty_name, eval[0][specialty]]
            for other_eval in evaluations[1:]:
                save_row.extend([other_eval[1], specialty_name, other_eval[0][specialty]])
            save_info.append(save_row)

        """
        for eval in evaluations:
            for specialty in eval[0]:
                try:
                    specialty_name = self.taxonomy_info[specialty]
                except:
                    specialty_name = specialty
                save_row = [eval[1], specialty_name, eval[0][specialty]]
                save_info.append(save_row)
        """

        if save_type.lower().strip() == "append":
            self.append_results(save_file_name, save_info)
        else:
            self.save_results(save_file_name, save_info)

    def extract_ranking(self, file:str):
        with open(file, "r") as extract_file:
            extract = csv.reader(extract_file)
            header = next(extract)
            extracted_dict = {}
            for row in extract:
                specialty = row[0]
                if row[1][:10] == "dict_items":
                    row[1] = row[1][11:-1]
                row[1] = row[1].replace("[", " ")
                row[1] = row[1].replace("]", " ")

                rankings = row[1].split("), ")
                cleaned_rankings = []
                for entry in rankings:
                    entry = entry.strip()
                    if entry[-1] == ")":
                        entry = entry[1:-1]
                    else:
                        entry = entry[1:]
                    if entry[12] == "n":
                        if entry[-1] != ")":
                            entry += ")"
                    entry = entry.split(", ")
                    entry[0] = int(entry[0])
                    entry[1] = eval(entry[1])
                    entry = tuple(entry)
                    cleaned_rankings.append(entry)
                extracted_dict[specialty] = cleaned_rankings

        return extracted_dict

    def compare(self, graph:Graph, computed_ranking:dict, title="unknonwn",
                show_lists=True, hits_n=15, top_specialties=5):
        """
        deprecated due to inflexibility
        :param graph: graph used for calculations
        :param computed_ranking: computed ranking of method, formatted specialty: list of (provider, score)
        :param title: the method type used, prints
        :param show_lists: prints the raw rankings for actual and computed for all specialties (in actual)
        :param hits_n: hits at n to evaluate (top n providers in specialty)
        :param top_specialties: number of specialties to calculate hits at n for, more rankings in actual is prioritized
        :return:
        """
        # save raw computed ranking data
        print(f"Saving unfiltered results to ./results/results_unfiltered{title.strip()}.csv...")
        with open(f"./results/results_unfiltered{title.strip()}.csv", "w", newline='') as unfiltered:
            write = csv.writer(unfiltered)
            write.writerow(["key", "rankings"])
            for key in computed_ranking:
                write.writerow([key, computed_ranking[key]])

        self.import_provider_ranking()
        self.add_provider_specialties(graph)
        self.import_taxonomy_info()
        self.sort_scores()

        # only get results for top 5 specialties
        specialty_scores = []
        for specialty in self.provider_specialty_ranking:
            specialty_scores.append((specialty, len(self.provider_specialty_ranking[specialty])))

        specialty_scores = sorted(specialty_scores, key=lambda item: item[1], reverse=True)
        specialty_scores = specialty_scores[:top_specialties]

        mean_hits_at_n = 0

        # setup file saving
        save_file_name = f"./results/results{title.strip()}.csv"
        save_info = []

        print(f"comparison of {title} to expected")
        for entry in specialty_scores:
            specialty = entry[0]
            # calculate hits@n
            final_computed = []
            final_ranked = []
            hits_at_n = 0
            #
            for score in self.provider_specialty_ranking[specialty]:
                for calc_score in computed_ranking[specialty]:
                    if score[0] == calc_score[0]:
                        final_computed.append(calc_score)
                        final_ranked.append(score)
                        break

            # remove duplicates (move to dict creation probs)
            for score in final_ranked:
                count = final_ranked.count(score)
                for i in range(count - 1):
                    final_ranked.remove(score)

            for score in final_computed:
                count = final_computed.count(score)
                for i in range(count - 1):
                    final_computed.remove(score)

            final_ranked = sorted(final_ranked, key= lambda item: item[1], reverse=True)
            final_computed = sorted(final_computed, key= lambda item: item[1], reverse=True)

            final_computed = final_computed[:hits_n]
            final_ranked = final_ranked[:hits_n]

            # check percentage of correct shared
            for i in final_computed:
                for j in final_ranked:
                    if i[0] == j[0]:
                        hits_at_n += 1


            # check if in exact position
            """
            for i in range(hits_n):
                if final_ranked[i][0] == final_computed[i][0]:
                    hits_at_n += 1
            """

            for score in final_computed:
                print(f"%10d %15f" % (score[0], score[1]), end=" ")

            print()

            for score in final_ranked:
                print(f"%10d %15f" % (score[0], score[1]), end=" ")

            print()

            hits_at_n /= hits_n
            mean_hits_at_n += hits_at_n


            try:
                print(f"Hits at {hits_n} for {self.taxonomy_info[specialty]}:")
                save_row = [hits_n, self.taxonomy_info[specialty]]
            except:
                print(f"Hits at {hits_n} for {specialty}:")
                save_row = [hits_n, specialty]
            print(hits_at_n)
            save_row.append(hits_at_n)
            save_row.append('')
            save_info.append(save_row)
            print()

        mean_hits_at_n = mean_hits_at_n / top_specialties

        save_info.append(['', '', '', mean_hits_at_n])

        print(f"mean hits at {hits_n}: {mean_hits_at_n}")

        if show_lists:
            for specialty in self.provider_specialty_ranking:
                try:
                    print(f"Ranking for {self.taxonomy_info[specialty]}")
                except:
                    f"Ranking for {specialty}"
                print("Computed:")
                print(f"{computed_ranking[specialty]}")
                print("Actual:")
                try:
                    print(f"{self.provider_specialty_ranking[specialty]}")
                except:
                    print("None")
                print()

        self.save_results(save_file_name, save_info)
        specialty_scores = [name[0] for name in specialty_scores]
        return specialty_scores


class EvaluationMethods:
    def __init__(self, graph):
        self.graph = graph
        self.laplacian = None
        self.page_rank_scores = {}

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
        for specialty in specialties:
            spec_subgraph = self.subgraph_given_specialty(specialty)
            spec_pr = pagerank(spec_subgraph, alpha, personalization, max_iter, tol,nstart, weight, dangling)
            self.page_rank_scores[specialty] = spec_pr.items()

        return self.page_rank_scores

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
        self.page_rank_scores = None
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
            # Get nonzero row indices and values in specialty column
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
            # centrality (impact) for each specialty of each node
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