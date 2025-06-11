import csv

from fontTools.varLib.interpolatableHelpers import find_parents_and_order
from networkx import Graph
from random import shuffle
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
                # score for new dataset; 24 old: 11/12
                if (line[11].strip() == ''):
                    continue
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

    def save_actual_rankings(self):
        save_info = []
        for specialty in self.provider_specialty_ranking:
            save_info.append([specialty, self.provider_specialty_ranking[specialty]])
        self.save_results("./results/unfilteredActual.csv", save_info)

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
            final_ranked = self.groupify_same_scores(trimmed_rankings[specialty]["final_ranked"])[:n]

            # check percentage of correct shared (if correct in computed top n, add to total)
            for i in final_computed:
                for j in range(len(final_ranked)):
                    for score in final_ranked[j]:
                        if i[0] == score[0]:
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

        for specialty in trimmed_rankings:
            final_computed = trimmed_rankings[specialty]["final_computed"][:n]
            final_ranked = self.groupify_same_scores(trimmed_rankings[specialty]["final_ranked"])[:n]

            # need to create a placing dictionary for distance comparison
            id_to_ideal_positions = {}
            current_pos = 0
            for group in final_ranked:
                group_indices = list(range(current_pos, current_pos + len(group)))
                for item in group:
                    id_to_ideal_positions[item[0]] = group_indices
                current_pos += len(group)

            # relevancy based on distances
            final_computed_relevancy = []
            for i, (item_id, score) in enumerate(final_computed):
                ideal_positions = id_to_ideal_positions.get(item_id, [])
                if not ideal_positions:
                    final_computed_relevancy.append(len(final_computed))
                elif i in ideal_positions:
                    final_computed_relevancy.append(0)
                else:
                    final_computed_relevancy.append(min(abs(i - p) for p in ideal_positions))

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

    def slice_by_unique(self, rank_list, n):
        scores = [score for id, score in rank_list]
        last_score = scores[0]
        us_i = 0
        unique_scores = [0]
        for score in scores:
            if last_score == score:
                unique_scores[us_i] += 1
            else:
                unique_scores.append(1)
                us_i += 1

        actual_n = sum(unique_scores[:n])
        return rank_list[:actual_n]

    def groupify_same_scores(self, rank_list):
        """
        group scores into a list of lists where each sub-list is nodes with the same score
        doesn't work if scores aren't sorted
        :param rank_list: a sorted list of tuples [(node, score)]
        :return: list of lists
        """
        grouped_rankings = [[rank_list[0]]]
        group_index = 0
        last_score = rank_list[0]
        for score in rank_list[1:]:
            if score[1] == last_score[1]:
                grouped_rankings[group_index].append(score)
            else:
                grouped_rankings.append([])
                group_index += 1
                grouped_rankings[group_index].append(score)
            last_score = score

        return grouped_rankings

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
        if save_type.lower().strip() != "none":

            if save_type.lower().strip() == "append":
                self.append_results(save_file_name, save_info)
            else:
                self.save_results(save_file_name, save_info)

    def get_mean_score(self, computed_ranking:dict, n_range=range(10, 50, 10), top_specialties=5):
        trimmed_rankings_by_specialty = self.trim_rankings(computed_ranking, top_specialties)

        # calculate evaluations
        total_mean = 0
        evaluations = []
        for n in n_range:
            hits_at_n = self.evaluate_hits(trimmed_rankings_by_specialty, n)
            evaluations.append((hits_at_n, f"hits@{n}"))
            ndcg_at_n = self.evaluate_NDCG(trimmed_rankings_by_specialty, n)
            evaluations.append((ndcg_at_n, f"NDCG@{n}"))

        for scores in evaluations:
            total_mean += scores[0]["mean"]

        return total_mean / len(evaluations)

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