import csv
import scipy as sp
# need for extract
import numpy as np
from networkx import Graph
from random import shuffle
from math import log2


class CompareData:
    def __init__(self, provider_ranking_file:str="./datasets/pa_scores_2017.csv",
                 taxonomy_info_file:str="./datasets/taxonomy_info.csv",
                 specialty_info_file:str="./datasets/specialty_2018_reformatted.csv"):
        """
        object used to compare computed rankings to the ground truth
        :param provider_ranking_file: filename of ground truth dataset
        :param taxonomy_info_file: filename of taxonomy code > actual name dataset
        :param specialty_info_file: filename of ground truth provider specialties, in reformatted format
        """
        self.provider_ranking_file = provider_ranking_file
        self.taxonomy_info_file = taxonomy_info_file
        self.specialty_info_file = specialty_info_file
        self.provider_ranking = []
        self.provider_specialty_ranking = {}
        self.taxonomy_info = {}

    def import_provider_ranking(self, score_index):
        """
        import ground truth ranking information from dataset file
        :param score_index: column index of score to give to ground truth npis
        :return:
        """
        with open(self.provider_ranking_file, "r") as rank_file:
            rank_file = csv.reader(rank_file)
            next(rank_file)
            providers = {}
            for line in rank_file:
                provider = line[5].strip()
                # possible score indices: quality=24, pi=47, ia=75, cost=85, mips=20
                # if no score, ignore
                if (line[score_index].strip() == ''):
                    continue
                else:
                    score = float(line[score_index].strip())
                # if duplicate scores, take highest
                if provider not in providers:
                    self.provider_ranking.append((int(provider), score))
                    providers[provider] = score
                else:
                    if providers[provider] < score:
                        providers[provider] = score

    def import_taxonomy_info(self):
        """
        import taxonomy name information from dataset
        :return:
        """
        with open(self.taxonomy_info_file, "r") as tax_file:
            tax_file = csv.reader(tax_file)
            next(tax_file)
            for line in tax_file:
                # formatted as taxonomy code to the left of the taxonomy name
                self.taxonomy_info[line[2].strip()] = line[3].strip()

    def add_provider_specialties(self):
        """
        organize ground truth npis by specialty based on specialty info file
        :return:
        """
        # map specialties to npis
        specialty_dict = {}
        with open(self.specialty_info_file, "r") as specialty_info:
            spec_info = csv.reader(specialty_info)
            next(spec_info)
            for row in spec_info:
                npi = int(row[0].strip())
                specialty_dict[npi] = []
                # specialties stored in columns after npi, where last indicates the primary
                for sc in row[1:-1]:
                    if sc:
                        specialty_dict[npi].append(sc)

        # create new dictionary organized by specialty
        for entry in self.provider_ranking:
            npi = int(entry[0])
            if npi in specialty_dict:
                for specialty in specialty_dict[npi]:
                    if specialty in self.provider_specialty_ranking:
                        self.provider_specialty_ranking[specialty].append(entry)
                    else:
                        self.provider_specialty_ranking[specialty] = []
                        self.provider_specialty_ranking[specialty].append(entry)

    def sort_scores(self):
        """
        sort ground truth scores as as a whole and for each specialty node list
        :return:
        """
        self.provider_ranking = sorted(self.provider_ranking, key=lambda item: item[1], reverse=True)
        for specialty in self.provider_specialty_ranking:
            values = self.provider_specialty_ranking[specialty]
            self.provider_specialty_ranking[specialty] = sorted(values, key=lambda item: item[1], reverse=True)

    def setup_evaluate(self, score_index=20):
        """
        setup comparison object for evaluation
        :param score_index: for self.import_provider_ranking, column index of score measure to use
        :return:
        """
        self.import_provider_ranking(score_index)
        self.add_provider_specialties()
        self.import_taxonomy_info()
        self.sort_scores()

    def save_results(self, file:str, row):
        """
        save data to csv file in ./results/ (overwrites)
        :param file: filename to be created
        :param row: row or list of rows to add to file
        :return:
        """
        print(f"saving results to {file}...")
        with open(file, "w", newline='') as save_file:
            write = csv.writer(save_file)
            if type(row) == list:
                write.writerows(row)
            else:
                write.writerow(row)

    def append_results(self, file:str, row):
        """
        save data to csv file ./results/ , doesn't destroy data already in there
        :param file: filename to save to
        :param row: row or list of rows of data to append
        :return:
        """
        print(f"saving results to {file}...")
        with open(file, "a", newline='') as save_file:
            write = csv.writer(save_file)
            if type(row) == list:
                write.writerows(row)
            else:
                write.writerow(row)

    def save_actual_rankings(self):
        """
        save ground truth ranking information to csv
        :return:
        """
        save_info = []
        for specialty in self.provider_specialty_ranking:
            save_info.append([specialty, self.provider_specialty_ranking[specialty]])
        self.save_results("./results/unfilteredActual.csv", save_info)

    def evaluate_hits(self, trimmed_rankings:dict, n=15):
        """
        evaluate the hits@n for a given ranking
        :param trimmed_rankings: trimmed rankings for each top specialty
        :param n: get the hits within the top n scores
        :return: dict of specialties: spec : value_name: values, mean : mean hits
        """
        output = {}
        mean_hits_at_n = 0

        for specialty in trimmed_rankings:

            # get trimmed rankings
            final_computed = self.groupify_same_scores(trimmed_rankings[specialty]["final_computed"])[:n]
            final_ranked = self.groupify_same_scores(trimmed_rankings[specialty]["final_ranked"])[:n]
            final_computed = [{item[0] for item in group} for group in final_computed]
            final_ranked = [{item[0] for item in group} for group in final_ranked]

            hits = 0
            for group_set in final_computed:
                for gt_set in final_ranked:
                    if any(group_set & gt_set):
                        hits += 1
                        break

            # calculate percentage of correct in computed
            hits_at_n = hits / n
            # add to total of percentages for mean calculation later
            mean_hits_at_n += hits_at_n

            output[specialty] = hits_at_n

        # calculate mean hits over all specialties, if trimmed empty set as 0
        try:
            mean_hits_at_n = mean_hits_at_n / len(trimmed_rankings.keys())
        except:
            mean_hits_at_n = 0

        output["mean"] = mean_hits_at_n

        return output

    def evaluate_NDCG(self, trimmed_rankings:dict, n=15):
        """
        get the normalized discounted gain for computed vs ground truth
        :param trimmed_rankings: dictionary of specialty : computed and ground truth : rankings
        :param n: number of top scores to consider
        :return: dict of specialty : ndcg
        """
        output = {}

        for specialty in trimmed_rankings:
            final_computed = trimmed_rankings[specialty]["final_computed"][:n]
            final_ranked = self.groupify_same_scores(trimmed_rankings[specialty]["final_ranked"])[:n]
            # create a placing dictionary for distance comparison
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

            # calculate ndcg w/ formula and relevancy list
            discounted_gain = 0
            for i, distance in enumerate(final_computed_relevancy):
                i += 1
                # add 1 to distance to prevent division by 0
                discounted_gain += ( 1 / (distance + 1) ) * log2(i + 1)

            ideal_discounted_gain = 0
            for i in range(len(final_computed_relevancy)):
                i += 1
                ideal_discounted_gain += log2(i + 1)

            normalized_discounted_gain = discounted_gain / ideal_discounted_gain
            output[specialty] = normalized_discounted_gain

        # calculate mean ndcg over all specialties, if trimmed empty set as 0
        try:
            output["mean"] = sum(specialty_score[1] for specialty_score in output.items()) / len(trimmed_rankings)
        except:
            output["mean"] = 0

        return output

    def evaluate_correlation(self, trimmed_rankings):
        """
        get the correlation between ground truth and computed
        :param trimmed_rankings: dictionary of specialty : computed and ground truth : rankings
        :return: dict of specialty : correlation
        """
        output = {}
        for specialty in trimmed_rankings:
            final_computed = self.groupify_same_scores(trimmed_rankings[specialty]["final_computed"])
            final_ranked = self.groupify_same_scores(trimmed_rankings[specialty]["final_ranked"])

            # create average rank dictionary for ideal ranking and computed (handle ties)
            true_positions = {}
            current_rank = 0
            for group in final_ranked:
                avg_rank = current_rank + (len(group) - 1) / 2
                for item in group:
                    true_positions[item[0]] = avg_rank
                current_rank += len(group)

            computed_positions = {}
            current_rank = 0
            for group in final_computed:
                avg_rank = current_rank + (len(group) - 1) / 2
                for item in group:
                    computed_positions[item[0]] = avg_rank
                current_rank += len(group)

            # match computed items to ranks
            shared_ids = sorted(set(computed_positions) & set(true_positions))

            computed_ranks = [computed_positions[id] for id in shared_ids]
            true_ranks = [true_positions[id] for id in shared_ids]

            output[specialty] = sp.stats.kendalltau(true_ranks, computed_ranks, variant='b').statistic

        return output

    def evaluate_accuracy(self, trimmed_rankings, n=15):
        """
        evaluate the accuracy@n for a given ranking
        :param trimmed_rankings: trimmed rankings for each top specialty
        :param n: get the accuracy within the top n scores
        :return: dict of specialties: spec : value_name: values, mean : mean accuracy
        """
        output = {}
        mean_accuracy = 0

        for specialty in trimmed_rankings:
            # get trimmed rankings
            final_computed = trimmed_rankings[specialty]["final_computed"][:n]
            grouped_ranked = self.groupify_same_scores(trimmed_rankings[specialty]["final_ranked"])
            final_ranked = grouped_ranked[:n]

            # track what npi have already been counted (ensures duplicates not counted multiple times)
            counted = set()
            true_positive = 0
            # check percentage of correct shared (if correct in computed top n, add to total)
            for i in final_computed:
                for j in range(len(final_ranked)):
                    for score in final_ranked[j]:
                        if i[0] == score[0]:
                            if i[0] not in counted:
                                counted.add(i[0])
                                true_positive += 1
                            break

            # false negative/positive are identical, so just get incorrect guesses and multiply by 2 for formula
            incorrect = n - true_positive
            true_negative = len(grouped_ranked) - n - incorrect
            # formula = TP + TN / TP + FP + FN + TN
            accuracy = (true_positive + true_negative) / ((incorrect * 2) + true_negative + true_positive)
            # add to total of percentages for mean calculation later
            mean_accuracy += accuracy

            output[specialty] = accuracy

        # calculate mean hits over all specialties, if trimmed empty set as 0
        try:
            mean_accuracy = mean_accuracy / len(trimmed_rankings.keys())
        except:
            mean_accuracy = 0

        output["mean"] = mean_accuracy

        return output

    def evaluate_RBO(self, trimmed_rankings, n=None, p=.9):
        """
        evaluate results with Rank Biased Overlap
        :param trimmed_rankings:  trimmed rankings for each top specialty
        :param n: depth to go down to
        :param p: 0-1 or None for dynamic, lower = more weight at early depths
        :return: dict of specialties: spec : value_name: values, mean : mean hits
        """
        # adjust p based on n amount (become more lower-weighted)
        if not p:
            p = 1 - (1 / n)
        output = {}
        mean = 0
        total_scores = 0
        for specialty in trimmed_rankings:
            final_computed = trimmed_rankings[specialty]["final_computed"]
            final_ranked = self.groupify_same_scores(trimmed_rankings[specialty]["final_ranked"])
            # get lists/groups of just npi
            final_computed = [item[0] for item in final_computed]
            final_ranked = [[item[0] for item in group] for group in final_ranked]
            # if no n just match for max (likely leads to final score 1 for all)
            if not n:
                n = max(len(final_computed), len(final_ranked))
            S_set, T_set = set(), set()
            overlap_ratios = []
            # get overlap ratio at each depth
            for d in range(1, n + 1):
                # add npis for set comparison (overlap)
                if d <= len(final_computed):
                    S_set.add(final_computed[d - 1])
                # adds group of same scores
                if d <= len(final_ranked):
                    T_set.update(final_ranked[d - 1])
                # overlap percentage
                overlap_ratios.append(len(S_set & T_set) / d)

            weighted_sum = sum((1 - p) * (p ** d) * overlap_ratios[d] for d in range(n))
            # residual allows to max possible score to be 1
            residual = (p ** n) * sum(overlap_ratios) / n
            rbo = weighted_sum + residual
            output[specialty] = rbo
            mean += rbo
            total_scores += 1

        output["mean"] = mean / total_scores
        return output

    def groupify_same_scores(self, rank_list):
        """
        group scores into a list of lists where each sub-list is nodes with the same score
        doesn't work if scores aren't sorted
        :param rank_list: a sorted list of tuples [(node, score)]
        :return: list of lists
        """
        if not rank_list:
            return []
        grouped_rankings = [[rank_list[0]]]
        group_index = 0
        last_score = rank_list[0]
        for score in rank_list[1:]:
            # if score still same, add to current group
            if score[1] == last_score[1]:
                grouped_rankings[group_index].append(score)
            else:
                grouped_rankings.append([])
                group_index += 1
                grouped_rankings[group_index].append(score)
            last_score = score

        return grouped_rankings

    def trim_rankings(self, computed_ranking:dict, n, top_spec_num=5):
        """
        trim rankings to top n specialties
        :param computed_ranking: ranking computed
        :param n: number of elements specialties must have to be included
        :param top_spec_num: number of top specialties to consider, determined by gt frequency
        :return: dict of specialty: dict of label:ranking
        """
        output = {}

        # only get results for top n specialties
        specialty_scores = []
        for specialty in self.provider_specialty_ranking:
            if (specialty in computed_ranking and len(self.provider_specialty_ranking[specialty]) > 2*n and
                    len(computed_ranking[specialty]) > 2*n):
                specialty_scores.append((specialty, len(self.provider_specialty_ranking[specialty])))

        specialty_scores = sorted(specialty_scores, key=lambda item: item[1], reverse=True)
        specialty_scores = specialty_scores[:top_spec_num]
        top_specialties_names = [specialty_info[0] for specialty_info in specialty_scores]

        for specialty in top_specialties_names:
            final_computed = []
            final_ranked = []
            computed_ranking_dict = dict(computed_ranking[specialty])
            # only eval for providers that are shared between computed and rank dataset
            for score in self.provider_specialty_ranking[specialty]:
                if score[0] in computed_ranking_dict:
                    final_computed.append((score[0], computed_ranking_dict[score[0]]))
                    final_ranked.append(score)

            # remove duplicates (shouldn't be any, but do just in case)
            duplicate_score_info = {}
            for entry in final_ranked:
                if entry[0] in duplicate_score_info:
                    duplicate_score_info[entry[0]]["total"] += entry[1]
                    duplicate_score_info[entry[0]]["number"] += 1
                else:
                    duplicate_score_info[entry[0]] = {}
                    duplicate_score_info[entry[0]]["total"] = entry[1]
                    duplicate_score_info[entry[0]]["number"] = 1

            averaged_final = []
            for entry in final_ranked:
                mean = duplicate_score_info[entry[0]]["total"] / duplicate_score_info[entry[0]]["number"]
                averaged_final.append((entry[0], mean))

            final_ranked = list(dict.fromkeys(averaged_final))
            final_computed = list(dict.fromkeys(final_computed))

            # remove biases if method gives a lot of the same score
            shuffle(final_computed)

            # sorts descending based on score, additionally sorts ascending for npi
            final_ranked = sorted(final_ranked, key=lambda item: (-item[1], item[0]))
            final_computed = sorted(final_computed, key=lambda item: (-item[1], item[0]))

            if final_computed and final_ranked:
                output[specialty] = {}
                output[specialty]["final_computed"] = final_computed
                output[specialty]["final_ranked"] = final_ranked

        return output


    def evaluate_all_and_save(self, computed_ranking:dict, title:str="unknonwn",
                              save_unfiltered:bool=True, n:int=10, top_specialties:int=10,
                              hits:bool=True, ndcg:bool=True, accuracy:bool=True, rbo:bool=True,
                              save_type:str="new"):
        """
        evaluate for all evaluation methods (optionally) and save the output
        :param computed_ranking: dictionary of specialty : scores
        :param title: scoring method name, used for saving
        :param save_unfiltered: if true, save the computed_ranking information to a csv before processing
        :param n: depth in rankings to go to, e.g. hits@10
        :param top_specialties: specialties to consider in comparison, based on most available scores in ground truth
        :param hits: evaluate hits at n
        :param ndcg: evaluate ndcg at n
        :param accuracy: evaluate accuracy at n
        :param rbo: evaluate rbo at n
        :param save_type: "new" or "append" or "none" -- how to save data to file
        :return:
        """
        if save_unfiltered:
            print(f"Saving unfiltered results to ./results/results_unfiltered{title.strip()}.csv...")
            with open(f"./results/results_unfiltered{title.strip()}.csv", "w", newline='') as unfiltered:
                write = csv.writer(unfiltered)
                write.writerow(["key", "rankings"])
                for key in computed_ranking:
                    write.writerow([key, computed_ranking[key]])

        # trim, also removes duplicates
        trimmed_rankings_by_specialty = self.trim_rankings(computed_ranking, n, top_specialties)

        # calculate evaluations
        evaluations = []
        if hits:
            hits_at_n = self.evaluate_hits(trimmed_rankings_by_specialty, n)
            evaluations.append((hits_at_n, f"hits@{n}"))
        if ndcg:
            ndcg_at_n = self.evaluate_NDCG(trimmed_rankings_by_specialty, n)
            evaluations.append((ndcg_at_n, f"NDCG@{n}"))
        if accuracy:
            accuracies = self.evaluate_accuracy(trimmed_rankings_by_specialty, n)
            evaluations.append((accuracies, f"Accuracy@{n}"))
        if rbo:
            rbo = self.evaluate_RBO(trimmed_rankings_by_specialty, n=n, p=.95)
            evaluations.append((rbo, f"RBO@{n}"))

        # save evaluations to file
        save_file_name = f"./results/results{title.strip()}.csv"
        save_info = []
        save_header = ["eval", "specialty", "score"]
        save_info.append(save_header)

        eval = evaluations[0]
        for specialty in eval[0]:
            # convert code to actual name, if not found just use code
            try:
                specialty_name = self.taxonomy_info[specialty]
            except:
                specialty_name = specialty
            save_row = [eval[1], specialty_name, eval[0][specialty]]
            for other_eval in evaluations[1:]:
                try:
                    save_row.extend([other_eval[1], specialty_name, other_eval[0][specialty]])
                except:
                    pass
            save_info.append(save_row)

        if save_type.lower().strip() != "none":
            if save_type.lower().strip() == "append":
                self.append_results(save_file_name, save_info)
            else:
                self.save_results(save_file_name, save_info)

    def get_mean_score(self, computed_ranking:dict, n_range=range(10, 60, 10), top_specialties=5):
        """
        get the mean of all ndcg and hits@n scores, "overall" ranking
        :param computed_ranking: dictionary of specialty : scores
        :param n_range: range to evaluate ns at (hits@n, ndcg at top n)
        :param top_specialties: specialties to consider in comparison, based on most available scores in ground truth
        :return: mean of all ndcg and hits scores
        """
        trimmed_rankings_by_specialty = self.trim_rankings(computed_ranking, 100, top_specialties)

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
        """
        get ranking from unfiltered ranking file
        :param file: unfiltered rank file name
        :return: dict of specialty : scores
        """
        # bad storage method so need to change max field length
        csv.field_size_limit((2**31 - 1))
        with open(file, "r") as extract_file:
            extract = csv.reader(extract_file)
            header = next(extract)
            extracted_dict = {}
            for row in extract:
                specialty = row[0]
                # clean non-score info
                if row[1][:10] == "dict_items":
                    row[1] = row[1][11:-1]
                row[1] = row[1].replace("[", " ")
                row[1] = row[1].replace("]", " ")

                rankings = row[1].split("), ")
                cleaned_rankings = []
                # clean each ranking and add to cleaned rankings
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
                    # eval to compute into actual number (needed because of numpy floats)
                    entry[1] = eval(entry[1])
                    entry = tuple(entry)
                    cleaned_rankings.append(entry)
                extracted_dict[specialty] = cleaned_rankings

        return extracted_dict

    def get_top_spec_names(self, n, top_spec_num=10):
        """
        get the most common specialties in the ground truth
        :param n: number of elements needed to be considered
        :param top_spec_num: number of top specialties to return
        :return: list of specialty codes
        """
        # only get results for top n specialties
        specialty_scores = []
        for specialty in self.provider_specialty_ranking:
            if (len(self.provider_specialty_ranking[specialty]) > n):
                specialty_scores.append((specialty, len(self.provider_specialty_ranking[specialty])))

        specialty_scores = sorted(specialty_scores, key=lambda item: item[1], reverse=True)
        specialty_scores = specialty_scores[:top_spec_num]
        top_specialties_names = [specialty_info[0] for specialty_info in specialty_scores]
        return top_specialties_names

    def get_top_specs(self, n, top_spec_num=10):
        """
        get the most common specialties in the ground truth
        :param n: number of elements needed to be considered
        :param top_spec_num: number of top specialties to return
        :return: list of specialty codes
        """
        # only get results for top n specialties
        specialty_scores = []
        for specialty in self.provider_specialty_ranking:
            if (len(self.provider_specialty_ranking[specialty]) > n):
                specialty_scores.append((specialty, len(self.provider_specialty_ranking[specialty])))

        specialty_scores = sorted(specialty_scores, key=lambda item: item[1], reverse=True)
        specialty_scores = specialty_scores[:top_spec_num]
        return specialty_scores