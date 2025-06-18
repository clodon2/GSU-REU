import csv

from scipy.stats import spearmanr
from networkx import Graph
from random import shuffle
from math import log2

class CompareDataNoSpecialty:
    def __init__(self, provider_ranking_file:str="pa_scores.csv"):
        """
        object used to compare computed rankings to the ground truth
        :param provider_ranking_file: filename of ground truth dataset
        """
        self.provider_ranking_file = provider_ranking_file
        self.provider_ranking = []
        self.taxonomy_info = {}

    def import_provider_ranking(self):
        """
        import ground truth ranking information from dataset file
        :return:
        """
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

    def sort_scores(self):
        """
        sort ground truth scores as as a whole and for each specialty node list
        :return:
        """
        self.provider_ranking = sorted(self.provider_ranking, key=lambda item: item[1], reverse=True)

    def setup_evaluate(self, graph:Graph):
        """
        setup comparison object for evaluation
        :param graph: graph to get specialty info from
        :return:
        """
        self.import_provider_ranking()
        self.sort_scores()

    def save_results(self, file:str, row):
        """
        save data to csv file in ./results/
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
        for score in self.provider_ranking:
            save_info.append([score[0], score[1]])
        self.save_results("./results/unfilteredActual.csv", save_info)

    def evaluate_hits(self, computed_rank, n=15):
        f"""
        evaluate the hits@n for a given ranking
        :param trimmed_rankings: trimmed rankings for each top specialty
        :param n: get the hits within the top n scores
        :return: dict of specialties: spec : value_name: values, mean : mean hits
        """
        output = []
        mean_hits_at_n = 0

        # calculate hits@n
        hits_at_n = 0

        # get trimmed rankings
        final_computed = computed_rank[:n]
        final_ranked = self.groupify_same_scores(self.provider_ranking)[:n]

        counted = set()
        # check percentage of correct shared (if correct in computed top n, add to total)
        for i in final_computed:
            for j in range(len(final_ranked)):
                for score in final_ranked[j]:
                    if i[0] == score[0]:
                        if i[0] not in counted:
                            counted.add(i[0])
                            hits_at_n += 1
                        break

        # calculate percentage of correct in computed
        hits_at_n /= n
        # add to total of percentages for mean calculation later
        mean_hits_at_n += hits_at_n

        return hits_at_n

    def evaluate_NDCG(self, computed_rank, n=15):
        """
        get the normalized discounted gain for computed vs ground truth
        :param computed_rank: list of (id, score)
        :param n: number of top scores to consider
        :return: dict of specialty : ndcg
        """
        final_computed = computed_rank[:n]
        final_ranked = self.groupify_same_scores(self.provider_ranking)[:n]

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

        return normalized_discounted_gain

    def evaluate_correlation(self, computed_ranking):
        """
        get the correlation between ground truth and computed
        :param trimmed_rankings: dictionary of specialty : computed and ground truth : rankings
        :return: dict of specialty : correlation
        """
        final_computed = self.groupify_same_scores(computed_ranking)
        final_ranked = self.groupify_same_scores(self.provider_ranking)

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
        shared_ids = list(set(computed_positions) & set(true_positions))

        computed_ranks = [computed_positions[id] for id in shared_ids]
        true_ranks = [true_positions[id] for id in shared_ids]

        return spearmanr(true_ranks, computed_ranks).statistic

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

    def trim_rankings(self, computed_ranking:list):
        """
        trim rankings to top n specialties
        :param computed_ranking: ranking computed
        :return: dict of specialty: dict of label:ranking
        """
        final_computed = []
        final_ranked = []
        computed_ranking_dict = dict(computed_ranking)
        # only eval for providers that are shared between computed and rank dataset
        for score in self.provider_ranking:
            if score[0] in computed_ranking_dict:
                final_computed.append((score[0], computed_ranking_dict[score[0]]))
                final_ranked.append(score)

        # remove duplicates (move to dict creation probs)
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

        # remove biases, need to check this
        shuffle(final_computed)

        final_ranked = sorted(final_ranked, key=lambda item: item[1], reverse=True)
        final_computed = sorted(final_computed, key=lambda item: item[1], reverse=True)

        return final_ranked, final_computed


    def evaluate_all_and_save(self, computed_ranking:list, title="unknonwn",
                              save_unfiltered=True, hits_n=15, ndcg_n=15, correlation=True, top_specialties=5,
                              save_type="new"):
        """
        evaluate for all evaluation methods (optionally) and save the output
        :param computed_ranking: dictionary of specialty : scores
        :param title: scoring method name, used for saving
        :param save_unfiltered: if true, save the computed_ranking information to a csv before processing
        :param hits_n: evaluate hits at top n
        :param ndcg_n: evaluate hits at top n
        :param correlation: evaluate correlation to ground truth or not
        :param top_specialties: specialties to consider in comparison, based on most available scores in ground truth
        :param save_type: "new" or "append" or "none" -- how to save data to file
        :return:
        """
        if save_unfiltered:
            print(f"Saving unfiltered results to ./results/results_unfiltered{title.strip()}.csv...")
            with open(f"./results/results_unfiltered{title.strip()}.csv", "w", newline='') as unfiltered:
                write = csv.writer(unfiltered)
                for row in computed_ranking:
                    write.writerow([row[0], row[1]])

        self.provider_ranking, computed_ranking = self.trim_rankings(computed_ranking)

        # calculate evaluations
        evaluations = []
        if hits_n:
            hits_at_n = self.evaluate_hits(computed_ranking, hits_n)
            evaluations.append((hits_at_n, f"hits@{hits_n}"))
        if ndcg_n:
            ndcg_at_n = self.evaluate_NDCG(computed_ranking, ndcg_n)
            evaluations.append((ndcg_at_n, f"NDCG@{ndcg_n}"))
        if correlation:
            correlations = self.evaluate_correlation(computed_ranking)
            evaluations.append((correlations, f"Correlation"))

        # save evaluations to file
        save_file_name = f"./results/results{title.strip()}.csv"
        save_info = []
        save_header = ["eval", "score"]
        save_info.append(save_header)

        eval = evaluations[0]
        save_row = [eval[1], eval[0]]
        for other_eval in evaluations[1:]:
            try:
                save_row.extend([other_eval[1], other_eval[0]])
            except:
                pass
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

    def extract_ranking(self, file:str):
        """
        get ranking from unfiltered ranking file
        :param file: unfiltered rank file name
        :return: dict of specialty : scores
        """
        with open(file, "r") as extract_file:
            extract = csv.reader(extract_file)
            extracted_list = []
            for row in extract:
                print(row)
                npi = int(row[0].strip())
                score = float(row[1].strip())
                extracted_list.append((npi, score))

        return extracted_list

def add_specialties(ranking:list, graph:Graph):
    specialty_ranking = {}
    for entry in ranking:
        print(entry)
        npi = entry[0]
        score = entry[1]
        for specialty in graph.nodes[npi]["specialties"]:
            if specialty in specialty_ranking:
                specialty_ranking[specialty].append(entry)
            else:
                specialty_ranking[specialty] = [entry]

    return specialty_ranking