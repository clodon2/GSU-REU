from graph_construction import GraphBuilder
from sheaf_laplacian import SheafLaplacian
from other_methods import EvaluationMethods
from data_comparison import CompareData
from weight_optimize import DifferentialEvolution
from dataset_analysis import find_best_graph_specialties, get_graph_information
from combine_comparisons import combine_for_graphs
import time
import json


class OptimizeWeights:
    def __init__(self):
        graph_builder = GraphBuilder(primary_specialty_weight=2)
        graph = graph_builder.build_graph(rows=999999999999999, remove_unscored_nodes_file="./datasets/pa_scores_2017.csv",
                                          remove_non_overlap_spec_file="./datasets/specialty_2018_reformatted.csv")
        self.sheaf_laplacian = SheafLaplacian(graph=graph,
                                              coboundary_columns=graph_builder.coboundary_columns,
                                              restriction_weights=[1, 1, 1],
                                              primary_specialty_weight=2)
        self.eval_compare = CompareData()
        self.eval_compare.setup_evaluate(score_index=20)
        self.top_scores = self.eval_compare.get_top_spec_names(100, 10)

    def get_weight_score(self, weights):
        self.sheaf_laplacian.primary_specialty_weight = weights[:1]
        self.sheaf_laplacian.restriction_weights = weights[1:]
        sheaf_laplacian_rankings = self.sheaf_laplacian.compute_all_give_rankings(only_top_specialties=self.top_scores)
        return self.eval_compare.get_mean_score(sheaf_laplacian_rankings, top_specialties=10)

    def find_best_weights(self):
        start = time.time()
        DE = DifferentialEvolution(population_size=8, problem_dimensions=4, iterations=20, scaling_factor=.6,
                                   crossover_chance=.7, search_space=[0, 2], fitness_function=self.get_weight_score)
        results = DE.run()
        end = time.time() - start
        print(f"finished DE algo in {end}")
        return results


def evalute_sheaf_by_node_from_json():
    """
    save evaluations for sheaf laplacian centrality where centrality is found
    by removing the whole node from the coboundary map, read from json instead of recomputing
    :return:
    """
    with open("./results/removeByNodeRankings.json") as file:
        sorted_rankings = {}
        rankings = json.load(file)
        for specialty in rankings:
            values = rankings[specialty]
            # reorder to see best provider
            sorted_rankings[specialty] = sorted(values.items(), key=lambda item: item[1], reverse=True)

        int_rankings = {}

        for specialty in sorted_rankings:
            converted_scores = []
            for npi, centrality in sorted_rankings[specialty]:
                converted_scores.append((int(npi), float(centrality)))
            int_rankings[specialty] = converted_scores

        sorted_rankings = int_rankings

        eval_compare = CompareData()
        eval_compare.setup_evaluate()
        eval_compare.evaluate_all_and_save(sorted_rankings, title="SheafLaplacian", save_unfiltered=True,
                                           save_type="write", hits_n=10, ndcg_n=10, top_specialties=10)
        for i in range(20, 110, 10):
            eval_compare.evaluate_all_and_save(sorted_rankings, title="SheafLaplacian", save_unfiltered=False,
                                               save_type="append", hits_n=i, ndcg_n=i, top_specialties=10)


def evaluate_sheaf_by_node():
    """
    save evaluations for sheaf laplacian centrality where centrality is found
    by removing the whole node from the coboundary map
    :return:
    """
    graph_builder = GraphBuilder(primary_specialty_weight=1.5)
    graph = graph_builder.build_graph(remove_unscored_nodes_file="./datasets/pa_scores_2017.csv",
                                      remove_non_overlap_spec_file="./datasets/specialty_2018_reformatted.csv")
    sheaf_laplacian = SheafLaplacian(graph,
                                     graph_builder.coboundary_columns,
                                     restriction_weights=[1, 1, 1], primary_specialty_weight=1.5)
    eval_compare = CompareData()
    eval_compare.setup_evaluate()
    top_specs = eval_compare.get_top_spec_names(200, 10)
    sheaf_laplacian_rankings = sheaf_laplacian.compute_all_give_rankings_whole_removal(top_specs)
    eval_compare.evaluate_all_and_save(sheaf_laplacian_rankings, title="SheafLaplacian", save_unfiltered=True,
                                       save_type="write", hits_n=10, ndcg_n=10, top_specialties=10)
    for i in range(20, 110, 10):
        eval_compare.evaluate_all_and_save(sheaf_laplacian_rankings, title="SheafLaplacian", save_unfiltered=False,
                                           save_type="append", hits_n=i, ndcg_n=i, accuracy_n=i, top_specialties=10)


def evaluate_sheaf_by_specialty():
    """
    save evaluations for sheaf laplacian centrality where centrality is found
    by removing the specialty columns of a node from the coboundary map
    :return:
    """
    graph_builder = GraphBuilder(primary_specialty_weight=1.5)
    graph = graph_builder.build_graph(remove_unscored_nodes_file="./datasets/pa_scores_2017.csv",
                                      remove_non_overlap_spec_file="./datasets/specialty_2018_reformatted.csv")
    sheaf_laplacian = SheafLaplacian(graph,
                                     graph_builder.coboundary_columns,
                                     restriction_weights=[.92428796, 2, .3785489], primary_specialty_weight=1.777955)
    # restriction_weights=[0.48546858, -1.72720085, 1.51242945], primary_specialty_weight=1.05053757)
    eval_compare = CompareData()
    eval_compare.setup_evaluate()
    top_specs = eval_compare.get_top_spec_names(200, 10)
    sheaf_laplacian_rankings = sheaf_laplacian.compute_all_give_rankings(top_specs)
    eval_compare.evaluate_all_and_save(sheaf_laplacian_rankings, title="SheafLaplacian", save_unfiltered=True,
                                       save_type="write", hits_n=10, ndcg_n=10, top_specialties=10)
    for i in range(20, 110, 10):
        eval_compare.evaluate_all_and_save(sheaf_laplacian_rankings, title="SheafLaplacian", save_unfiltered=False,
                                           save_type="append", hits_n=i, ndcg_n=i, accuracy_n=i, top_specialties=10)



def evaluate_all_methods_all_scores_sheaf_by_specialty():
    """
    save evaluations for sheaf laplacian centrality where centrality is found
    by removing the specialty columns of a node from the coboundary map, also evaluate all other methods
    over all score categories
    :return:
    """
    graph_builder = GraphBuilder()
    graph = graph_builder.build_graph(remove_unscored_nodes_file="./datasets/pa_scores_2017.csv",
                                      remove_non_overlap_spec_file="./datasets/specialty_2018_reformatted.csv")

    sheaf_laplacian = SheafLaplacian(graph=graph,
                                     coboundary_columns=graph_builder.coboundary_columns,
                                     restriction_weights=[.92428796, 2, .3785489], primary_specialty_weight=1.777955)
    # restriction_weights=[0.48546858, -1.72720085, 1.51242945], primary_specialty_weight=1.05053757
    eval_compare = CompareData()
    eval_compare.setup_evaluate()

    specialty_names = eval_compare.get_top_spec_names(n=200, top_spec_num=10)
    sheaf_laplacian_rankings = sheaf_laplacian.compute_all_give_rankings(only_top_specialties=specialty_names)

    ev = EvaluationMethods(graph)

    print("page ranking...")
    rankings_pr = ev.page_rank(graph)
    print("regular laplacian...")
    rankings_rl = ev.regular_laplacian(graph)
    print("katz...")
    rankings_kz = ev.katz(graph)
    print("evaluating...")

    method_rankings = [(sheaf_laplacian_rankings, "SheafLaplacian"), (rankings_pr, "PageRank"),
                       (rankings_rl, "RegularLaplacian"), (rankings_kz, "Katz")]

    eval_compare.save_actual_rankings()

    # quality=24, pi=47, ia=75, cost=85, mip=20
    score_indices = [("Mips", 20), ("Quality", 24), ("PI", 47), ("IA", 75), ("Cost", 85)]
    for score_method in score_indices:
        score_name = score_method[0]
        score_index = score_method[1]
        eval_compare.setup_evaluate(score_index)
        for method_info in method_rankings:
            ranking = method_info[0]
            title = method_info[1]
            eval_compare.evaluate_all_and_save(ranking, title=title + score_name, save_unfiltered=True,
                                           save_type="write", hits_n=10, ndcg_n=10, top_specialties=10)
            for i in range(20, 60, 10):
                eval_compare.evaluate_all_and_save(ranking, title=title + score_name, save_unfiltered=False,
                                                   save_type="append", hits_n=i, ndcg_n=i, top_specialties=10)


def evaluate_sheaf_from_file():
    """
    evaluate sheaf laplacian from pre-computed centralities stored in csv
    :return:
    """
    eval_compare = CompareData()
    eval_compare.setup_evaluate()
    sheaf_laplacian_rankings = eval_compare.extract_ranking("./results/results_unfilteredSheafLaplacian.csv")
    eval_compare.evaluate_all_and_save(sheaf_laplacian_rankings, title="SheafLaplacian", save_unfiltered=True,
                                       save_type="write", hits_n=10, ndcg_n=10, top_specialties=10)
    for n in range(20, 110, 10):
        eval_compare.evaluate_all_and_save(sheaf_laplacian_rankings, title="SheafLaplacian", save_unfiltered=False,
                                           save_type="append", hits_n=n, ndcg_n=n, top_specialties=10)


def evaluate_other_method(method_name:str="katz"):
    """
    evaluate one non-sheaf laplacian method
    :return:
    """
    method_name = method_name.title().strip().replace(" ", "")
    graph_builder = GraphBuilder(primary_specialty_weight=1.5)
    graph = graph_builder.build_graph(remove_unscored_nodes_file="./datasets/pa_scores_2017.csv",
                                      remove_non_overlap_spec_file="./datasets/specialty_2018_reformatted.csv")
    eval_compare = CompareData()
    eval_compare.setup_evaluate()
    em = EvaluationMethods(graph)

    methods = {"PageRank":em.page_rank,
               "RegularLaplacian":em.regular_laplacian,
               "Katz":em.katz}

    if method_name not in methods:
        raise "Method Name Not Recognized"

    method_rank = methods[method_name](graph)
    eval_compare.evaluate_all_and_save(method_rank, title=method_name, save_unfiltered=True,
                                       save_type="write", hits_n=10, ndcg_n=10, top_specialties=10)
    for i in range(20, 110, 10):
        eval_compare.evaluate_all_and_save(method_rank, title=method_name, save_unfiltered=False,
                                           save_type="append", hits_n=i, ndcg_n=i, top_specialties=10)



def evaluate_all_other_methods():
    """
    evaluate all non-sheaf laplacian centrality methods
    :return:
    """
    graph_builder = GraphBuilder()
    graph = graph_builder.build_graph(remove_unscored_nodes_file="./datasets/pa_scores_2017.csv",
                                      remove_non_overlap_spec_file="./datasets/specialty_2018_reformatted.csv")

    eval_compare = CompareData()
    eval_compare.setup_evaluate()
    # replace this with some other specialty name list

    ev = EvaluationMethods(graph)

    print("page ranking...")
    rankings_pr = ev.page_rank(graph)
    print("katz...")
    rankings_kz = ev.katz(graph)
    print("regular laplacian")
    rankings_rl = ev.regular_laplacian(graph)
    print("evaluating...")

    method_rankings = [(rankings_rl, "RegularLaplacian"), (rankings_pr, "PageRank"), (rankings_kz, "Katz")]

    eval_compare.save_actual_rankings()

    for method_info in method_rankings:
        ranking = method_info[0]
        title = method_info[1]
        eval_compare.evaluate_all_and_save(ranking, title=title, save_unfiltered=True,
                                       save_type="write", hits_n=10, ndcg_n=10, top_specialties=10)
        for i in range(20, 110, 10):
            eval_compare.evaluate_all_and_save(ranking, title=title, save_unfiltered=False,
                                               save_type="append", hits_n=i, ndcg_n=i, top_specialties=10)


def get_top_specialties():
    """
    print the top 10 specialties (used in evaluation)
    :return:
    """
    eval_compare = CompareData()
    eval_compare.setup_evaluate()

    specialty_names = eval_compare.get_top_spec_names(n=200, top_spec_num=10)
    print(specialty_names)


def get_graph_statistics():
    """
    get the nodes, edges, average degrees, and average specialties for thw whole graph and
    the organized by the top specialties
    :return:
    """
    graph_builder = GraphBuilder()
    graph = graph_builder.build_graph(remove_unscored_nodes_file="./datasets/pa_scores_2017.csv",
                                      remove_non_overlap_spec_file="./datasets/specialty_2018_reformatted.csv")
    eval_compare = CompareData()
    eval_compare.setup_evaluate()

    specialty_names = eval_compare.get_top_spec_names(n=100, top_spec_num=10)
    get_graph_information(graph, specialty_names)


if __name__ == "__main__":
    evaluate_sheaf_by_node()
    evaluate_all_other_methods()