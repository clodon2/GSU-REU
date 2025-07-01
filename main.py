from graph_construction import GraphBuilder
from sheaf_laplacian import SheafLaplacian
from other_methods import EvaluationMethods
from data_comparison import CompareData
from data_comparison_no_specialty import CompareDataNoSpecialty, add_specialties
from weight_optimize import DifferentialEvolution
from import_from_outside import get_djalil_coboundary, import_djalil_sheaf_laplacian_centrality, \
    import_djalil_ground_truth, import_djalil_graph
from dataset_analysis import get_score_correlation
from combine_comparisons import combine_for_graphs
import time
import json


def load_in_from_json():
    with open("removeWholeReserve.json") as file:
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
        eval_compare.evaluate_all_and_save(sorted_rankings, title="SheafLaplacianWhole", save_unfiltered=True,
                                           save_type="write", hits_n=10, ndcg_n=10, top_specialties=10)
        for i in range(20, 110, 10):
            eval_compare.evaluate_all_and_save(sorted_rankings, title="SheafLaplacianWhole", save_unfiltered=False,
                                               save_type="append", hits_n=i, ndcg_n=i, top_specialties=10)

def evaluate_all_methods_whole():
    with open("removeWholeReserve.json") as file:
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

    sheaf_laplacian_rankings = int_rankings

    graph_builder = GraphBuilder()
    graph = graph_builder.build_graph(remove_unscored_nodes_file="./datasets/pa_scores_2017.csv",
                                      remove_non_overlap_spec_file="./datasets/specialty_2018_reformatted.csv")

    eval_compare = CompareData()
    eval_compare.setup_evaluate()

    specialty_names = eval_compare.get_top_spec_names(n=200, top_spec_num=10)
    # replace this with some other specialty name list

    ev = EvaluationMethods(graph)

    print("page ranking...")
    rankings_pr = ev.page_rank_all_specialties(specialty_names)
    print("degrees...")
    rankings_dg = ev.degrees(specialty_names)
    print("regular laplacian...")
    rankings_rl = ev.regular_laplacian(specialty_names)
    print("evaluating...")

    method_rankings = [(sheaf_laplacian_rankings, "SheafLaplacian"), (rankings_pr, "PageRank"),
                       (rankings_dg, "Degrees"), (rankings_rl, "RegularLaplacian")]
    # , (rankings_sir, "SIR")

    eval_compare.save_actual_rankings()

    for method_info in method_rankings:
        ranking = method_info[0]
        title = method_info[1]
        eval_compare.evaluate_all_and_save(ranking, title=title, save_unfiltered=True,
                                       save_type="write", hits_n=10, ndcg_n=10, accuracy_n=10,top_specialties=10)
        for i in range(20, 110, 10):
            eval_compare.evaluate_all_and_save(ranking, title=title, save_unfiltered=False,
                                               save_type="append", hits_n=i, ndcg_n=i, accuracy_n=i,top_specialties=10)

def eval_spec_number():
    graph_builder = GraphBuilder(primary_specialty_weight=2)
    graph = graph_builder.build_graph(remove_unscored_nodes_file="./datasets/pa_scores_2017.csv", remove_non_overlap_spec_file="./datasets/specialty_2018_reformatted.csv")
    total_specs = 0
    for node in graph.nodes:
        for spec in graph.nodes[node]["specialties"]:
            total_specs += 1

    all_indices = []
    for node in graph.nodes:
        all_indices.append(graph.nodes[node]["indices"][-1])

    print("total specialties in graph:", total_specs)
    print("max index in cm: ", max(all_indices))

def eval_other_method():
    graph_builder = GraphBuilder(primary_specialty_weight=2)
    graph = graph_builder.build_graph(remove_unscored_nodes_file="./datasets/pa_scores_2017.csv",
                                      remove_non_overlap_spec_file="./datasets/specialty_2018_reformatted.csv")
    eval_compare = CompareData()
    eval_compare.setup_evaluate()
    em = EvaluationMethods(graph)
    method_rank = em.regular_laplacian(eval_compare.get_top_spec_names(100, 10))
    eval_compare.evaluate_all_and_save(method_rank, title="RegularLaplacian", save_unfiltered=True,
                                       save_type="write", hits_n=10, ndcg_n=10, top_specialties=10)
    for i in range(20, 60, 10):
        eval_compare.evaluate_all_and_save(method_rank, title="RegularLaplacian", save_unfiltered=False,
                                           save_type="append", hits_n=i, ndcg_n=i, top_specialties=10)


def eval_sheaf_lap():
    graph_builder = GraphBuilder(primary_specialty_weight=1.5)
    graph = graph_builder.build_graph(remove_unscored_nodes_file="./datasets/pa_scores_2017.csv",
                                      remove_non_overlap_spec_file="./datasets/specialty_2018_reformatted.csv")
    sheaf_laplacian = SheafLaplacian(graph,
                                     graph_builder.coboundary_columns,
                                     restriction_weights=[.92428796, 2, .3785489], primary_specialty_weight=1.777955)
    # restriction_weights=[0.48546858, -1.72720085, 1.51242945], primary_specialty_weight=1.05053757)
    eval_compare = CompareData()
    eval_compare.setup_evaluate()
    top_specs = eval_compare.get_top_spec_names(100, 10)
    print(f"passed top specs: {top_specs}")
    sheaf_laplacian_rankings = sheaf_laplacian.compute_all_give_rankings(top_specs)
    eval_compare.evaluate_all_and_save(sheaf_laplacian_rankings, title="SheafLaplacian", save_unfiltered=True,
                                       save_type="write", hits_n=10, ndcg_n=10, top_specialties=10)
    eval_compare.evaluate_all_and_save(sheaf_laplacian_rankings, title="SheafLaplacian", save_unfiltered=False,
                                       save_type="append", hits_n=20, ndcg_n=20, top_specialties=10)
    eval_compare.evaluate_all_and_save(sheaf_laplacian_rankings, title="SheafLaplacian", save_unfiltered=False,
                                       save_type="append", hits_n=30, ndcg_n=30, top_specialties=10)
    eval_compare.evaluate_all_and_save(sheaf_laplacian_rankings, title="SheafLaplacian", save_unfiltered=False,
                                       save_type="append", hits_n=40, ndcg_n=40, top_specialties=10)
    eval_compare.evaluate_all_and_save(sheaf_laplacian_rankings, title="SheafLaplacian", save_unfiltered=False,
                                       save_type="append", hits_n=50, ndcg_n=50, top_specialties=10)

def eval_sheaf_lap_remove_whole():
    graph_builder = GraphBuilder(primary_specialty_weight=1.5)
    graph = graph_builder.build_graph(remove_unscored_nodes_file="./datasets/pa_scores_2017.csv",
                                      remove_non_overlap_spec_file="./datasets/specialty_2018_reformatted.csv")
    sheaf_laplacian = SheafLaplacian(graph,
                                     graph_builder.coboundary_columns,
                                     restriction_weights=[1, 1, 1], primary_specialty_weight=1.5)
    eval_compare = CompareData()
    eval_compare.setup_evaluate()
    top_specs = eval_compare.get_top_spec_names(100, 10)
    sheaf_laplacian_rankings = sheaf_laplacian.compute_all_give_rankings_whole_removal(top_specs)
    eval_compare.evaluate_all_and_save(sheaf_laplacian_rankings, title="SheafLaplacianWhole", save_unfiltered=True,
                                       save_type="write", hits_n=10, ndcg_n=10, top_specialties=10)
    for i in range(20, 110, 10):
        eval_compare.evaluate_all_and_save(sheaf_laplacian_rankings, title="SheafLaplacianWhole", save_unfiltered=False,
                                           save_type="append", hits_n=i, ndcg_n=i, accuracy_n=i, top_specialties=10)

def eval_degree_results():
    eval_compare = CompareData()
    eval_compare.setup_evaluate()
    degree_ranks = eval_compare.extract_ranking("./results/results_unfilteredDegrees.csv")
    top_specs = eval_compare.get_top_spec_names(100, 10)
    for i, spec in enumerate(top_specs):
        print("Specialty", i)
        print(spec, sorted(degree_ranks[spec], reverse=True, key=lambda score: score[1])[:20])
        print(spec, eval_compare.groupify_same_scores(eval_compare.provider_specialty_ranking[spec])[:20])

def eval_sheaf_lap_from_file():
    eval_compare = CompareData()
    eval_compare.setup_evaluate()
    sheaf_laplacian_rankings = eval_compare.extract_ranking("./results/results_unfilteredSheafLaplacian.csv")
    eval_compare.evaluate_all_and_save(sheaf_laplacian_rankings, title="SheafLaplacian", save_unfiltered=True,
                                       save_type="write", hits_n=10, ndcg_n=10, top_specialties=10)
    eval_compare.evaluate_all_and_save(sheaf_laplacian_rankings, title="SheafLaplacian", save_unfiltered=False,
                                       save_type="append", hits_n=20, ndcg_n=20, top_specialties=10)
    eval_compare.evaluate_all_and_save(sheaf_laplacian_rankings, title="SheafLaplacian", save_unfiltered=False,
                                       save_type="append", hits_n=30, ndcg_n=30, top_specialties=10)
    eval_compare.evaluate_all_and_save(sheaf_laplacian_rankings, title="SheafLaplacian", save_unfiltered=False,
                                       save_type="append", hits_n=40, ndcg_n=40, top_specialties=10)
    eval_compare.evaluate_all_and_save(sheaf_laplacian_rankings, title="SheafLaplacian", save_unfiltered=False,
                                       save_type="append", hits_n=50, ndcg_n=50, top_specialties=10)

def evaluate_all_methods():
    graph_builder = GraphBuilder()
    graph = graph_builder.build_graph(remove_unscored_nodes_file="./datasets/pa_scores_2017.csv",
                                      remove_non_overlap_spec_file="./datasets/specialty_2018_reformatted.csv")
    # 1, .1, .05
    sheaf_laplacian = SheafLaplacian(graph=graph,
                                     coboundary_columns=graph_builder.coboundary_columns,
                                     restriction_weights=[.92428796, 2, .3785489], primary_specialty_weight=1.777955)
    # [1.777955  , 0.92428796, 2.        , 0.3785489 ]

    eval_compare = CompareData()
    eval_compare.setup_evaluate()

    specialty_names = eval_compare.get_top_spec_names(n=100, top_spec_num=10)
    sheaf_laplacian_rankings = sheaf_laplacian.compute_all_give_rankings(only_top_specialties=specialty_names)
    # replace this with some other specialty name list

    ev = EvaluationMethods(graph)

    print("page ranking...")
    rankings_pr = ev.page_rank_all_specialties(specialty_names)
    print("degrees...")
    rankings_dg = ev.degrees(specialty_names)
    print("evaluating...")

    method_rankings = [(sheaf_laplacian_rankings, "SheafLaplacian"), (rankings_pr, "PageRank"), (rankings_dg, "Degrees")]
    # , (rankings_sir, "SIR")

    eval_compare.save_actual_rankings()

    for method_info in method_rankings:
        ranking = method_info[0]
        title = method_info[1]
        eval_compare.evaluate_all_and_save(ranking, title=title, save_unfiltered=True,
                                       save_type="write", hits_n=10, ndcg_n=10, top_specialties=10)
        for i in range(20, 60, 10):
            eval_compare.evaluate_all_and_save(ranking, title=title, save_unfiltered=False,
                                               save_type="append", hits_n=i, ndcg_n=i, top_specialties=10)

def evaluate_all_methods_all_scores():
    graph_builder = GraphBuilder()
    graph = graph_builder.build_graph(remove_unscored_nodes_file="./datasets/pa_scores_2017.csv",
                                      remove_non_overlap_spec_file="./datasets/specialty_2018_reformatted.csv")
    # 1, .1, .05
    sheaf_laplacian = SheafLaplacian(graph=graph,
                                     coboundary_columns=graph_builder.coboundary_columns,
                                     restriction_weights=[.92428796, 2, .3785489], primary_specialty_weight=1.777955)
    # restriction_weights=[0.48546858, -1.72720085, 1.51242945], primary_specialty_weight=1.05053757
    eval_compare = CompareData()
    eval_compare.setup_evaluate()

    specialty_names = eval_compare.get_top_spec_names(n=100, top_spec_num=10)
    sheaf_laplacian_rankings = sheaf_laplacian.compute_all_give_rankings(only_top_specialties=specialty_names)
    # replace this with some other specialty name list

    ev = EvaluationMethods(graph)

    print("page ranking...")
    rankings_pr = ev.page_rank_all_specialties(specialty_names)
    print("regular laplacian...")
    rankings_rl = ev.regular_laplacian(specialty_names)
    print("degrees...")
    rankings_dg = ev.degrees(specialty_names)
    print("evaluating...")

    method_rankings = [(sheaf_laplacian_rankings, "SheafLaplacian"), (rankings_pr, "PageRank"),
                       (rankings_rl, "RegularLaplacian"), (rankings_dg, "Degrees")]
    # , (rankings_sir, "SIR")

    eval_compare.save_actual_rankings()

    # quality=24, pi=47, ia=75, cost=85, mip=20
    score_indices = [("mips", 20), ("quality", 24), ("pi", 47), ("ia", 75), ("cost", 85)]
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


def load_djalil_stuff():
    coboundary_map, graph = get_djalil_coboundary()
    sheaf_laplacian = SheafLaplacian(graph, 5)
    sheaf_laplacian.coboundary_map = coboundary_map
    sheaf_laplacian.compute_sheaf_laplacian()
    ranking = sheaf_laplacian.compute_centralities_multiprocessing()
    return ranking

def eval_djalil_no_spec():
    coboundary_map, graph = get_djalil_coboundary()
    sheaf_laplacian = SheafLaplacian(graph, 5, primary_specialty_weight=1)
    sheaf_laplacian.coboundary_map = coboundary_map
    sheaf_laplacian.compute_sheaf_laplacian()
    rankings = sheaf_laplacian.compute_centralities_multiprocessing_remove_whole()
    eval_compare = CompareDataNoSpecialty()
    specialty_num = 100
    eval_compare.evaluate_all_and_save(rankings, title="Whole", save_unfiltered=True,
                                       ndcg_n=10, hits_n=10, save_type="write", top_specialties=specialty_num)
    eval_compare.evaluate_all_and_save(rankings, title="Whole", save_unfiltered=False,
                                       ndcg_n=20, hits_n=20, save_type="append", top_specialties=specialty_num)
    eval_compare.evaluate_all_and_save(rankings, title="Whole", save_unfiltered=False,
                                       ndcg_n=30, hits_n=30, save_type="append", top_specialties=specialty_num)
    eval_compare.evaluate_all_and_save(rankings, title="Whole", save_unfiltered=False,
                                       ndcg_n=40, hits_n=40, save_type="append", top_specialties=specialty_num)
    eval_compare.evaluate_all_and_save(rankings, title="Whole", save_unfiltered=False,
                                       ndcg_n=50, hits_n=50, save_type="append", top_specialties=specialty_num)


def eval_djalil_no_spec_import():
    coboundary_map, graph = get_djalil_coboundary()
    eval_compare = CompareDataNoSpecialty()
    rankings = eval_compare.extract_ranking("./results/results_unfilteredWhole.csv")
    eval_compare.setup_evaluate(graph)
    specialty_num = 100
    eval_compare.evaluate_all_and_save(rankings, title="Whole", save_unfiltered=False,
                                       ndcg_n=10, hits_n=10, save_type="write", top_specialties=specialty_num)
    eval_compare.evaluate_all_and_save(rankings, title="Whole", save_unfiltered=False,
                                       ndcg_n=20, hits_n=20, save_type="append", top_specialties=specialty_num)
    eval_compare.evaluate_all_and_save(rankings, title="Whole", save_unfiltered=False,
                                       ndcg_n=30, hits_n=30, save_type="append", top_specialties=specialty_num)
    eval_compare.evaluate_all_and_save(rankings, title="Whole", save_unfiltered=False,
                                       ndcg_n=40, hits_n=40, save_type="append", top_specialties=specialty_num)
    eval_compare.evaluate_all_and_save(rankings, title="Whole", save_unfiltered=False,
                                       ndcg_n=50, hits_n=50, save_type="append", top_specialties=specialty_num)


def eval_djalil_normal():
    rankings = load_djalil_stuff()
    eval_compare = CompareData()
    eval_compare.provider_specialty_ranking = import_djalil_ground_truth()
    specialty_num = 25
    eval_compare.evaluate_all_and_save(rankings, title="djalilSheaf", save_unfiltered=True,
                                       ndcg_n=10, hits_n=10, save_type="write", top_specialties=specialty_num)
    eval_compare.evaluate_all_and_save(rankings, title="djalilSheaf", save_unfiltered=False,
                                       ndcg_n=20, hits_n=20, save_type="append", top_specialties=specialty_num)
    eval_compare.evaluate_all_and_save(rankings, title="djalilSheaf", save_unfiltered=False,
                                       ndcg_n=30, hits_n=30, save_type="append", top_specialties=specialty_num)
    eval_compare.evaluate_all_and_save(rankings, title="djalilSheaf", save_unfiltered=False,
                                       ndcg_n=40, hits_n=40, save_type="append", top_specialties=specialty_num)
    eval_compare.evaluate_all_and_save(rankings, title="djalilSheaf", save_unfiltered=False,
                                       ndcg_n=50, hits_n=50, save_type="append", top_specialties=specialty_num)

def eval_djalil_centrality_direct():
    rankings = import_djalil_sheaf_laplacian_centrality()
    eval_compare = CompareData()
    eval_compare.provider_specialty_ranking = import_djalil_ground_truth()
    specialty_num = 25
    eval_compare.evaluate_all_and_save(rankings, title="djalilNew", save_unfiltered=True,
                                       ndcg_n=10, hits_n=10, save_type="write", top_specialties=specialty_num)
    eval_compare.evaluate_all_and_save(rankings, title="djalilNew", save_unfiltered=False,
                                       ndcg_n=20, hits_n=20, save_type="append", top_specialties=specialty_num)
    eval_compare.evaluate_all_and_save(rankings, title="djalilNew", save_unfiltered=False,
                                       ndcg_n=30, hits_n=30, save_type="append", top_specialties=specialty_num)
    eval_compare.evaluate_all_and_save(rankings, title="djalilNew", save_unfiltered=False,
                                       ndcg_n=40, hits_n=40, save_type="append", top_specialties=specialty_num)
    eval_compare.evaluate_all_and_save(rankings, title="djalilNew", save_unfiltered=False,
                                       ndcg_n=50, hits_n=50, save_type="append", top_specialties=specialty_num)

def eval_djalil_all():
    sheaf_laplacian_rankings = import_djalil_sheaf_laplacian_centrality()
    eval_compare = CompareData()
    eval_compare.provider_specialty_ranking = import_djalil_ground_truth()
    specialty_num = 5

    graph = import_djalil_graph()

    ev = EvaluationMethods(graph)

    print("page ranking...")
    rankings_pr = ev.page_rank_all_specialties(eval_compare.get_top_spec_names(specialty_num))
    print("regular laplacian...")
    rankings_rl = ev.regular_laplacian(eval_compare.get_top_spec_names(specialty_num))
    print("degrees...")
    rankings_dg = ev.degrees(eval_compare.get_top_spec_names(specialty_num))

    method_rankings = [(sheaf_laplacian_rankings, "SheafLaplacian"), (rankings_pr, "PageRank"),
                       (rankings_rl, "RegularLaplacian"), (rankings_dg, "Degrees")]

    eval_compare.save_actual_rankings()

    for method_info in method_rankings:
        ranking = method_info[0]
        title = method_info[1]
        eval_compare.evaluate_all_and_save(ranking, title=title, save_unfiltered=True,
                                       save_type="write", hits_n=10, ndcg_n=10, top_specialties=specialty_num)
        for i in range(20, 60, 10):
            eval_compare.evaluate_all_and_save(ranking, title=title, save_unfiltered=False,
                                               save_type="append", hits_n=i, ndcg_n=i, top_specialties=specialty_num)

def get_real_ranking_all_spec_removal():
    coboundary_map, graph = get_djalil_coboundary()
    eval_compare = CompareDataNoSpecialty()
    rankings = eval_compare.extract_ranking("./results/results_unfilteredWhole.csv")
    convert_rank = add_specialties(rankings, graph)
    eval_compare = CompareData()
    eval_compare.setup_evaluate()
    eval_compare.evaluate_all_and_save(convert_rank, title="WholeAll", save_unfiltered=False,
                                       save_type="write", hits_n=10, ndcg_n=10, top_specialties=25)
    for i in range(20, 60, 10):
        eval_compare.evaluate_all_and_save(convert_rank, title="WholeAll", save_unfiltered=False,
                                           save_type="append", hits_n=i, ndcg_n=i, top_specialties=25)

def get_type_correlation():
    gb = GraphBuilder()
    graph = gb.build_graph(remove_unscored_nodes_file="./datasets/pa_scores_2017.csv",
                           remove_non_overlap_spec_file="./datasets/specialty_2018_reformatted.csv")
    gb.get_graph_stats()
    get_score_correlation(graph, "./datasets/pa_scores_2017.csv")

def build_graph_test():
    gb = GraphBuilder()
    graph = gb.build_graph(remove_unscored_nodes_file="./datasets/pa_scores_new.csv")
    gb.get_graph_stats()
    coboundary_map, dgraph = get_djalil_coboundary()
    diff_nodes = []
    for node in dgraph.nodes:
        if node not in graph.nodes:
            diff_nodes.append(node)

    print(len(diff_nodes), dgraph.number_of_nodes() - graph.number_of_nodes())
    for node in diff_nodes:
        print(node, dgraph.nodes[node])


if __name__ == "__main__":
    #load_in_from_json()
    #eval_sheaf_lap_remove_whole()
    evaluate_all_methods_whole()
    #evaluate_all_methods_all_scores()
    #get_type_correlation()
    #eval_djalil_centrality_direct()
    #eval_djalil_no_spec_import()
    #evaluate_all_methods()
    #ow = OptimizeWeights()
    #print(ow.find_best_weights())
    # (0.26063333333333333, array([1.777955  , 0.92428796, 2.        , 0.3785489 ])), (0.26063333333333333, array([1.63593104, 0.85589051, 1.85620614, 0.36922718])