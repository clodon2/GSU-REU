from graph_construction import GraphBuilder
from sheaf_laplacian import SheafLaplacian
from other_methods import EvaluationMethods
from data_comparison import CompareData
from data_comparison_no_specialty import CompareDataNoSpecialty, add_specialties
from weight_optimize import DifferentialEvolution
from import_from_outside import get_djalil_coboundary, import_djalil_sheaf_laplacian_centrality, \
    import_djalil_ground_truth, import_djalil_graph
from dataset_analysis import get_score_correlation


def eval_sheaf_lap():
    graph_builder = GraphBuilder(primary_specialty_weight=2)
    graph = graph_builder.build_graph(rows=1000)
    sheaf_laplacian = SheafLaplacian(graph,
                                     graph_builder.coboundary_columns,
                                     restriction_weights=[1, 1, 1],
                                     primary_specialty_weight=2)
    eval_compare = CompareData()
    eval_compare.setup_evaluate(graph)
    sheaf_laplacian_rankings = sheaf_laplacian.compute_all_give_rankings()
    eval_compare.evaluate_all_and_save(sheaf_laplacian_rankings, title="SheafLaplacian", save_unfiltered=True,
                                       save_type="write", hits_n=10, ndcg_n=10)
    eval_compare.evaluate_all_and_save(sheaf_laplacian_rankings, title="SheafLaplacian", save_unfiltered=True,
                                       save_type="append", hits_n=20, ndcg_n=20)
    eval_compare.evaluate_all_and_save(sheaf_laplacian_rankings, title="SheafLaplacian", save_unfiltered=True,
                                       save_type="append", hits_n=30, ndcg_n=30)
    eval_compare.evaluate_all_and_save(sheaf_laplacian_rankings, title="SheafLaplacian", save_unfiltered=True,
                                       save_type="append", hits_n=40, ndcg_n=40)

    sheaf_laplacian.restriction_weights = [1, .1, .05]
    sheaf_laplacian_rankings = sheaf_laplacian.compute_all_give_rankings()
    eval_compare.evaluate_all_and_save(sheaf_laplacian_rankings, title="SheafLaplacianC", save_unfiltered=True,
                                       save_type="write", hits_n=10, ndcg_n=10)
    eval_compare.evaluate_all_and_save(sheaf_laplacian_rankings, title="SheafLaplacianC", save_unfiltered=True,
                                       save_type="append", hits_n=20, ndcg_n=20)
    eval_compare.evaluate_all_and_save(sheaf_laplacian_rankings, title="SheafLaplacianC", save_unfiltered=True,
                                       save_type="append", hits_n=30, ndcg_n=30)
    eval_compare.evaluate_all_and_save(sheaf_laplacian_rankings, title="SheafLaplacianC", save_unfiltered=True,
                                       save_type="append", hits_n=40, ndcg_n=40)


def evaluate_all_methods():
    graph_builder = GraphBuilder()
    graph = graph_builder.build_graph(remove_unscored_nodes_file="pa_scores.csv")
    # 1, .1, .05
    sheaf_laplacian = SheafLaplacian(graph=graph,
                                     coboundary_columns=graph_builder.coboundary_columns,
                                     restriction_weights=[.36131, 1.1985, 1.888], primary_specialty_weight=1.0434)
    sheaf_laplacian_rankings = sheaf_laplacian.compute_all_give_rankings()
    # replace this with some other specialty name list

    eval_compare = CompareData()
    eval_compare.setup_evaluate(graph)

    specialty_names = eval_compare.get_top_spec_names(5)

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

    for method_info in method_rankings:
        ranking = method_info[0]
        title = method_info[1]
        eval_compare.evaluate_all_and_save(ranking, title=title, save_unfiltered=True,
                                       save_type="write", hits_n=10, ndcg_n=10, top_specialties=10)
        for i in range(20, 60, 10):
            eval_compare.evaluate_all_and_save(ranking, title=title, save_unfiltered=False,
                                               save_type="append", hits_n=i, ndcg_n=i, top_specialties=10)

class OptimizeWeights:
    def __init__(self):
        graph_builder = GraphBuilder(primary_specialty_weight=2)
        graph = graph_builder.build_graph(rows=10000, remove_unscored_nodes_file="pa_scores.csv")
        self.sheaf_laplacian = SheafLaplacian(graph=graph,
                                              coboundary_columns=graph_builder.coboundary_columns,
                                              restriction_weights=[1, 1, 1],
                                              primary_specialty_weight=2)
        self.eval_compare = CompareData()
        self.eval_compare.setup_evaluate(graph)

    def get_weight_score(self, weights):
        self.sheaf_laplacian.primary_specialty_weight = weights[:1]
        self.sheaf_laplacian.restriction_weights = weights[1:]
        sheaf_laplacian_rankings = self.sheaf_laplacian.compute_all_give_rankings()
        return self.eval_compare.get_mean_score(sheaf_laplacian_rankings)

    def find_best_weights(self):
        DE = DifferentialEvolution(population_size=20, problem_dimensions=4, iterations=20, scaling_factor=.5,
                                   crossover_chance=.7, search_space=[0, 2], fitness_function=self.get_weight_score)
        return DE.run()


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
    specialty_num = 100
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
    specialty_num = 100
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
    graph = gb.build_graph(remove_unscored_nodes_file="pa_scores.csv")
    gb.get_graph_stats()
    get_score_correlation(graph, "pa_scores.csv")

def build_graph_test():
    gb = GraphBuilder()
    graph = gb.build_graph(remove_unscored_nodes_file="pa_scores_new.csv")
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
    get_real_ranking_all_spec_removal()
    #get_type_correlation()
    #eval_djalil_centrality_direct()
    #eval_djalil_no_spec_import()
    #evaluate_all_methods()
    #ow = OptimizeWeights()
    #print(ow.find_best_weights())
    # suggested weights at 1000: (0.5440900111139723, array([0.69067041, 0.59055783, 0.        ]))