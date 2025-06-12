from graph_construction import GraphBuilder
from sheaf_laplacian import SheafLaplacian
from other_methods import EvaluationMethods
from data_comparison import CompareData
from weight_optimize import DifferentialEvolution


def eval_sheaf_lap():
    graph_builder = GraphBuilder(primary_specialty_weight=2)
    graph = graph_builder.build_graph(rows=1000)
    sheaf_laplacian = SheafLaplacian(graph,
                                     graph_builder.coboundary_columns,
                                     restriction_weights=[1, 1, 1])
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
    graph_builder = GraphBuilder(primary_specialty_weight=2)
    graph = graph_builder.build_graph(rows=10000)
    # 1, .1, .05
    sheaf_laplacian = SheafLaplacian(graph=graph,
                                     coboundary_columns=graph_builder.coboundary_columns,
                                     restriction_weights=[0, .23854934, .62817431])
    sheaf_laplacian_rankings = sheaf_laplacian.compute_all_give_rankings()
    # replace this with some other specialty name list
    specialty_names = list(sheaf_laplacian_rankings.keys())

    eval_compare = CompareData()
    eval_compare.setup_evaluate(graph)

    ev = EvaluationMethods(graph)

    print("page ranking...")
    rankings_pr = ev.page_rank_all_specialties(specialty_names)
    print("regular laplacian...")
    rankings_rl = ev.regular_laplacian()
    print("SIR...")
    #rankings_sir = ev.SIR_vectors(specialty_names)
    print("evaluating...")

    method_rankings = [(sheaf_laplacian_rankings, "SheafLaplacian"), (rankings_pr, "PageRank"),
                       (rankings_rl, "RegularLaplacian")]
    # , (rankings_sir, "SIR")

    eval_compare.save_actual_rankings()

    for method_info in method_rankings:
        ranking = method_info[0]
        title = method_info[1]
        eval_compare.evaluate_all_and_save(ranking, title=title, save_unfiltered=True,
                                       save_type="write", hits_n=10, ndcg_n=10)
        for i in range(20, 50, 10):
            eval_compare.evaluate_all_and_save(ranking, title=title, save_unfiltered=False,
                                               save_type="append", hits_n=i, ndcg_n=i)

class OptimizeWeights:
    def __init__(self):
        graph_builder = GraphBuilder(primary_specialty_weight=2)
        graph = graph_builder.build_graph(rows=10000)
        self.sheaf_laplacian = SheafLaplacian(graph=graph,
                                              coboundary_columns=graph_builder.coboundary_columns,
                                              restriction_weights=[1, 1, 1])
        self.eval_compare = CompareData()
        self.eval_compare.setup_evaluate(graph)

    def get_weight_score(self, weights):
        self.sheaf_laplacian.restriction_weights = weights
        sheaf_laplacian_rankings = self.sheaf_laplacian.compute_all_give_rankings()
        return self.eval_compare.get_mean_score(sheaf_laplacian_rankings)

    def find_best_weights(self):
        DE = DifferentialEvolution(population_size=20, problem_dimensions=3, iterations=20, scaling_factor=.5,
                                   crossover_chance=.7, search_space=[0, 1], fitness_function=self.get_weight_score)
        return DE.run()


if __name__ == "__main__":
    evaluate_all_methods()
    #ow = OptimizeWeights()
    #print(ow.find_best_weights())
    # suggested weights at rows=1000 [0.77577079, 0.        , 1.        ]
    # rows=10000 (0.08504771744480666, array([1.        , 0.85599797, 0.76237238]))
    # new suggested at 5000
    # (0.19478033188859017, array([0.99688818, 0.43987648, 0.94187009]))