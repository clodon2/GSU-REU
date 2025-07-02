from graph_construction import GraphBuilder
from sheaf_laplacian import SheafLaplacian
from other_methods import EvaluationMethods
from data_comparison import CompareData
from data_comparison_no_specialty import CompareDataNoSpecialty, add_specialties
from weight_optimize import DifferentialEvolution
from import_from_outside import get_djalil_coboundary, import_djalil_sheaf_laplacian_centrality, \
    import_djalil_ground_truth, import_djalil_graph
from dataset_analysis import get_score_correlation, find_best_graph_specialties, get_graph_information
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


def remove_whole_from_json():
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

def eval_other_method():
    graph_builder = GraphBuilder(primary_specialty_weight=1.5)
    graph = graph_builder.build_graph(remove_unscored_nodes_file="./datasets/pa_scores_2017.csv",
                                      remove_non_overlap_spec_file="./datasets/specialty_2018_reformatted.csv")
    eval_compare = CompareData()
    eval_compare.setup_evaluate()
    em = EvaluationMethods(graph)
    method_rank = em.betweenness(graph)
    eval_compare.evaluate_all_and_save(method_rank, title="Betweenness", save_unfiltered=True,
                                       save_type="write", hits_n=10, ndcg_n=10, top_specialties=10)
    for i in range(20, 110, 10):
        eval_compare.evaluate_all_and_save(method_rank, title="Betweenness", save_unfiltered=False,
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

def evaluate_all_other_methods():
    graph_builder = GraphBuilder()
    graph = graph_builder.build_graph(remove_unscored_nodes_file="./datasets/pa_scores_2017.csv",
                                      remove_non_overlap_spec_file="./datasets/specialty_2018_reformatted.csv")

    eval_compare = CompareData()
    eval_compare.setup_evaluate()
    # replace this with some other specialty name list

    ev = EvaluationMethods(graph)

    print("page ranking...")
    rankings_pr = ev.page_rank(graph)
    print("degrees...")
    rankings_dg = ev.closeness(graph)
    print("regular laplacian")
    rankings_rl = ev.regular_laplacian(graph)
    print("evaluating...")

    method_rankings = [(rankings_rl, "RegularLaplacianNew"), (rankings_pr, "PageRankNew"), (rankings_dg, "DegreesNew")]
    # , (rankings_sir, "SIR")

    eval_compare.save_actual_rankings()

    for method_info in method_rankings:
        ranking = method_info[0]
        title = method_info[1]
        eval_compare.evaluate_all_and_save(ranking, title=title, save_unfiltered=True,
                                       save_type="write", hits_n=10, ndcg_n=10, top_specialties=10)
        for i in range(20, 110, 10):
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

def get_type_correlation():
    gb = GraphBuilder()
    graph = gb.build_graph(remove_unscored_nodes_file="./datasets/pa_scores_2017.csv",
                           remove_non_overlap_spec_file="./datasets/specialty_2018_reformatted.csv")
    gb.get_graph_stats()
    get_score_correlation(graph, "./datasets/pa_scores_2017.csv")

def get_top_specialties():
    eval_compare = CompareData()
    eval_compare.setup_evaluate()

    specialty_names = eval_compare.get_top_spec_names(n=200, top_spec_num=10)
    print(specialty_names)

def best_spec_graph():
    gb = GraphBuilder()
    graph = gb.build_graph(remove_unscored_nodes_file="./datasets/pa_scores_2017.csv",
                           remove_non_overlap_spec_file="./datasets/specialty_2018_reformatted.csv")
    graph_specs = find_best_graph_specialties(graph, 500)
    eval_compare = CompareData()
    eval_compare.setup_evaluate()
    gt_specs = eval_compare.get_top_specs(n=200, top_spec_num=500)
    gt_lookup = dict(gt_specs)

    best_specialties = []
    for g_specialty, g_number in graph_specs:
        try:
            gt_number = gt_lookup[g_specialty]
            best_specialties.append((g_specialty, min(g_number, gt_number)))
        except:
            pass

    print(sorted(best_specialties, key=lambda item: item[1], reverse=True))
    # [('207R00000X', 44657), ('207Q00000X', 31036), ('207RC0000X', 14335), ('207P00000X', 13557), ('363A00000X', 11946), ('2085R0202X', 10374), ('363LF0000X', 10303), ('208600000X', 9697), ('367500000X', 9331), ('207X00000X', 8442), ('207L00000X', 7944), ('363L00000X', 7580), ('207RG0100X', 7444), ('174400000X', 7138), ('207W00000X', 6742), ('207RP1001X', 6738), ('390200000X', 6575), ('2084N0400X', 5699), ('208M00000X', 5211), ('208800000X', 5091), ('363AM0700X', 4802), ('207RN0300X', 4627), ('207RH0003X', 3964), ('207RC0200X', 3846), ('207N00000X', 3645), ('2084P0800X', 3566), ('363LA2200X', 3455), ('207RI0011X', 3007), ('152W00000X', 2946), ('207RI0200X', 2865), ('363AS0400X', 2804), ('207Y00000X', 2610), ('207ZP0102X', 2509), ('207RE0101X', 2502), ('208100000X', 2485), ('213E00000X', 2368), ('207T00000X', 2249), ('2086S0129X', 2068), ('208G00000X', 2020), ('163W00000X', 1981), ('213ES0103X', 1951), ('2085R0001X', 1874), ('207RR0500X', 1827), ('363LA2100X', 1762), ('207RG0300X', 1606), ('207RC0001X', 1504), ('207LP2900X', 1407), ('208000000X', 1396), ('207RX0202X', 1351), ('207V00000X', 1258), ('2085R0204X', 1257), ('207XX0005X', 1168), ('207PE0004X', 1065), ('207RS0012X', 1064), ('208D00000X', 984), ('2085N0700X', 882), ('207XS0117X', 826), ('2086S0102X', 729), ('207XS0106X', 728), ('208VP0014X', 709), ('207ZP0101X', 698), ('2085B0100X', 692), ('207XS0114X', 621), ('363LG0600X', 620), ('207UN0901X', 609), ('207QG0300X', 599), ('207ND0101X', 588), ('208C00000X', 575), ('2086S0127X', 573), ('2086X0206X', 549), ('207LC0200X', 515), ('2081P2900X', 515), ('207NS0135X', 505), ('367H00000X', 503), ('213ES0131X', 486), ('207QS0010X', 472), ('2085N0904X', 461), ('207XX0801X', 453), ('363LP0808X', 446), ('207RH0000X', 442), ('2085U0001X', 428), ('207RH0002X', 405), ('204F00000X', 393), ('207ND0900X', 376), ('208VP0000X', 368), ('2084N0600X', 366), ('208200000X', 357), ('207ZP0105X', 345), ('207ZC0500X', 342), ('207VX0201X', 333), ('207QA0505X', 327), ('207XX0004X', 325), ('363LP2300X', 316), ('207K00000X', 316), ('207U00000X', 306), ('207YX0905X', 285), ('111N00000X', 282), ('2085P0229X', 273), ('213EP1101X', 267), ('2085R0203X', 252), ('2084S0012X', 251), ('2084P0804X', 240), ('207VG0400X', 239), ('2084V0102X', 228), ('2084P0805X', 222), ('207ZH0000X', 221), ('2086S0122X', 205), ('207RI0008X', 204), ('173000000X', 201), ('282N00000X', 185), ('207WX0107X', 175), ('364SA2200X', 172), ('207QH0002X', 171), ('363LW0102X', 164), ('364S00000X', 160), ('207YS0123X', 158), ('207YX0007X', 155), ('207ZD0900X', 149), ('207YX0602X', 148), ('204D00000X', 142), ('207RA0000X', 128), ('207NP0225X', 125), ('207ZB0001X', 122), ('207PE0005X', 119), ('146D00000X', 119), ('363LC0200X', 119), ('2081S0010X', 113), ('2085D0003X', 107), ('207KA0200X', 106), ('152WC0802X', 105), ('207LP3000X', 102), ('2084N0402X', 101), ('207YX0901X', 96), ('204C00000X', 93), ('2086S0105X', 89), ('207ZN0500X', 89), ('207VF0040X', 84), ('207VX0000X', 83), ('2082S0105X', 81), ('213ES0000X', 81), ('2083P0011X', 77), ('207RA0201X', 77), ('207RS0010X', 77), ('207NI0002X', 77), ('163WC0200X', 72), ('2083X0100X', 68), ('2084P0802X', 58), ('2083P0901X', 56), ('207WX0009X', 55), ('207YP0228X', 54), ('225100000X', 53), ('363LX0001X', 53), ('163WG0000X', 53), ('207QA0401X', 52), ('363LP0200X', 50), ('364SF0001X', 50), ('207XP3100X', 48), ('207PP0204X', 47), ('2083P0500X', 42), ('152WL0500X', 42), ('2084F0202X', 41), ('103TC0700X', 37), ('364SP0809X', 37), ('231H00000X', 35), ('2088P0231X', 35), ('152WP0200X', 34), ('364SP0808X', 32), ('163WP0808X', 31), ('152WV0400X', 29), ('2086S0120X', 26), ('103T00000X', 23), ('207WX0200X', 22), ('204E00000X', 21), ('1223S0112X', 20), ('133V00000X', 20), ('1041C0700X', 19), ('207VM0101X', 19), ('367A00000X', 19), ('2080P0202X', 19), ('2080P0204X', 18), ('2080A0000X', 18), ('207RA0001X', 16), ('2080P0210X', 15), ('122300000X', 15), ('2080P0207X', 14), ('2080P0201X', 14), ('207SG0201X', 13), ('225X00000X', 12), ('2080P0203X', 12), ('2080P0214X', 11), ('2080P0208X', 10), ('1223G0001X', 9), ('104100000X', 9), ('207VE0102X', 7), ('2080P0206X', 6), ('2080P0205X', 6), ('235Z00000X', 5), ('2080N0001X', 2), ('363LN0000X', 1), ('363LN0005X', 1)]

if __name__ == "__main__":
    #eval_sheaf_lap_remove_whole()
    #evaluate_all_methods_whole()
    #evaluate_all_other_methods()
    #eval_other_method()
    gb = GraphBuilder()
    graph = gb.build_graph(remove_unscored_nodes_file="./datasets/pa_scores_2017.csv",
                           remove_non_overlap_spec_file="./datasets/specialty_2018_reformatted.csv")
    get_graph_information(graph, ['207R00000X', '207Q00000X', '363LF0000X', '363A00000X', '207P00000X', '367500000X', '207L00000X', '2085R0202X', '390200000X', '363L00000X'])
    #best_spec_graph()
    #evaluate_all_methods_all_scores()
    #get_type_correlation()
    #evaluate_all_methods()
    #ow = OptimizeWeights()
    #print(ow.find_best_weights())
    # (0.26063333333333333, array([1.777955  , 0.92428796, 2.        , 0.3785489 ])), (0.26063333333333333, array([1.63593104, 0.85589051, 1.85620614, 0.36922718])