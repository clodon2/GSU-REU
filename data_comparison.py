import csv
from networkx import Graph


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
            for line in rank_file:
                provider = line[0].strip()
                if line[11].strip() == '':
                    score = 0
                else:
                    score = float(line[11].strip())
                self.provider_ranking.append((int(provider), score))

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

    def compare(self, graph:Graph, computed_ranking:dict, show_lists=True, hits_n=20):
        self.import_provider_ranking()
        self.add_provider_specialties(graph)
        self.import_taxonomy_info()
        self.sort_scores()

        # only get results for top 5 specialties
        specialty_scores = []
        for specialty in self.provider_specialty_ranking:
            specialty_scores.append((specialty, len(self.provider_specialty_ranking[specialty])))

        specialty_scores = sorted(specialty_scores, key=lambda item: item[1], reverse=True)
        specialty_scores = specialty_scores[:5]

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

            for i in range(hits_n):
                if final_ranked[i][0] == final_computed[i][0]:
                    hits_at_n += 1

            for score in final_computed:
                print(f"%10d %15f" % (score[0], score[1]), end=" ")

            print()

            for score in final_ranked:
                print(f"%10d %15f" % (score[0], score[1]), end=" ")

            print()

            hits_at_n /= hits_n

            print(f"Hits at {hits_n} for {self.taxonomy_info[specialty]}:")
            print(hits_at_n)
            print()


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

"""
        for specialty in specialty_scores:
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
"""