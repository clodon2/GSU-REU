import networkx as nx
import matplotlib.pyplot as plt
from matplotlib import colormaps
import csv
import numpy as np


class ProviderConnections:
    def __init__(self, provider_data_file:str="pa_data.txt",
                 specialty_data_file:str="specialty_reformatted.csv",
                 graph_data_file:str="physician_graph.gexf"):
        self.provider_data_file = provider_data_file
        self.provider_specialty_data_file = specialty_data_file
        self.graph_data_file = graph_data_file
        self.graph = nx.Graph()

    def import_txt_data(self, rows:int=500):
        """
        add data from the provider txt dataset to self.graph
        :param rows: number of rows of data to add (edges)
        :return: None
        """
        print("importing provider data...")
        lines_read = 0
        with open(self.provider_data_file, "r") as data:
            for line in data:
                # line format:
                # npi1, npi2, pair count, beneficiary count, same day count
                lines_read += 1
                # extract data
                row_data = line.split(",")
                provider1 = int(row_data[0].strip())
                provider2 = int(row_data[1].strip())
                pairs = int(row_data[2].strip())
                benes = int(row_data[3].strip())
                sameday = int(row_data[4].strip())

                self.graph.add_edge(provider1, provider2, weight=pairs, beneficiaries=benes, same_day=sameday)

                # stop at however many rows
                if lines_read >= rows:
                    break

        print(f"{lines_read} edges added")

    def add_specialties_fast(self):
        """
        add specialties to all possible nodes, delete those without specialties
        :return:
        """
        print("adding specialties...")
        line_count = 0
        with open(self.provider_specialty_data_file, "r") as data:
            csv_reader = csv.reader(data)
            for line in csv_reader:
                line_count += 1
                provider = int(line[0])
                if self.graph.has_node(provider):
                    specialties = []
                    for sc in line[1:-1]:
                        if sc:
                            specialties.append(sc)

                    self.graph.nodes[provider]["specialties"] = specialties
                    self.graph.nodes[provider]["primary"] = line[-1]

        print(f"{line_count} in reformatted csv")

        remove_nodes = []
        for node in self.graph.nodes:
            try:
                spec = self.graph.nodes[node]["specialties"]
            except:
                remove_nodes.append(node)
                print(f"no specialties for node {node}")

        for node in remove_nodes:
            self.graph.remove_node(node)

        print(f"{len(remove_nodes)} no specialty nodes removed")


    def build_graph(self):
        """
        create graph structure for providers and add specialties
        :return:
        """
        print("building graph...")
        self.import_txt_data(rows=999999999999999999)
        self.add_specialties_fast()
        self.sheaf_specialty_conversion()
        self.add_provider_totals()
        self.add_coboundary_matrices()


    def save_graph(self):
        """
        save graph locally as graphml file, expensive
        :return:
        """
        print("writing graph...")
        nx.write_graphml(self.graph, self.graph_data_file)


    def load_graph(self):
        """
        update graph to match graph file
        :return:
        """
        print("importing graph...")
        self.graph = nx.read_graphml(self.graph_data_file)


    def sheaf_specialty_conversion(self):
        print("adding numerical specialty vectors...")
        for node in self.graph.nodes:
            num_specialties = []
            for spec in self.graph.nodes[node]["specialties"]:
                if spec == self.graph.nodes[node]["primary"]:
                    num_specialties.append(2)
                else:
                    num_specialties.append(1)

            self.graph.nodes[node]["sheaf_vector"] = np.array(num_specialties)


    def add_provider_totals(self):
        print("adding edge totals to providers...")
        for node in self.graph.nodes:
            self.graph.nodes[node]["pair_total"] = 1
            self.graph.nodes[node]["beneficiary_total"] = 1
            self.graph.nodes[node]["same_total"] = 1
            for connection in self.graph[node]:
                pairs = self.graph[node][connection]["weight"]
                benes = self.graph[node][connection]["beneficiaries"]
                same_days = self.graph[node][connection]["same_day"]

                self.graph.nodes[node]["pair_total"] += pairs
                self.graph.nodes[node]["beneficiary_total"] += benes
                self.graph.nodes[node]["same_total"] += same_days


    def add_coboundary_matrices(self):
        print("adding coboundary matrices to edges...")
        for edge in self.graph.edges:
            edge_attr = self.graph.get_edge_data(edge[0], edge[1])
            edge_pairs = edge_attr["weight"]
            edge_benes = edge_attr["beneficiaries"]
            edge_same_days = edge_attr["same_day"]
            edge_restrictions = []
            for provider in edge:
                pair_percentage = edge_pairs / self.graph.nodes[provider]["pair_total"]
                bene_total = edge_benes / self.graph.nodes[provider]["beneficiary_total"]
                same_day_total = edge_same_days / self.graph.nodes[provider]["same_total"]
                restriction = np.array([pair_percentage, bene_total, same_day_total])
                edge_restrictions.append(self.graph.nodes[provider]["sheaf_vector"] * np.sum(restriction))

            self.graph[edge[0]][edge[1]]["coboundary"] = np.array([edge_restrictions[1] - edge_restrictions[0]])


    def sheaf_linear_transform(self):
        for node in self.graph.nodes:
            if node["specialties"]:
                restriction_map = []
                node["restriction"] = restriction_map
                score = 0
                # linear transformation of specialties with restriction map to unify data
                for s, r in zip(node["specialties"], restriction_map):
                    score += s * r
                node["score"] = score

    def sheaf_laplacian(self):
        sheaf_laplacian = []
        for edge in self.graph.edges:
            try:
                # basically just the two restriction maps combined
                for r in self.graph.nodes[edge[0]]["restriction"]:
                    r *= -1
                coboundary_map = self.graph.nodes[edge[0]]["restriction"].extend(self.graph.nodes[edge[1]]["restriction"])
                # linear multiply column coboundary map x row coboundary map
                for x in coboundary_map:
                    row = []
                    for y in coboundary_map:
                        row.append(x * y)
                    sheaf_laplacian.append(row)
                # revert restriction
                for r in self.graph.nodes[edge[0]]["restriction"]:
                    r *= -1
            except:
                print("Error: restriction not available for one or more nodes")

        return sheaf_laplacian

    def draw_graph(self, edge_colors:bool=True, edge_labels:bool=True):
        """
        draw self.graph in a new window
        :param edge_colors: color edges based on intensity (darker larger weight)
        :param edge_labels: label weights of edges
        :return:
        """
        # best layouts
        # can look confusing
        pos = nx.spring_layout(self.graph, seed=7443, k=20)
        # probably the best but intensive
        #pos = nx.kamada_kawai_layout(self.graph)
        # circular (like a better spring)
        #pos = nx.forceatlas2_layout(self.graph)
        # good variety of distances but can look crowded
        pos = nx.fruchterman_reingold_layout(self.graph)

        if edge_colors:
            color_map = colormaps["Blues"]
            weights = [self.graph[u][v]["weight"] for u, v in self.graph.edges()]
            # scale color map to maximum weight in graph
            #min_weight = min(weights)
            max_weight = max(weights)
        else:
            color_map = None
            weights = None
            max_weight = 0

        if edge_labels:
            edge_weights = nx.get_edge_attributes(self.graph, "weight")
            nx.draw_networkx_edge_labels(self.graph, pos, edge_weights)

        nx.draw_networkx(self.graph, pos=pos, with_labels=True,
                         edge_color=weights, edge_cmap=color_map, edge_vmin=0, edge_vmax=max_weight)
        plt.show()


if __name__ == "__main__":
    pc = ProviderConnections()
    pc.build_graph()
    #pc.sheaf_laplacian()
    #pc.draw_graph(edge_colors=True, edge_labels=True)