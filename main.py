import networkx as nx
import matplotlib.pyplot as plt
from matplotlib import colormaps
import csv


class ProviderConnections:
    def __init__(self, provider_data_file:str="pa_data.txt",
                 specialty_data_file:str="specialty_data.csv"):
        self.provider_data_file = provider_data_file
        self.provider_specialty_data_file = specialty_data_file
        self.graph = nx.Graph()

    def import_txt_data(self, rows:int=500):
        """
        add data from the provider txt dataset to self.graph
        :param rows: number of rows of data to add (edges)
        :return: None
        """
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

                self.graph.add_edge(provider1, provider2, weight=pairs)

                # stop at however many rows
                if lines_read >= rows:
                    break

    def add_specialties(self, rows:int=500):
        """
        add specialty data to nodes in graph
        :param rows: number of rows of data to go through (nodes to update)
        :return: None
        """
        lines_read = 0
        taxonomy_columns = []
        with open(self.provider_specialty_data_file, "r") as data:
            csv_reader = csv.reader(data)
            for line in csv_reader:
                # line format:
                # provider, specialties
                lines_read += 1
                # extract data
                # this will probably have to be changed
                if lines_read == 1:
                    for i in range(len(line)):
                        column = line[i]
                        if "Taxonomy Code" in column:
                            taxonomy_columns.append(i)

                else:
                    if self.graph.nodes[line[0]]:
                        specialties = []
                        for index in taxonomy_columns:
                            if line[index]:
                                specialties.append(line[index])

                        print(specialties)
                        self.graph.nodes[line[0]]["specialties"] = specialties

                # stop at however many rows
                if lines_read >= rows:
                    break

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
    pc.import_txt_data(rows=100)
    #pc.add_specialties(rows=50)
    #pc.sheaf_laplacian()
    pc.draw_graph(edge_colors=True, edge_labels=True)