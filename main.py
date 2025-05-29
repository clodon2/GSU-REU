import networkx as nx
import matplotlib.pyplot as plt
from matplotlib import colormaps
import csv


class ProviderConnections:
    def __init__(self, provider_data_file:str="pa_data.txt",
                 specialty_data_file:str="specialty_reformatted.csv"):
        self.provider_data_file = provider_data_file
        self.provider_specialty_data_file = specialty_data_file
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

                self.graph.add_edge(provider1, provider2, weight=pairs)

                # stop at however many rows
                if lines_read >= rows:
                    break

        print(lines_read)

    def add_specialties_fast(self):
        """
        add specialties to all possible nodes, delete those without specialties
        :return:
        """
        print("adding specialties...")
        with open(self.provider_specialty_data_file, "r") as data:
            csv_reader = csv.reader(data)
            for line in csv_reader:
                provider = int(line[0])
                if self.graph.has_node(provider):
                    specialties = []
                    for sc in line[1:-1]:
                        if sc:
                            specialties.append(sc)

                    self.graph.nodes[provider]["specialties"] = specialties
                    self.graph.nodes[provider]["primary"] = line[-1]

        for node in self.graph.nodes:
            try:
                print(self.graph.nodes[node]["specialties"])
            except:
                print(f"no specialties for node {node}")

    def add_specialties(self):
        """
        **deprecated**
        add specialty data to nodes in graph
        :return: None
        """
        remove_nodes = []
        with open(self.provider_specialty_data_file, "r") as data:
            csv_reader = csv.reader(data)
            for provider in self.graph.nodes:
                data.seek(0)
                csv_reader = csv.reader(data)
                specialty_set = False
                count = 0
                for line in csv_reader:
                    count += 1
                    # line format:
                    # provider, specialty1. specialty2, etc., primary spec
                    # extract data
                    if int(provider) == int(line[0]):
                        print("id seen")
                        specialty_set = True
                        specialties = []
                        for sc in line[1:-1]:
                            specialties.append(sc)

                        self.graph.nodes[provider]["specialties"] = specialties
                        self.graph.nodes[provider]["primary"] = line[-1]
                        break
                print(count)
                if not specialty_set:
                    remove_nodes.append(provider)

        for node in remove_nodes:
            print(f"removed node: {node}")
            self.graph.remove_node(node)


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
    pc.import_txt_data(rows=99999999999999999999999)
    print(pc.graph.number_of_nodes())
    pc.add_specialties_fast()
    #pc.add_specialties()
    #pc.add_specialties(rows=50)
    #pc.sheaf_laplacian()
    #pc.draw_graph(edge_colors=True, edge_labels=True)