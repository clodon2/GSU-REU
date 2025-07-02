import csv

def combine_for_graphs(method_files:list[str]=None, output_file:str="./results/analysis.csv",
                       output_write_type:str="w"):
    print("Combining graph info...")
    mean_title_index = 1
    if method_files is None:
        method_files = ["./results/resultsSheafLaplacian.csv",
                        "./results/resultsRegularLaplacianNew.csv",
                        "./results/resultsKatz.csv",
                        "./results/resultsPageRankNew.csv"]
    method_means = {}
    for method in method_files:
        with open(method, "r") as method_file:
            method_file = csv.reader(method_file)
            means = []
            for line in method_file:
                if line[mean_title_index].strip() == "mean":
                    means.append([line[mean_title_index + 1], line[mean_title_index + 4], line[mean_title_index + 10]])

            # format method name to have a space between words
            method_name = ""
            for char in method[16:-4]:
                if char.isupper():
                    method_name += " "
                method_name += char

            method_means[method_name[1:]] = means

    hits_rows = [["Mean Hits@N"]]
    ndcg_rows = [["Mean NDCG@N"]]
    rbo_rows = [["Mean RBO@N"]]
    for i in range(10, 110, 10):
        hits_rows[0].append(f"{i}")
        ndcg_rows[0].append(f"{i}")
        rbo_rows[0].append(f"{i}")
    print(method_means)
    for method in method_means:
        hit_row = [method]
        ndcg_row = [method]
        rbo_row = [method]
        for hit, ndcg, rbo in method_means[method]:
            print(hit, ndcg, method)
            hit_row.append(hit)
            ndcg_row.append(ndcg)
            rbo_row.append(rbo)

        hits_rows.append(hit_row)
        ndcg_rows.append(ndcg_row)
        rbo_rows.append(rbo_row)
    with open(output_file, output_write_type, newline="") as output:
        output_writer = csv.writer(output)
        output_writer.writerows(hits_rows)
        output_writer.writerow([])
        output_writer.writerows(ndcg_rows)
        output_writer.writerow([])
        output_writer.writerows(rbo_rows)

    print(f"Combined Graph Info saved to {output_file}")


def combine_for_sheaf_graphs(method_files:list[str]=None, output_file:str="./results/analysisSheaf.csv",
                       output_write_type:str="w"):
    print("Combining graph info...")
    mean_title_index = 1
    if method_files is None:
        method_files = ["./results/resultsSheafLaplacianNormalized.csv",
                        "./results/resultsSheafLaplacianNonNormalized.csv",
                        "./results/resultsSheafLaplacianNormalizedWithDegree.csv",
                        "./results/resultsSheafLaplacianNonNormalizedWithDegree.csv"]
    method_means = {}
    for method in method_files:
        with open(method, "r") as method_file:
            method_file = csv.reader(method_file)
            means = []
            for line in method_file:
                if line[mean_title_index].strip() == "mean":
                    means.append([line[mean_title_index + 1], line[mean_title_index + 4], line[mean_title_index + 7]])

            # format method name to have a space between words
            method_name = ""
            for char in method[16:-4]:
                if char.isupper():
                    method_name += " "
                method_name += char

            method_means[method_name[1:]] = means

    hits_rows = [["Mean Hits@N"]]
    ndcg_rows = [["Mean NDCG@N"]]
    rbo_rows = [["Mean RBO@N"]]
    for i in range(10, 110, 10):
        hits_rows[0].append(f"{i}")
        ndcg_rows[0].append(f"{i}")
        rbo_rows[0].append(f"{i}")
    print(method_means)
    for method in method_means:
        hit_row = [method]
        ndcg_row = [method]
        rbo_row = [method]
        for hit, ndcg, rbo in method_means[method]:
            print(hit, ndcg, method)
            hit_row.append(hit)
            ndcg_row.append(ndcg)
            rbo_row.append(rbo)

        hits_rows.append(hit_row)
        ndcg_rows.append(ndcg_row)
        rbo_rows.append(rbo_row)
    with open(output_file, output_write_type, newline="") as output:
        output_writer = csv.writer(output)
        output_writer.writerows(hits_rows)
        output_writer.writerow([])
        output_writer.writerows(ndcg_rows)

    print(f"Combined Graph Info saved to {output_file}")


if __name__ == "__main__":
    combine_for_graphs(output_write_type="w")