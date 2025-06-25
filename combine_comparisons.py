import csv

def combine_for_graphs(method_files:list[str]=None, output_file:str="./results/analysis.csv",
                       output_write_type:str="w"):
    print("Combining graph info...")
    mean_title_index = 1
    if method_files is None:
        method_files = ["./results/resultsSheafLaplacianmips.csv",
                        "./results/resultsRegularLaplacianmips.csv",
                        "./results/resultsDegreesmips.csv",
                        "./results/resultsPageRankmips.csv"]
    method_means = {}
    for method in method_files:
        with open(method, "r") as method_file:
            method_file = csv.reader(method_file)
            means = []
            for line in method_file:
                if line[mean_title_index].strip() == "mean":
                    means.append([line[mean_title_index + 1], line[mean_title_index + 4]])

            # format method name to have a space between words
            method_name = ""
            for char in method[16:-4]:
                if char.isupper():
                    method_name += " "
                method_name += char

            method_means[method_name[1:]] = means

    hits_rows = [["Mean Hits@N"]]
    ndcg_rows = [["Mean NDCG@N"]]
    for i in range(10, 60, 10):
        hits_rows[0].append(f"{i}")
        ndcg_rows[0].append(f"{i}")
    print(method_means)
    for method in method_means:
        hit_row = [method]
        ndcg_row = [method]
        for hit, ndcg in method_means[method]:
            print(hit, ndcg, method)
            hit_row.append(hit)
            ndcg_row.append(ndcg)

        hits_rows.append(hit_row)
        ndcg_rows.append(ndcg_row)
    print(hits_rows)
    with open(output_file, output_write_type, newline="") as output:
        output_writer = csv.writer(output)
        output_writer.writerows(hits_rows)
        output_writer.writerow([])
        output_writer.writerows(ndcg_rows)

    print(f"Combined Graph Info saved to {output_file}")


if __name__ == "__main__":
    combine_for_graphs(output_write_type="w")