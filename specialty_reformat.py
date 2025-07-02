import csv


def create_new_specialty_csv(input_filename="specialty_data.csv",
                             output_filename="specialty_reformatted.csv"):
    """
    trim csv dataset file to only get necessary rows
    :return:
    """
    lines_read = 0
    taxonomy_columns = []
    switch_columns = []
    with (open(input_filename, "r", encoding="utf-8") as data,
          open(output_filename, "w", newline='', encoding="utf-8") as reformatted):
        csv_reader = csv.reader(data)
        csv_writer = csv.writer(reformatted)
        for line in csv_reader:
            # line format:
            # provider, specialties
            lines_read += 1
            print(lines_read)
            # extract data
            # this will probably have to be changed
            if lines_read == 1:
                for i in range(len(line)):
                    column = line[i]
                    if "Taxonomy Code" in column:
                        taxonomy_columns.append(i)
                    if "Taxonomy Switch" in column:
                        switch_columns.append(i)

            else:
                row = [line[0]]
                for index in taxonomy_columns:
                    row.append(line[index])
                for i, index in enumerate(switch_columns):
                    if line[index] == "Y":
                        row.append(row[i + 1])
                        break

                if row[1] != '':
                    csv_writer.writerow(row)


if __name__ == "__main__":
    create_new_specialty_csv(input_filename="./datasets/specialty_2018.csv",
                             output_filename="./datasets/ specialty_2018_reformatted.csv")