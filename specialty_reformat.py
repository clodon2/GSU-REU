import csv


def create_new_specialty_csv(self, rows: int = 500):
    """
    add specialty data to nodes in graph
    :param rows: number of rows of data to go through (nodes to update)
    :return: None
    """
    lines_read = 0
    taxonomy_columns = []
    with open("specialty_data.csv", "r") as data, open("specialty_reformatted.csv", "w") as reformatted:
        csv_reader = csv.reader(data)
        csv_writer = csv.writer(reformatted)
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
                row = [line[0]]
                for index in taxonomy_columns:
                    row.append(line[index])
                csv_writer.writerow(row)

            # stop at however many rows
            if lines_read >= rows:
                break