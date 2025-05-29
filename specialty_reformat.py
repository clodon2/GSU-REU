import csv
import json


def create_new_specialty_csv():
    """
    trim csv dataset file to only get necessary rows
    :return: None
    """
    lines_read = 0
    taxonomy_columns = []
    switch_columns = []
    with open("specialty_data.csv", "r") as data, open("specialty_reformatted.csv", "w", newline='') as reformatted:
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
                    if "Taxonomy Switch" in column:
                        switch_columns.append(i)

            else:
                row = [line[0]]
                for index in taxonomy_columns:
                    row.append(line[index])
                for index in switch_columns:
                    if line[index] == "Y":
                        row.append(row[switch_columns.index(index) + 1])
                        break

                if row[1] != '':
                    csv_writer.writerow(row)


def create_new_specialty_json():
    """
    convert dataset to json
    :return: None
    """
    lines_read = 0
    taxonomy_columns = []
    switch_columns = []
    with open("specialty_data.csv", "r") as data:
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
                    if "Taxonomy Switch" in column:
                        switch_columns.append(i)

            else:
                row = []
                for index in taxonomy_columns:
                    if line[index]:
                        row.append(line[index])
                for index in switch_columns:
                    if line[index] == "Y":
                        row.append(row[switch_columns.index(index)])
                        break

                if row:
                    add_key_value_to_json(line[0], row)


def add_key_value_to_json(key, value):
    try:
        # Read the existing data from the JSON file
        with open("specialty_reformatted.json", "r") as file:
            data = json.load(file)
    except FileNotFoundError:
        # If the file doesn't exist, start with an empty dictionary
        data = {}

    # Add the new key-value pair
    data[key] = value

    # Write the updated dictionary back to the JSON file
    with open("specialty_reformatted.json", "w") as file:
        json.dump(data, file)


if __name__ == "__main__":
    #create_new_specialty_csv()
    create_new_specialty_json()