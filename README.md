# Identifying Top Performing  Providers: A Sheaf-theoretic Approach on Healthcare Networks
By Corey Verkouteren, Djalil Sawadogo, Mehmet Aktas, and Esra Akbas

This program is built to find the top healthcare providers within specific specialties in a given network. 
This network is modeled as a graph in networkx. In this program, we test 3 methods of finding top providers, 
including sheaf Laplacian centrality, graph (regular) Laplacian centrality, page rank, and katz centrality. 
Our developed method is the sheaf Laplacian centrality, explored in an earlier paper on influence maximization.
Sheaf Laplacian centrality also consistently performs significantly better when predicting top healthcare providers.

## Getting Results
All of the functions needed to get results from the program are included in main.py, and it is suggested that you run
the functions directly in main.py via the if statement at the bottom of the file. Every function describes exactly what
result you will get out as an output file. 

## Included Results
Included results in "results" are derived from evaluate_sheaf_by_node() and evaluate_all_other_methods(), but 
evaluate_sheaf_by_node() is computationally expensive, so it is suggested that you load the sheaf scores from 
the included files via evaluate_sheaf_from_file() or evalute_sheaf_by_node_from_json().

## Dataset Preprocessing
The specialty datasets should be preprocessed with specialty_reformat.py to reduce size and format them correctly.
This will create new files with the ending _reformatted, which are referenced in main.py

## Dataset Links and Information
Physician Data (2015 30 day): https://www.nber.org/research/data/physician-shared-patient-patterns-data <br/>
Specialty Data (2018 for scoring, 2015 for graph): https://data.nber.org/nppes/zip-orig/ <br/>
Taxonomy Info Data: https://data.cms.gov/provider-characteristics/medicare-provider-supplier-enrollment/medicare-provider-and-supplier-taxonomy-crosswalk/data <br />
Provider Ranking Data (2017): https://healthdata.gov/dataset/Quality-Payment-Program-Experience/bxpa-574c/about_data <br/>

**Move all dataset files to the "datasets" folder.**

**Default filenames (rename to these):** <br/>
Physician Data: pa_data.txt <br/>
Specialty Data: specialty_2015.csv, specialty_2018.csv <br/>
Taxonomy Data: taxonomy_info.csv <br/>
Provider Ranking Data: pa_scores_2017.csv <br/>
