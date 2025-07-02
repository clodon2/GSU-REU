# Detecting the Top Healthcare Providers by Specialty in a Network
## Graph Data Analytics REU at Georgia State University
### Dr. Mehmet Aktas, Corey Verkouteren, Djalil Sawadogo

This program is built to find the top healthcare providers within specific specialties in a given network. 
This network is modeled as a graph in networkx. In this program, we test 3 methods of finding top providers, 
including sheaf Laplacian centrality, graph (regular) Laplacian centrality, page rank, and katz centrality. 
Our developed method is the sheaf Laplacian centrality, explored in an earlier paper on influence maximization.
Sheaf Laplacian centrality also consistently performs significantly better when predicting top healthcare providers.

### Getting Results
All of the functions needed to get results from the program are included in main.py, and it is suggested that you run
the functions directly in main.py via the if statement at the bottom of the file. Every function describes exactly what
result you will get out as an output file. 

### Included Results
Included results in "results" are derived from evaluate_sheaf_by_node() and evaluate_all_other_methods(), but 
evaluate_sheaf_by_node() is computationally expensive, so it is suggested that you load the sheaf scores from 
the included files via evaluate_sheaf_from_file() or evalute_sheaf_by_node_from_json(). 

### Graph Structure
Our graph is built with providers as nodes and edges defined as providers working together. 
These edges have 3 attributes: pair count (number of times worked together), 
beneficiary count (unique patients together), and same day count (number of times connections where made in the same 
day). 
This data is gathered from the [physician shared patient dataset.](https://www.nber.org/research/data/physician-shared-patient-patterns-data). 

### Sheaf Information
Numerical specialy information is stored in the graph for each node, where each node has a few attributes. 
First, the specialties a provider (node) has are stored as a list in the `"specialties"` attribute, 
with the primary specialty stored in a separate attribute, `"primary"`. 
This data is gathered from the [specialty dataset](https://download.cms.gov/nppes/NPI_Files.html) Then, another representation is stored in `"sheaf_vector"` 
where each specialty is mapped to a number, where the primary specialty is given a weight and the rest are 1s. 
For example, `["cardiology", "toxicology", "surgery"]` with primary `"toxicology"` is `[1, 1.5, 1]`.

Additionally, edge value totals for each node are stored in `"pair_total"` `"beneficiary_total"` and `"same_total"`. 
These totals can be used to modify the linear transformation in the coboundary map construction.

### Coboundary Map Construction
Our coboundary map for the sheaf laplacian is constructed by adding restriction maps for each vector>edge connection. 
These restriction maps are found by taking the node `"sheaf_vector"` and summing every value found by multiplying the 
global edge attribute weights by the edge attributes. For example vector with sheaf `[1, 2, 1]`, 
global edge attribute weights wp, wb, ws `[1, .5, .5]`  and edge attributes `[2, 2, 4]` 
will have a restriction map `[1, 1.5, 1] * ((2 * 1 + 2 * .5 + 4 * .5) = [1, 1.5, 1] * 5 = [5, 7.5, 5]`. 
Additionally, this vector is multiplied by -1 if it is the first listed in the edge tuple. 
Translating these restriction maps to the coboundary map, each edge is a row and each column is a specialty 
(value from the restriction map). Thus, the coboundary map is very sparse and is such represented by scipy csc 
matrices.

### Sheaf Laplacian
The sheaf laplacian is calculated by simply multiplying the transposition of the coboundary map by the original 
coboundary map.

### Centrality Calculation
Centralities are found by masking all of a node's specialty columns and edges from the coboundary map, recalculating 
the sheaf Laplacian, and summing the square of all entries in the matrix. This gives an energy where the node is removed 
which we compare to the energy of the whole matrix to get the centrality: `(original - removed) / original`.

Optionally, there are included functions where the centrality is calculated by removing individual specialty columns
from the matrix. This method tends to give much worse results, though.

### Evaluation
Evaluation is calculated with Hits@n, NDCG@n, and RBO@n. Hits@n is the percentage of top n providers that we calculated 
in the actual top n providers. Normalized discounted gain (NDCG) is calculated by giving each ranking a relevancy score 
(distance from actual position) and multiplying this by the log of the index, making top placements more impactful to 
the final score. Rank Biased Overlap (RBO) gets the number of overlapping providers at a depth and multiplies by the p 
value to the power of the depth. These numbers for each depth up to n are summed, multiplied by 1 minus p, and then 
summed with p to the power of n multiplied by the percentage of overlap at n to produce a score between 0 and 1.
The actual rankings--ground truth--is from the MIPS score of each provider from 
the [ranking dataset](https://data.cms.gov/provider-data/dataset/a174-a962#data-table). These scores are given based on the specialties of each provider, so a provider could be an 
top provider in multiple specialties, just one, or, if they have low scores, none.

### Output
In the result files, the [taxonomy info dataset](https://data.cms.gov/provider-characteristics/medicare-provider-supplier-enrollment/medicare-provider-and-supplier-taxonomy-crosswalk/data?) is used to name each specialty code. 
For sheaf laplacian, each specialty will have a different centrality, but page rank and regular laplacian have the same 
scores for each specialty of a provider because of how their scoring works. Each method result is stored in a 
separate file.

### Dataset Preprocessing
The specialty datasets should be preprocessed with specialty_reformat.py to reduce size and format them correctly.
This will create new files with the ending _reformatted, which are referenced in main.py

### Dataset Links and Information
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