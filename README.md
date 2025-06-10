# Finding the Top Healthcare Providers in a Network
## Graph Data Analytics REU at Georgia State University
### Dr. Mehmet Aktas, Corey Verkouteren, Djalil Sawadogo

This program is built to find the top healthcare providers in a given network. This network is modeled as a graph in networkx. In this program, we test 3 methods of finding top providers, including sheaf laplacian, graph (regular) laplacian, and page rank. Our premier method is the sheaf laplacian, explored in an earlier paper on influence maximization.

### Graph Structure
Our graph is built with providers as nodes and edges defined as providers working together. These edges have 3 attributes: pair count (number of times worked together), beneficiary count (unique patients together), and same day count (number of times connections where made in the same day). This data is gathered from the [physician shared patient dataset.](https://www.nber.org/research/data/physician-shared-patient-patterns-data)

### Sheaf Information
Sheaf information is stored in the graph for each node, where each node has a few attributes. First, the specialties a provider (node) has are stored as a list in the "specialties" attribute, with the primary specialty stored in a separate attribute, "primary". This data is gathered from the [specialty dataset](https://download.cms.gov/nppes/NPI_Files.html) Then, another representation is stored in "sheaf_vector" where each specialty is mapped to a number, where the primary specialty is given a weight and the rest are 1s. For example, `["cardiology", "toxicology", "surgery"]` with primary `"toxicology"` is `[1, 2, 1]`.

Additionally, edge value totals for each node are needed later for the the coboundary matrix construction so these are calculated for each node and stored in `"pair_total"` `"beneficiary_total"` and `"same_total"`. 

### Coboundary Map Construction
Our coboundary map for the sheaf laplacian is constructed by adding restriction maps for each vector>edge connection. These restriction maps are found by taking the node `"sheaf_vector"` and summing every value found by multiplying the global edge attribute weights which themselves are multiplied by the percentage that edge has for that provider for each attribute, done for every `"sheaf_vector"` value individually. For example vector with sheaf `[1, 2, 1]`, global edge attribute weights wp, wb, ws `[1, .5, .5]` total edge attributes `[2, 2, 8]` and edge attributes `[2, 2, 4]` will have a restriction map `[1, 2, 1] * ([1, .5, .5] * [2/2, 2/2, 4/8]) = [1, 2, 1] * [1, .5, .25] = [1.75, 3.5, 1.75]`. Additionally, this vector is multiplied by -1 if it is the first listed in the edge tuple. Translating these restriction maps to the coboundary map, each edge is a row and each column is a specialty (value from the restriction map). Thus, the coboundary map is very sparse and is such represented by scipy csr/lil matrices.

### Sheaf Laplacian
The sheaf laplacian is calculated by simply multiplying the transposition of the coboundary map by the original coboundary map.

### Centrality Calculation
Centralities are found by removing an individual specialty column (making the values all 0) and removing the edges of the vector connected to that specialty (making the row values 0). The energy is then calculated by squaring all non-zero values and summing them. This energy is compared to the energy without the specialty removed with `(original - removed) / original` and this final value is the centrality for a specialty of a node. 

### Evaluation
Evaluation is calculated with hits@n and ndcg. Hits@n is the percentage of top n providers that we calculated in the actual top n providers. Normalized discounted gain (NDCG) is calculated by giving each ranking a relevancy score (distance from actual position) and multiplying this by the log of the index, making top placements more impactful to the final score. The actual rankings--ground truth--is from the MIPS score of each provider from the [ranking dataset](https://data.cms.gov/provider-data/dataset/a174-a962#data-table). 

### Output
In the result files, the [taxonomy info dataset](https://data.cms.gov/provider-characteristics/medicare-provider-supplier-enrollment/medicare-provider-and-supplier-taxonomy-crosswalk/data?) is used to name each specialty code.

Physician Data (2015 30 day for now): https://www.nber.org/research/data/physician-shared-patient-patterns-data <br/>
Specialty Data: https://download.cms.gov/nppes/NPI_Files.html or https://data.nber.org/nppes/zip-orig/ <br/>
Taxonomy Info Data: https://data.cms.gov/provider-characteristics/medicare-provider-supplier-enrollment/medicare-provider-and-supplier-taxonomy-crosswalk/data?query=%7B"filters"%3A%7B"rootConjunction"%3A%7B"label"%3A"And"%2C"value"%3A"AND"%7D%2C"list"%3A%5B%5D%7D%2C"keywords"%3A""%2C"offset"%3A270%2C"limit"%3A10%2C"sort"%3A%7B"sortBy"%3Anull%2C"sortOrder"%3Anull%7D%2C"columns"%3A%5B%5D%7D <br/>
Provider Ranking Data: https://data.cms.gov/provider-data/dataset/a174-a962#data-table <br/>
