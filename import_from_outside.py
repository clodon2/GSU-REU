from joblib import load
import json
from scipy.sparse import csr_matrix


def import_djalil_graph():
    file_name = "./datasets/graph_with_attributes.gpickle"
    graph = load(file_name)
    return graph


def import_djalil_ground_truth():
    file_name = "./datasets/path_to_ec_score_file_order_by_specialties.json"
    with open(file_name, 'r') as f:
        scores = json.load(f)
    for specialty in scores:
        tuple_scores = []
        for score in scores[specialty]:
            tuple_scores.append(tuple(score))
        scores[specialty] = tuple_scores
    return scores


def import_djalil_sheaf_laplacian_centrality():
    file_name = "./datasets/sheaf_laplacian_centrality.json"
    with open(file_name, 'r') as f:
        centralities = json.load(f)
    for score in centralities:
        centralities[score] = tuple(centralities[score])
    return centralities

def get_djalil_coboundary():
    graph = import_djalil_graph()
    # Step 1: Get the dimensions of specialties for each node
    node_dimension = {}
    for node in list(graph.nodes()):
        sp_vec_len = len(graph.nodes[node]["specialties"])  # Lenght of specialties_vector = specialties
        node_dimension[node] = sp_vec_len

    # Step 2: Assign global column indices to each node's specialties
    node_indices = {}
    col = 0
    for node, dim in node_dimension.items():
        node_indices[node] = list(
            range(col, col + dim))  # look like {"node1": [1], "node2":[2,3], "node3": [4], "node4": [5,6,7]}
        col += dim
        graph.nodes[node]["indicies"] = node_indices[node]
    node_indices = node_indices

    cols_num = col  # Total columns = sum of all vertex dimensions
    rows_num = len(list(graph.edges()))  # Number of rows = number of edges

    # Step 3: Process edges and build the coboundary map
    edges_tail_head = {}
    for edge in list(graph.edges()):
        u, v = edge  # tail and head
        u_v_rest_map = graph.nodes[u][f"linked_to_{v}"][v]  # restriction map from u to v
        v_u_rest_map = graph.nodes[v][f"linked_to_{u}"][u]  # restriction map from v to u
        edges_tail_head[edge] = (u, v, u_v_rest_map, v_u_rest_map)

    # Step 4: Process the edges and add data to the coboundary map
    nonzero_restriction_map = []
    nrm_rows_indices = []
    nrm_cols_indices = []
    for edg_idx, (edge, (u, v, u_v_rest_map, v_u_rest_map)) in enumerate(edges_tail_head.items()):
        # u = tail, v = head
        # Add u_v_rest_map to columns of u
        for col_idx_u, val_u in zip(node_indices[u], u_v_rest_map):
            nonzero_restriction_map.append(val_u)
            nrm_rows_indices.append(edg_idx)
            nrm_cols_indices.append(col_idx_u)
        # u = tail, v = head
        # Add -(v_u_rest_map) to columns of v
        for col_idx_v, val_v in zip(node_indices[v], v_u_rest_map):
            nonzero_restriction_map.append(-val_v)
            nrm_rows_indices.append(edg_idx)
            nrm_cols_indices.append(col_idx_v)

    delta = csr_matrix((nonzero_restriction_map, (nrm_rows_indices, nrm_cols_indices)),
                             shape=(rows_num, cols_num))
    print("\nCoboundary Map Successfully Computed!!!")
    return delta, graph