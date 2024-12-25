from sage.graphs.graph import Graph
from sage.algebras.free_algebra import FreeAlgebra
from sage.matrix.constructor import Matrix  # Optional, only if matrix constructor is required
from sage.rings.rational_field import QQ  # For defining the field of rationals in FreeAlgebra
from sage.combinat.permutation import Permutation

from collections import Counter

import numpy as np
import subprocess

def cr(graph):
    """
    retunrns:
        - dimension of graph
        - fibres rank
        - coherent rank
        - array of configurations of the adjacency matrix
    """
    no_v = len(graph.vertices())
    with open("input.txt", "w") as f:
        f.write(f"order {no_v}\n")
        f.write(f"dimension {no_v}\n")
        
        # Write each row of the adjacency matrix on a single line
        for row in graph.adjacency_matrix():
            f.write(" ".join(map(str, row)) + "\n")
    
    return subprocess.run(["./weisfeiler"], input=open("input.txt", "r").read(),
                            text=True, capture_output=True, shell=True).stdout[:-1]

def get_intervals(lst):
    """Return intervals of consecutive identical elements in a sorted list."""
    intervals = []
    n = len(lst)
    start = 0
    
    for i in range(1, n + 1):
        if i == n or lst[i] != lst[start]:
            intervals.append((start, i - 1))
            start = i
            
    return intervals

    
def type_matrix(results):
    """
    input:
        - results: the results from the weisfeiler algorithm
    output:
        - a list of lists of the condensed type matrix
    """
    dimension = int(results[0])
    matrix_data = list(map(int, filter(None, results[3].split(" "))))
    config_matrix = np.array(matrix_data).reshape((dimension, dimension))
    
    diags = list(map(int, filter(None, results[1].split(" ")[:-1])))
    sorted_diag, perm = zip(*sorted((value, index) for index, value in enumerate(diags)))
    
    perm = [i+1 for i in perm] # making the perm values non-zero
    perm_matrix = Permutation(perm).to_matrix() # matrix generated is actually the transpose, so we multiply Pt A P instead
    sorted_matrix = perm_matrix.T @ config_matrix @ perm_matrix
    
    diags = [sorted_matrix[i, i] for i in range(len(sorted_matrix))]
    # Step 2: Count occurrences of each unique symbol on the diagonal
    counts = Counter(diags)
    
    # Step 3: Sort symbols by their count
    sorted_symbols = sorted(counts, key=lambda x: counts[x])
    
    # Step 4: Create a mapping to new values based on sorted order
    mapping = {symbol: i for i, symbol in enumerate(sorted_symbols)}
    
    # Step 5: Replace each diagonal element in sorted order
    sorted_diag = [mapping[symbol] for symbol in diags]
    
    # Apply sorted diagonal back to sorted_matrix's diagonal
    for i in range(len(sorted_diag)):
        sorted_matrix[i, i] = sorted_diag[i]
    
    # # Generate the permutation order for diagonal elements
    sorted_diag_with_indices = sorted((value, index) for index, value in enumerate(sorted_diag))
    sorted_diag, perm = zip(*sorted_diag_with_indices)
    
    # # Make permutation indices non-zero based on your requirements
    perm = [i + 1 for i in perm]
    
    # Step 6: Create permutation matrix
    perm_matrix = Permutation(perm).to_matrix()
    
    # Apply the permutation as per Pt A P
    sorted_matrix = perm_matrix.T @ sorted_matrix @ perm_matrix
    
    intervals = get_intervals(sorted_diag)
    res = []
    
    for i_start, i_end in intervals:
        tmp = []
        for j_start, j_end in intervals:                    
            # Use set comprehension to gather unique elements in the specified sub-matrix
            unique_elements = {sorted_matrix[row][col] for row in range(i_start, i_end + 1) for col in range(j_start, j_end + 1)}
            tmp.append(len(unique_elements))
        res.append(tmp)
    type_matrix = res
    return type_matrix
    

def switch_graph(vertices, G):
    """
    Switch the edges for the given vertex in the graph G.
    
    Args:
    vertex: The vertex for which the switch operation is performed.
    G: A SageMath Graph object.

    This function modifies the graph in place by switching the edges for the given vertex.
    """

    # Create a copy of the original graph
    new_G = G.copy()
    
    
    for j in G.vertex_iterator():
        for vertex in vertices:
            if vertex == j:
                # Ignore self-loops
                continue
                
            if G.has_edge(vertex, j):
                new_G.delete_edge(vertex, j)  # Remove the edge if it exists in original graph
            else:
                new_G.add_edge(vertex, j)  # Add the edge if it doesn't exist in original graph
    
    # Return the new graph with the switched edges
    return new_G

def delete_vertex(graph):
    new_graph = graph.copy()
    new_graph.delete_vertex(new_graph.vertices()[0])
    return new_graph

def visualise_matrix(sub_matrix):
    contiguous_matrix = np.ascontiguousarray(sub_matrix)
    
    # Convert to a Sage matrix
    adj_matrix = Matrix(contiguous_matrix)
    
    # Create the graph from the adjacency matrix
    G = Graph(adj_matrix)
    
    # Visualize the graph
    G.show()
