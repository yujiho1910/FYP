from sage.graphs.graph import Graph
from sage.graphs.graph_generators import graphs
from sage.algebras.free_algebra import FreeAlgebra
from sage.matrix.constructor import Matrix  # Optional, only if matrix constructor is required
from sage.rings.rational_field import QQ  # For defining the field of rationals in FreeAlgebra
from sage.combinat.permutation import Permutation

from abc import ABC, abstractmethod
from collections import Counter

import numpy as np
import subprocess

from sage.graphs.graph_generators import graphs

class FYP_Graph:
    def __init__(self, obj):
        """
        Initialize the graph with a SageMath Graph object.

        Args:
            obj (sage.graphs.graph.Graph): A SageMath Graph object.
        """
        self.obj = obj
        self.dimension = obj.order() 
        self.adjacency_matrix = obj.adjacency_matrix() 
        self.type_matrix = None
        self.coherent_rank = None
        self.weisfeiler_results = None

    def __repr__(self):
        """prints out all attributes of the graph"""
        return f"Graph object with attributes: {self.__dict__}"

    def get_weisfeiler_results(self):
        """
        Compute and return the Weisfeiler results of the graph.
        Placeholder implementation for Weisfeiler-Leman refinement.
        """
        self.weisfeiler_results = cr(self.obj).split("\n")
        return self.weisfeiler_results

    def get_coherent_rank(self):
        """
        Compute and return the coherent rank of the graph.
        """
        if self.coherent_rank is not None:
            return self.coherent_rank
        
        if self.weisfeiler_results is None:
            self.get_weisfeiler_results()

        self.coherent_rank = self.weisfeiler_results[2]
        return self.coherent_rank

    def show_graph(self):
        """
        Visualize the graph using SageMath's built-in plot function.
        """
        return self.obj.show()

    def get_adjacency_matrix(self):
        """
        Return the adjacency matrix of the graph.
        """
        return self.adjacency_matrix

    def get_type_matrix(self):
        """
        Compute and return the type matrix of the graph.
        Placeholder implementation: degrees of vertices.
        """
        if self.type_matrix is not None:
            return self.type_matrix
        
        if self.weisfeiler_results is None:
            self.get_weisfeiler_results()

        self.type_matrix = type_matrix(self.weisfeiler_results)
        return self.type_matrix

    def switch_graph(self, i):
        """
        Perform edge switching for the first vertex and return a new graph.

        Args:
            i: index of vertex to switch edges

        Returns:
            FYP_Graph: A new graph object with switched edges.

        """
        
        new_g = self.obj.copy()
        vertex = self.obj.vertices()[i]

        for j in self.obj.vertex_iterator():

            if vertex == j:
                continue
                
            if self.obj.has_edge(vertex, j):
                new_g.delete_edge(vertex, j)  # Remove the edge if it exists in original graph
            else:
                new_g.add_edge(vertex, j)  # Add the edge if it doesn't exist in original graph
        
        return FYP_Graph(new_g)

    def delete_vertex(self, i):
        """
        Delete the first vertex and return a new graph without the vertex.

        Args:
            i: index of vertex to delete

        Returns:
            FYP_Graph: A new graph object without the first vertex.
        """
        new_graph = self.obj.copy()
        new_graph.delete_vertex(new_graph.vertices()[i])
        return FYP_Graph(new_graph)
    
    def new_line_graph(self):
        """
        Creates a line graph of the current graph (copy) and return it as a new FYP_Graph object

        Args:
            None

        Returns:
            FYP_Graph: A new graph object that is the line graph of the current graph
        """
        new_graph = self.obj.copy()
        line_graph = new_graph.line_graph()

        return FYP_Graph(line_graph)


class OA_Graph(FYP_Graph):
    def __init__(self, m, n):
        """
        Initialize an Orthogonal Array Block Graph.
        """
        obj = self._create_orthogonal_array_block_graph(m, n)
        super().__init__(obj)

    def _create_orthogonal_array_block_graph(self, m, n):
        """
        Create an Orthogonal Array Block Graph.

        Replace this logic with the actual implementation.
        """
        # Placeholder logic: Create a complete graph as an example
        return graphs.OrthogonalArrayBlockGraph(m, n)


class Triangle_Graph(FYP_Graph):
    def __init__(self, n):
        """
        Initialize the line graph of a Complete Graph.
        """
        obj = self._create_complete_graph_line_graph(n)
        super().__init__(obj)

    def _create_complete_graph_line_graph(self, n):
        """
        Create the line graph of a Complete Graph.
        """
        complete_graph = graphs.CompleteGraph(n).line_graph()
        return complete_graph

# Add on more types of graph here (follow the above 2 conventions)



# Helper functions

def cr(graph):
    """
    returns:
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
        - a list of lists of the condensed type matrix, reordered so that its diagonal is in ascending order
    """
    dimension = int(results[0])
    matrix_data = list(map(int, filter(None, results[3].split(" "))))
    config_matrix = np.array(matrix_data).reshape((dimension, dimension))
    
    # ----------------------------------------------------------------
    # 1) Initial diagonal labels
    # ----------------------------------------------------------------
    diags = list(map(int, filter(None, results[1].split(" ")[:-1])))
    
    # Sort by ascending diagonal values, build permutation perm
    sorted_diag, perm = zip(*sorted((value, index) for index, value in enumerate(diags)))
    
    # Convert to 1-based indices for Sage
    perm = [i+1 for i in perm]
    
    # Apply Pt * A * P
    perm_matrix = Permutation(perm).to_matrix()
    sorted_matrix = perm_matrix.T @ config_matrix @ perm_matrix
    
    # ----------------------------------------------------------------
    # 2) Frequency-based re-labeling of diagonal
    # ----------------------------------------------------------------
    diags = [sorted_matrix[i, i] for i in range(len(sorted_matrix))]
    counts = Counter(diags)
    
    # Sort symbols by their count (if you need to break ties numerically, use (counts[x], x))
    sorted_symbols = sorted(counts, key=lambda x: counts[x])
    
    # Map old symbols -> new labels
    mapping = {symbol: i for i, symbol in enumerate(sorted_symbols)}
    
    # Replace diagonal with newly mapped labels
    updated_diag = [mapping[symbol] for symbol in diags]
    for i in range(len(updated_diag)):
        sorted_matrix[i, i] = updated_diag[i]
    
    # ----------------------------------------------------------------
    # 3) Second permutation: reorder by the new diagonal labels
    # ----------------------------------------------------------------
    sorted_diag_with_indices = sorted((value, index) for index, value in enumerate(updated_diag))
    sorted_diag, perm = zip(*sorted_diag_with_indices)
    perm = [i + 1 for i in perm]
    
    perm_matrix = Permutation(perm).to_matrix()
    sorted_matrix = perm_matrix.T @ sorted_matrix @ perm_matrix
    
    # ----------------------------------------------------------------
    # 4) Build condensed type matrix from block intervals
    # ----------------------------------------------------------------
    intervals = get_intervals(sorted_diag)  # intervals of identical diagonal labels
    res = []
    for i_start, i_end in intervals:
        row_block = []
        for j_start, j_end in intervals:                    
            unique_elements = {
                sorted_matrix[row][col]
                for row in range(i_start, i_end + 1)
                for col in range(j_start, j_end + 1)
            }
            row_block.append(len(unique_elements))
        res.append(row_block)
    
    # ----------------------------------------------------------------
    # 5) Reorder the *type matrix* itself by its diagonal
    #    so that res[i][i] is in ascending order.
    # ----------------------------------------------------------------
    # res is an (m x m) matrix of block-to-block counts, where m = len(intervals).
    n = len(res)                       # number of blocks
    diag_tm = [res[i][i] for i in range(n)]
    
    # Sort the block indices by the diagonal values in ascending order
    perm_blocks = [idx for idx, val in sorted(enumerate(diag_tm), key=lambda x: x[1])]
    
    # Rebuild the type matrix rows/columns using perm_blocks
    reordered_res = []
    for i in perm_blocks:
        reordered_res.append([res[i][j] for j in perm_blocks])
    
    # Return the final type matrix, whose diagonal is now in ascending order
    return reordered_res
    


def visualise_matrix(sub_matrix):
    contiguous_matrix = np.ascontiguousarray(sub_matrix)
    
    # Convert to a Sage matrix
    adj_matrix = Matrix(contiguous_matrix)
    
    # Create the graph from the adjacency matrix
    G = Graph(adj_matrix)
    
    # Visualize the graph
    G.show()