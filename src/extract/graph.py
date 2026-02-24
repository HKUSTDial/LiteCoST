"""
This module provides functionality for creating and managing a semantic analysis graph.
"""

import networkx as nx
from pyvis.network import Network
from nltk.stem import WordNetLemmatizer
from collections import defaultdict
import nltk
from typing import List, Tuple, Optional

import nltk
nltk.download('wordnet')


class Graph:
    """
    A directed graph class for semantic analysis.
    """
    def __init__(self, description=""):
        """
        Initialize a directed graph with an optional description.

        Args:
        - description (str): Description of the graph.
        """
        self.graph = nx.DiGraph()
        self.lemmatizer = WordNetLemmatizer()
        self.node_map = {}  # Maps original nodes to their lemmatized versions
        self.edge_map = defaultdict(set)  # Stores edges to avoid duplicates
        self.description = description

    def add_node(self, node: str) -> None:
        """
        Add a node to the graph.

        Args:
        - node (str): The node to be added.
        """
        lemmatized_node = self.lemmatize_word(node)
        if lemmatized_node not in self.graph:
            self.graph.add_node(lemmatized_node)
        self.node_map[node] = lemmatized_node

    def add_edge(self, start_node: str, relation: str, end_node: str) -> None:
        """
        Add a directed edge between two nodes with a relation label.

        Args:
        - start_node (str): The starting node.
        - relation (str): The label for the relation.
        - end_node (str): The ending node.
        """
        lemmatized_start = self.lemmatize_word(start_node)
        lemmatized_end = self.lemmatize_word(end_node)
        lemmatized_relation = self.lemmatize_word(relation)

        if (lemmatized_start, lemmatized_end, lemmatized_relation) not in self.edge_map[lemmatized_start]:
            self.graph.add_edge(lemmatized_start, lemmatized_end, label=lemmatized_relation)
            self.edge_map[lemmatized_start].add((lemmatized_end, lemmatized_relation))

    def get_nodes(self) -> List[str]:
        """
        Get all nodes in the graph.

        Returns:
        - list: List of all nodes.
        """
        return list(self.graph.nodes)

    def get_edges(self) -> List[Tuple[str, str, str]]:
        """
        Get all edges in the graph with their labels.

        Returns:
        - list: List of edges with (start, end, label).
        """
        return [(u, v, d['label']) for u, v, d in self.graph.edges(data=True)]

    def get_neighbors(self, node: str) -> List[str]:
        """
        Get all neighboring nodes of a given node.

        Args:
        - node (str): The node for which neighbors are to be found.

        Returns:
        - list: List of neighboring nodes.
        """
        return list(self.graph.successors(self.lemmatize_word(node))) + list(self.graph.predecessors(self.lemmatize_word(node)))

    def create_graph_from_triplets(self, triplets: List) -> None:
        """
        Create a graph from a list of triples (head, relation, tail).

        Args:
        - triplets (list): List of triples in the form (head, relation, tail).
        """
        # self.merge_semantic_nodes(triplets)
        for triplet in triplets:
            if len(triplet) == 3:  # Only process valid triplets
                head, relation, tail = triplet
                self.add_edge(head, relation, tail)

    def delete_graph(self) -> None:
        """
        Clear the entire graph.
        """
        self.graph.clear()

    def delete_node(self, node: str) -> None:
        """
        Remove a node from the graph.

        Args:
        - node (str): The node to be removed.
        """
        lemmatized_node = self.lemmatize_word(node)
        if lemmatized_node in self.graph:
            self.graph.remove_node(lemmatized_node)
        else:
            print(f"Node {node} not found in graph")

    def delete_edge(self, start_node: str, end_node: str) -> None:
        """
        Remove an edge from the graph.

        Args:
        - start_node (str): The starting node of the edge.
        - end_node (str): The ending node of the edge.
        """
        lemmatized_start = self.lemmatize_word(start_node)
        lemmatized_end = self.lemmatize_word(end_node)
        if self.graph.has_edge(lemmatized_start, lemmatized_end):
            self.graph.remove_edge(lemmatized_start, lemmatized_end)
        else:
            print("Edge not found in graph")

    def search(self, node:str) -> bool:
        """
        Check if a node exists in the graph.

        Args:
        - node (str): The node to search for.

        Returns:
        - bool: True if the node exists, False otherwise.
        """
        return self.lemmatize_word(node) in self.graph

    def lemmatize_word(self, word: str) -> str:
        """
        Lemmatize a word to its base form using WordNet lemmatizer.

        Args:
        - word (str): The word to be lemmatized.

        Returns:
        - str: The lemmatized word.
        """
        return self.lemmatizer.lemmatize(word.lower())

    def merge_semantic_nodes(self, triplets: List) -> None:
        """
        Merge semantically similar nodes based on lemmatization.

        Args:
        - triplets (list): List of triples (head, relation, tail).
        """
        for head, _, tail in triplets:
            self.add_node(head)
            self.add_node(tail)

    def add_subgraph(self, subgraph: 'Graph') -> None:
        """
        Merge another graph into this graph.

        Args:
        - subgraph (Graph): Another graph to be merged.
        """
        for node in subgraph.get_nodes():
            self.add_node(node)

        for start_node, end_node, relation in subgraph.get_edges():
            self.add_edge(start_node, relation, end_node)

        self.description += f"\nMerged with subgraph: {subgraph.description}"
        self.node_map.update(subgraph.node_map)
        for node, edges in subgraph.edge_map.items():
            self.edge_map[node].update(edges)

    def delete_subgraph(self, subgraph: 'Graph') -> None:
        """
        Delete a subgraph from the current graph.

        Args:
        - subgraph (Graph): The subgraph to be removed.
        """
        for start_node, end_node, relation in subgraph.get_edges():
            if self.graph.has_edge(start_node, end_node):
                self.delete_edge(start_node, end_node)

        self.description = self.description.replace(f"\nMerged with subgraph: {subgraph.description}", "")

    def to_desc(self) -> None:
        """
        Return the textual description of the graph.

        Returns:
        - str: Graph description.
        """
        return self.description

    def nx_to_pyvis(self) -> Network:
        """
        Convert the graph into a PyVis network for visualization.

        Returns:
        - Network: A PyVis network object for visualization.
        """
        pyvis_graph = Network(notebook=True, cdn_resources='remote')
        for node in self.graph.nodes():
            pyvis_graph.add_node(node)
        for edge in self.graph.edges(data=True):
            pyvis_graph.add_edge(edge[0], edge[1], label=edge[2]["label"])
        return pyvis_graph

    # Generates a visual representation of the graph as an iframe
    def generateGraph(self) -> str:
        """
        Generate an interactive HTML representation of the graph using PyVis.

        Returns:
        - str: HTML string containing an iframe with the graph visualization.
        """

        pyvis_network = self.nx_to_pyvis()

        pyvis_network.toggle_hide_edges_on_drag(True)
        pyvis_network.toggle_physics(False)
        pyvis_network.set_edge_smooth('discrete')

        for edge in pyvis_network.edges:
            edge['arrows'] = 'to'
        html = pyvis_network.generate_html()
        html = html.replace("'", "\"")

        return f"""<iframe style="width: 100%; height: 600px;margin:0 auto" name="result" allow="midi; geolocation; microphone; camera;
        display-capture; encrypted-media;" sandbox="allow-modals allow-forms
        allow-scripts allow-same-origin allow-popups
        allow-top-navigation-by-user-activation allow-downloads" allowfullscreen=""
        allowpaymentrequest="" frameborder="0" srcdoc='{html}'></iframe>"""
    


if __name__ == "__main__":
    # Example usage
    # Create a graph object
    G = Graph()
    
    # Add triplets
    triplets = [
        ["A", "knows", "B"],
        ["B", "likes", "C"],
        ["C", "works_with", "D"]
    ]
    
    G.create_graph_from_triplets(triplets)
    
    print("Nodes:", G.get_nodes())
    print("Edges:", G.get_edges())
    
    # Get neighbors
    print("Neighbors of 'B':", G.get_neighbors("B"))

    # Delete an edge
    G.delete_edge("B", "C")
    print("Edges after deletion:", G.get_edges())
    
    # Delete a node
    G.delete_node("D")
    print("Nodes after deletion:", G.get_nodes())
    
    # Search for nodes
    print("Search for node 'A':", G.search("A"))
    print("Search for node 'D':", G.search("D"))

    # Generate graph visualization
    graph_html = G.generateGraph()
    with open("graph_test.html", "w") as file:
        file.write(graph_html)
    print("Graph visualization saved as graph_visualization.html")

    # Add a subgraph
    G2 = Graph("It's a subgraph of G")
    triplets2 = [
        ["A", "loves", "C"],
        ["D", "hates", "C"],
    ]
    
    G2.create_graph_from_triplets(triplets2)
    G.add_subgraph(G2)

    print("Nodes after adding subgraph:", G.get_nodes())
    print("Edges after adding subgraph:", G.get_edges())
    print("Description after adding subgraph:", G.to_desc())

    # Delete the subgraph
    G.delete_subgraph(G2)
    print("Nodes after deleting subgraph:", G.get_nodes())
    print("Edges after deleting subgraph:", G.get_edges())
    print("Description after deleting subgraph:", G.to_desc())