import networkx as nx
import matplotlib.pyplot as plt

# Number of visible and hidden nodes
num_visible_nodes = 4
num_hidden_nodes = 3

# Create an empty graph
G = nx.Graph()

# Add visible nodes (labeled as V1, V2, ...)
for i in range(num_visible_nodes):
    G.add_node(f"V{i+1}", layer="visible")

# Add hidden nodes (labeled as H1, H2, ...)
for i in range(num_hidden_nodes):
    G.add_node(f"H{i+1}", layer="hidden")

# Connect visible nodes to hidden nodes
for visible_node in range(1, num_visible_nodes + 1):
    for hidden_node in range(1, num_hidden_nodes + 1):
        G.add_edge(f"V{visible_node}", f"H{hidden_node}")

# Position nodes for visualization
pos = nx.multipartite_layout(G, subset_key="layer")

# Draw and display the graph
nx.draw(G, pos, with_labels=True, node_color="skyblue", font_weight="bold", node_size=1000)
plt.show()
