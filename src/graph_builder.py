import networkx as nx

def build_network_graph(risk_scores):
    """
    Builds a simulated network topology as a directed graph.
    Assigns real-world risk scores to the edges based on the destination port.
    """
    G = nx.DiGraph()
    
    # 1. Add nodes (Network Devices)
    G.add_nodes_from(["web_server", "workstation", "app_server", "database"])
    
    # 2. Add edges (Connections) with their port and calculated risk score
    G.add_edge("web_server", "app_server", port=8080, risk_score=risk_scores.get(8080, 0.5), allowed=True)
    G.add_edge("app_server", "database", port=3306, risk_score=risk_scores.get(3306, 0.9), allowed=True)
    G.add_edge("web_server", "workstation", port=22, risk_score=risk_scores.get(22, 0.8), allowed=True)
    G.add_edge("workstation", "database", port=3306, risk_score=risk_scores.get(3306, 0.9), allowed=True)
    
    return G
