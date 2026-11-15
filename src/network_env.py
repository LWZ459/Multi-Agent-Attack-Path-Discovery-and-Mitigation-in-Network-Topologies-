import gymnasium as gym
from gymnasium import spaces
import numpy as np

class NetworkAttackEnv(gym.Env):
    def __init__(self, graph):
        super().__init__()
        self.graph = graph
        self.nodes = list(graph.nodes())
        self.edges = list(graph.edges())
        self.n_nodes = len(self.nodes)
        self.n_edges = len(self.edges)
        
        self.action_space = spaces.Discrete(self.n_edges)
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.n_nodes + self.n_edges,), dtype=np.float32)
        
        self.entry_node = "web_server"
        self.target_node = "database"
        self.max_steps = 10
        self.current_step = 0
        self.current_node = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_node = self.entry_node
        self.current_step = 0
        for u, v in self.edges:
            self.graph[u][v]['allowed'] = True
        # FIX: We must return the current node in the info dictionary
        return self._get_obs(), {"current_node": self.current_node}

    def _get_obs(self):
        node_obs = np.zeros(self.n_nodes, dtype=np.float32)
        node_obs[self.nodes.index(self.current_node)] = 1.0
        edge_obs = np.array([1.0 if self.graph[u][v]['allowed'] else 0.0 for u, v in self.edges], dtype=np.float32)
        return np.concatenate([node_obs, edge_obs])

    def step(self, action):
        self.current_step += 1
        src, dst = self.edges[action]
        
        if src != self.current_node or not self.graph[src][dst]['allowed']:
            reward = -5.0
            terminated = False
        else:
            self.current_node = dst
            risk = self.graph[src][dst].get('risk_score', 0.1)
            
            if dst == self.target_node:
                reward = 100.0 * risk
                terminated = True
            else:
                reward = -1.0
                terminated = False
                
        truncated = self.current_step >= self.max_steps
        return self._get_obs(), reward, terminated, truncated, {"current_node": self.current_node}
