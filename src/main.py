from data_loader import get_port_risk_scores
from graph_builder import build_network_graph
from network_env import NetworkAttackEnv
from stable_baselines3 import DQN

def main():
    print("--- 1. Processing Data (CICIDS2017) ---")
    # If you have the dataset, replace 'None' with the actual file path (e.g., 'train.parquet')
    risk_scores = get_port_risk_scores(None) 
    
    print("\n--- 2. Building Network Graph ---")
    G = build_network_graph(risk_scores)
    print(f"Graph built: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges.")
    
    print("\n--- 3. Initializing Gymnasium Sandbox ---")
    env = NetworkAttackEnv(G)
    
    print("\n--- 4. Training Explorer Agent (DQN) ---")
    # Train a Deep Q-Network agent on the environment
    model = DQN("MlpPolicy", env, verbose=0, learning_starts=50)
    model.learn(total_timesteps=300) # 300 steps is enough for a basic proof-of-concept
    print("Training complete.")
    
    print("\n--- 5. Testing Agent (Attack Path Discovery) ---")
    obs, info = env.reset()
    done = False
    path = [info['current_node']]
    
    # Let the trained agent navigate the network
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        path.append(info['current_node'])
        done = terminated or truncated
        
    print(f"Discovered Attack Path: {' -> '.join(path)}")

if __name__ == "__main__":
    main()
