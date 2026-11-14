from data_loader import get_port_risk_scores
from graph_builder import build_network_graph
from network_env import NetworkAttackEnv
from random_agent import RandomAgent # Import the new baseline
from stable_baselines3 import DQN

def test_agent(agent, env, name="Agent"):
    obs, info = env.reset()
    done = False
    path = [info['current_node']]
    total_reward = 0
    
    while not done:
        action, _ = agent.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        path.append(info['current_node'])
        total_reward += reward
        done = terminated or truncated
        
    print(f"[{name}] Path: {' -> '.join(path)}")
    print(f"[{name}] Total Reward: {total_reward:.2f}\n")

def main():
    print("--- 1. Processing Data (CICIDS2017) ---")
    risk_scores = get_port_risk_scores(None) 
    
    print("\n--- 2. Building Network Graph ---")
    G = build_network_graph(risk_scores)
    
    print("\n--- 3. Initializing Sandbox ---")
    env = NetworkAttackEnv(G)
    
    print("\n--- 4. Testing Baseline (Random Agent) ---")
    random_agent = RandomAgent(env)
    # Run 3 times to show it randomly fails or gets stuck
    for i in range(1, 4):
        test_agent(random_agent, env, name=f"Random Agent Run {i}")
    
    print("\n--- 5. Training Explorer Agent (DQN) ---")
    model = DQN("MlpPolicy", env, verbose=0, learning_starts=50)
    model.learn(total_timesteps=300)
    print("Training complete.")
    
    print("\n--- 6. Testing Trained DQN Agent ---")
    test_agent(model, env, name="Trained DQN Agent")

if __name__ == "__main__":
    main()
