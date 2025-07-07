import os
import numpy as np
from dqnModel import dqnModel
import gymnasium as gym
import DQNAgent as DQNAgent
import time
import json

def run_visual_demo():
    """Run demo with visual simulation and save data for dashboard"""
    print("🚀 Starting Visual Lunar Lander Demo")
    print("🌐 Dashboard will update automatically!")
    print("=" * 50)
    
    # Clear old data file at start
    try:
        os.remove("demo_data.json")
        print("🗑️ Cleared previous demo data")
    except FileNotFoundError:
        pass  # No old data to clear
    
    # Create environment with visual rendering
    env = gym.make("LunarLander-v3", render_mode="human")
    agent = DQNAgent.DQNAgent(env=env, model_class=r"dqnModel_lunar_lander.pt")
    
    # Set to exploit mode (use trained knowledge)
    agent.mode = DQNAgent.DQNAGENT_MODE.EXPLOIT
    
    total_episodes = 10
    all_scores = []
    
    # Initialize data file
    demo_data = {
        "episodes": [],
        "scores": [],
        "timestamps": [],
        "status": "running",
        "current_episode": 0
    }
    
    for episode in range(total_episodes):
        print(f"\n🌙 Episode {episode + 1}/{total_episodes}")
        
        observation, info = env.reset()
        state = agent.state_processor(observation)
        
        episode_score = 0
        step_count = 0
        terminated = False
        truncated = False
        
        while not (terminated or truncated):
            # Get action from trained agent
            action = agent.policy_net(state).max(1).indices.view(1, 1).item()
            
            # Take action
            observation, reward, terminated, truncated, info = env.step(action)
            next_state = None if terminated else agent.state_processor(observation)
            
            episode_score += reward
            step_count += 1
            state = next_state
            
            # Small delay to make it watchable
            time.sleep(0.03)
        
        all_scores.append(episode_score)
        
        # Determine landing status
        if episode_score >= 200:
            status = "🎉 EXCELLENT LANDING!"
            landing_success = True
        elif episode_score >= 100:
            status = "✅ Good landing"
            landing_success = True
        elif episode_score >= 0:
            status = "⚠️ Rough landing"
            landing_success = False
        else:
            status = "💥 CRASHED!"
            landing_success = False
        
        print(f"Score: {episode_score:.1f}, Steps: {step_count}, Status: {status}")
        
        # Update data for dashboard
        demo_data["episodes"].append(episode + 1)
        demo_data["scores"].append(round(episode_score, 2))
        demo_data["timestamps"].append(time.time())
        demo_data["current_episode"] = episode + 1
        demo_data["avg_score"] = round(np.mean(all_scores), 2)
        demo_data["best_score"] = round(max(all_scores), 2)
        demo_data["success_rate"] = round((sum(1 for score in all_scores if score >= 100) / len(all_scores)) * 100, 1)
        demo_data["last_status"] = status
        demo_data["landing_success"] = landing_success
        
        # Save data to file for dashboard
        with open("demo_data.json", "w") as f:
            json.dump(demo_data, f, indent=2)
        
        time.sleep(1.5)  # Pause between episodes
    
    # Mark as complete
    demo_data["status"] = "complete"
    with open("demo_data.json", "w") as f:
        json.dump(demo_data, f, indent=2)
    
    # Final results
    print("\n" + "=" * 50)
    print("📊 FINAL RESULTS")
    print("=" * 50)
    print(f"Average Score: {demo_data['avg_score']}")
    print(f"Best Score: {demo_data['best_score']}")
    print(f"Success Rate: {demo_data['success_rate']}%")
    
    env.close()
    print("\n🎉 Demo complete! Check the dashboard for results.")

if __name__ == "__main__":
    run_visual_demo()