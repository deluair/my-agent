#!/usr/bin/env python3

import asyncio
import traceback
from my_agent.utils.config import load_config
from my_agent.agent.my_agent import MyAgent

async def debug_agent():
    try:
        # Load configuration
        config = load_config()
        print(f"Config loaded: {config}")
        
        # Create agent
        agent = MyAgent(config)
        print(f"Agent created: {agent}")
        
        # Set up trajectory recording
        trajectory_path = agent.setup_trajectory_recording()
        print(f"Trajectory recording set up: {trajectory_path}")
        
        # Set up the task
        task = "Create a text file named debug_test.txt with content 'Debug test successful!'"
        agent.new_task(
            task=task,
            extra_args={
                "project_path": "C:\\Users\\mhossen\\OneDrive - University of Tennessee\\AI\\my-agent\\my-agent",
                "issue": task
            }
        )
        print(f"Task set up: {task}")
        
        # Execute the task
        print("Starting task execution...")
        execution = await agent.execute_task()
        print(f"Task execution completed: {execution}")
        print(f"Success: {execution.success}")
        print(f"Final result: {execution.final_result}")
        print(f"Steps: {len(execution.steps)}")
        
        for i, step in enumerate(execution.steps):
            print(f"Step {i+1}: {step.state} - Error: {step.error}")
            if step.error:
                print(f"  Error details: {step.error}")
        
    except Exception as e:
        print(f"Exception occurred: {e}")
        print(f"Traceback: {traceback.format_exc()}")

if __name__ == "__main__":
    asyncio.run(debug_agent())