# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

"""Trajectory recording for agent execution."""

import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional


class TrajectoryRecorder:
    """Records detailed execution trajectories for debugging and analysis."""
    
    def __init__(self, trajectory_path: Optional[str] = None):
        if trajectory_path is None:
            # Generate default trajectory filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            trajectory_path = f"trajectory_{timestamp}.json"
        
        self.trajectory_path = trajectory_path
        self.trajectory_data = {
            "metadata": {
                "created_at": datetime.now().isoformat(),
                "version": "1.0"
            },
            "task_info": {},
            "agent_steps": [],
            "llm_interactions": [],
            "final_result": None
        }
        self.step_counter = 0
    
    def get_trajectory_path(self) -> str:
        """Get the path where trajectory will be saved."""
        return self.trajectory_path
    
    def start_recording(self, task: str, provider: str, model: str, max_steps: int):
        """Start recording a new task execution."""
        self.trajectory_data["task_info"] = {
            "task": task,
            "provider": provider,
            "model": model,
            "max_steps": max_steps,
            "started_at": datetime.now().isoformat()
        }
    
    def record_agent_step(
        self,
        step_number: int,
        state: str,
        llm_messages: Optional[List[Any]] = None,
        llm_response: Optional[Any] = None,
        tool_calls: Optional[List[Any]] = None,
        tool_results: Optional[List[Any]] = None,
        reflection: Optional[str] = None,
        error: Optional[str] = None
    ):
        """Record a single agent step."""
        step_data = {
            "step_number": step_number,
            "state": state,
            "timestamp": datetime.now().isoformat()
        }
        
        if llm_messages:
            step_data["llm_messages"] = [msg.to_dict() if hasattr(msg, 'to_dict') else msg for msg in llm_messages]
        
        if llm_response:
            step_data["llm_response"] = llm_response.to_dict() if hasattr(llm_response, 'to_dict') else llm_response
        
        if tool_calls:
            step_data["tool_calls"] = [tc.to_dict() if hasattr(tc, 'to_dict') else tc for tc in tool_calls]
        
        if tool_results:
            step_data["tool_results"] = [tr.to_dict() if hasattr(tr, 'to_dict') else tr for tr in tool_results]
        
        if reflection:
            step_data["reflection"] = reflection
        
        if error:
            step_data["error"] = error
        
        self.trajectory_data["agent_steps"].append(step_data)
    
    def record_llm_request(self, provider: str, model: str, messages: List[Dict], tools: Optional[List[Dict]] = None):
        """Record an LLM request."""
        request_data = {
            "type": "request",
            "timestamp": datetime.now().isoformat(),
            "provider": provider,
            "model": model,
            "messages": messages
        }
        
        if tools:
            request_data["tools"] = tools
        
        self.trajectory_data["llm_interactions"].append(request_data)
    
    def record_llm_response(self, response: Dict):
        """Record an LLM response."""
        response_data = {
            "type": "response",
            "timestamp": datetime.now().isoformat(),
            "response": response
        }
        
        self.trajectory_data["llm_interactions"].append(response_data)
    
    def finalize_recording(self, success: bool, final_result: Optional[str] = None):
        """Finalize the recording and save to file."""
        self.trajectory_data["final_result"] = {
            "success": success,
            "result": final_result,
            "completed_at": datetime.now().isoformat()
        }
        
        # Save to file
        try:
            with open(self.trajectory_path, 'w', encoding='utf-8') as f:
                json.dump(self.trajectory_data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Warning: Could not save trajectory to {self.trajectory_path}: {e}")
    
    def get_trajectory_data(self) -> Dict[str, Any]:
        """Get the current trajectory data."""
        return self.trajectory_data.copy()