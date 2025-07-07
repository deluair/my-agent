import unittest
from unittest.mock import patch, MagicMock
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from my_agent.agent.my_agent import MyAgent
from my_agent.utils.config import Config, ModelParameters
from my_agent.utils.trajectory_recorder import TrajectoryRecorder
import subprocess

class TestTraeAgentExtended(unittest.TestCase):
    def setUp(self):
        # Create a proper config with ModelParameters
        test_model_params = ModelParameters(
            model="claude-3-5-sonnet-20241022",
            api_key="test-key",
            max_tokens=4096,
            temperature=0.5
        )
        self.config = Config(
            default_provider="anthropic",
            max_steps=20,
            enable_lakeview=True,
            model_providers={"anthropic": test_model_params}
        )
        self.agent = MyAgent(self.config)
        self.test_project_path = "/test/project"
        self.test_patch_path = "/test/patch.diff"

    @patch("my_agent.agent.my_agent.TrajectoryRecorder")
    def test_trajectory_setup(self, mock_recorder_class):
        # Set a task first so start_recording will be called
        self.agent.task = "test task"
        
        # Create a mock recorder instance
        mock_recorder_instance = MagicMock()
        mock_recorder_class.return_value = mock_recorder_instance
        mock_recorder_instance.get_trajectory_path.return_value = "/test/path"
        
        trajectory_path = self.agent.setup_trajectory_recording()
        
        # Verify the recorder was created and set
        mock_recorder_class.assert_called_once()
        self.assertIsNotNone(self.agent.trajectory_recorder)
        mock_recorder_instance.start_recording.assert_called_once_with(
            task="test task",
            provider="anthropic",
            model="claude-3-5-sonnet-20241022",
            max_steps=20
        )

    def test_new_task_initialization(self):
        with self.assertRaises(Exception):
            self.agent.new_task("test", {})  # Missing required params

        valid_args = {
            "project_path": self.test_project_path,
            "issue": "Test issue",
            "base_commit": "abc123",
            "must_patch": "true",
            "patch_path": self.test_patch_path
        }
        self.agent.new_task("test-task", valid_args)
        
        self.assertEqual(self.agent.project_path, self.test_project_path)
        self.assertEqual(self.agent.must_patch, "true")
        self.assertEqual(len(self.agent.tools), 4)
        self.assertTrue(any(tool.get_name() == "bash" for tool in self.agent.tools))

    @patch("os.path.isdir")
    @patch("os.chdir")
    @patch("os.getcwd")
    @patch("subprocess.check_output")
    def test_git_diff_generation(self, mock_subprocess, mock_getcwd, mock_chdir, mock_isdir):
        mock_subprocess.return_value = b"test diff"
        mock_getcwd.return_value = "/original/dir"
        mock_isdir.return_value = True
        self.agent.project_path = self.test_project_path
        
        diff = self.agent.get_git_diff()
        self.assertEqual(diff, "test diff")
        mock_subprocess.assert_called_with(['git', '--no-pager', 'diff'])

    def test_patch_filtering(self):
        test_patch = """diff --git a/tests/test_example.py b/tests/test_example.py
--- a/tests/test_example.py
+++ b/tests/test_example.py
@@ -5,6 +5,7 @@
     def test_example(self):
         assert True
"""
        filtered = self.agent.remove_patches_to_tests(test_patch)
        self.assertEqual(filtered, "")

    @patch("asyncio.create_task")
    @patch("my_agent.utils.cli_console.CLIConsole")
    def test_task_execution_flow(self, mock_console, mock_task):
        import asyncio
        
        # Mock the console instance
        mock_console_instance = mock_console.return_value
        mock_console_instance.start.return_value = asyncio.Future()
        mock_console_instance.start.return_value.set_result(None)
        
        self.agent.cli_console = mock_console_instance
        
        # Run the async method
        async def run_test():
            return await self.agent.execute_task()
        
        execution = asyncio.run(run_test())
        mock_console_instance.start.assert_called_once()

    def test_task_completion_detection(self):
        # Test empty patch scenario
        self.agent.must_patch = "true"
        self.assertFalse(self.agent.is_task_completed(MagicMock()))
        
        # Test valid patch scenario
        with patch.object(self.agent, 'get_git_diff', return_value="valid patch"):
            self.assertTrue(self.agent.is_task_completed(MagicMock()))

    def test_tool_initialization(self):
        tools = ["bash", "str_replace_based_edit_tool", "sequentialthinking", "task_done"]
        self.agent.new_task("test", {
            "project_path": self.test_project_path,
            "tool_names": tools
        })
        self.assertEqual(len(self.agent.tools), len(tools))
        self.assertEqual(self.agent.tools[0].get_name(), "bash")

if __name__ == '__main__':
    unittest.main()
