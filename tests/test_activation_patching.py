"""
Unit tests for the interventions module in Gdsc glassbox llms.
This suite verifies activation patching logic using mock objects to simulate 
transformer model behavior.

Implemented by Ankita Sharma (GitHub: sharmaankita3387)
Date: January 19, 2026
"""

import sys
import os
# Add the project root (parent directory) to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Now your existing imports will work with the Play button!
import unittest
from unittest.mock import MagicMock, patch
import torch
from glassboxllms.instrumentation.interventions import patch_activation

class TestPatchActivation(unittest.TestCase):
    """
    Test suite for the patch_activation function.
    Ensures that hooks are correctly attached, the forward pass is executed,
    and hooks are always cleaned up regardless of execution success.
    """

    def setUp(self):
        """
        Initialize common test fixtures.
        Sets up a mock model hierarchy: Model -> HookManager -> TargetModule.
        """
        self.mock_model = MagicMock()
        self.mock_target_module = MagicMock()
        self.mock_hook_id = "hook_ref_789"
        
        # Configure the mock model to return the expected sub-objects
        self.mock_model.get_layer_module.return_value = self.mock_target_module
        self.mock_model.hook_manager.add_hook.return_value = self.mock_hook_id
        
        # Test input data
        self.layer_name = "mlp.10"
        self.patch_tensor = torch.tensor([0.5, -0.5, 1.0])
        self.input_text = "Testing interventions logic."

    def test_successful_hook_lifecycle(self):
        """
        Verifying that the hook is added and removed correctly during a standard run.
        
        This test checks:
        1. Correct layer retrieval.
        2. Hook registration with the proper module.
        3. Invocation of the model's forward pass.
        4. Reliable removal of the hook post-execution.
        """
        # Execute the function under test
        output = patch_activation(
            model=self.mock_model, 
            layer=self.layer_name, 
            new_value=self.patch_tensor, 
            text=self.input_text
        )

        # Verify layer lookup
        self.mock_model.get_layer_module.assert_called_once_with(self.layer_name)

        # Verify hook registration
        self.mock_model.hook_manager.add_hook.assert_called_once()
        _, kwargs = self.mock_model.hook_manager.add_hook.call_args
        self.assertEqual(kwargs['module'], self.mock_target_module, "Hook must be attached to the target module.")

        # Verify forward pass call
        self.mock_model.forward.assert_called_once_with(self.input_text)

        # Verify hook removal
        self.mock_model.hook_manager.remove_hook.assert_called_once_with(self.mock_hook_id)

    def test_hook_cleanup_on_exception(self):
        """
        Ensure that the hook is removed even if the model's forward pass crashes.
        
        This is critical for preventing memory leaks and state contamination 
        in research environments where experiments run sequentially.
        """
        # Simulate a runtime error during the forward pass (e.g., OOM or Cuda error)
        self.mock_model.forward.side_effect = RuntimeError("Simulated GPU out of memory")

        # Verify the exception is raised but the cleanup logic executes
        with self.assertRaises(RuntimeError):
            patch_activation(self.mock_model, self.layer_name, self.patch_tensor, self.input_text)

        # The 'finally' block must ensure remove_hook is still called
        self.mock_model.hook_manager.remove_hook.assert_called_once_with(self.mock_hook_id)

if __name__ == '__main__':
    print("\n" + "="*50)
    print("STARTING GDSC-GlassboxLLMs INTERVENTION TESTS")
    print("="*50)
    
    suite = unittest.TestLoader().loadTestsFromTestCase(TestPatchActivation)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    if result.wasSuccessful():
        print("\n" + "-"*50)
        print("ALL TESTS PASSED SUCCESSFULLY")
        print("-"*50)
    else:
        print("\n" + "-"*50)
        print("TESTS FAILED")
        print("-"*50)