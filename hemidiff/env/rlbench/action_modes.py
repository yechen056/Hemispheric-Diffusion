import numpy as np
from rlbench.backend.scene import Scene
from rlbench.action_modes.action_mode import ActionMode
from rlbench.action_modes.arm_action_modes import BimanualJointPosition
from rlbench.action_modes.gripper_action_modes import (
    assert_action_shape, BimanualGripperJointPosition as BaseBimanualGripperJointPosition)

class BimanualGripperJointPositionCustom(BaseBimanualGripperJointPosition):
    def __init__(self, attach_grasped_objects: bool = True,
                 detach_before_open: bool = True,
                 absolute_mode: bool = True):
        super().__init__(attach_grasped_objects, detach_before_open, absolute_mode)
        self._absolute_mode = absolute_mode
        self._control_mode_set = False

    def action_shape(self, scene: Scene) -> tuple:
        return 4, 

    def action_bounds(self):
        return np.array([0.0]*4), np.array([0.04]*4)

    def action_pre_step(self, scene: Scene, action: np.ndarray):
        if not self._control_mode_set:
            scene.robot.right_gripper.set_control_loop_enabled(True)
            scene.robot.left_gripper.set_control_loop_enabled(True)
            self._control_mode_set = True
        
        assert_action_shape(action, self.action_shape(scene))
        
        right_action = action[:2]
        left_action = action[2:]

        if not self._absolute_mode:
            right_action = right_action + np.array(scene.robot.right_gripper.get_joint_positions())
            left_action = left_action + np.array(scene.robot.left_gripper.get_joint_positions())
            
        scene.robot.right_gripper.set_joint_target_positions(right_action)
        scene.robot.left_gripper.set_joint_target_positions(left_action)

    def action_post_step(self, scene: Scene, action: np.ndarray):
        scene.robot.right_gripper.set_joint_target_positions(
            scene.robot.right_gripper.get_joint_positions())
        scene.robot.left_gripper.set_joint_target_positions(
            scene.robot.left_gripper.get_joint_positions())


class BimanualJointPositionActionMode(ActionMode):
    def __init__(self):
        super().__init__(
            BimanualJointPosition(absolute_mode=True),
            BimanualGripperJointPositionCustom(absolute_mode=True)
        )

    def action(self, scene: Scene, action: np.ndarray):
        
        right_arm = action[0:7]
        right_gripper = action[7:9]
        left_arm = action[9:16]
        left_gripper = action[16:18]
        
        arm_action = np.concatenate([right_arm, left_arm])
        ee_action = np.concatenate([right_gripper, left_gripper])
        
        self.arm_action_mode.action_pre_step(scene, arm_action)
        self.gripper_action_mode.action_pre_step(scene, ee_action)
        scene.step()
        self.arm_action_mode.action_post_step(scene, arm_action)
        self.gripper_action_mode.action_post_step(scene, ee_action)

    def action_shape(self, scene: Scene):
        return (18,) 
        
    def action_bounds(self):
        return None, None
