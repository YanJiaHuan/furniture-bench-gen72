import torch
import numpy as np
import numpy.typing as npt

from furniture_bench.furniture.parts.part import Part
from furniture_bench.utils.pose import get_mat, is_similar_rot, rot_mat
import furniture_bench.utils.transform as T
import furniture_bench.controllers.control_utils as C


class Cube(Part):
    def __init__(self, part_config, part_idx):
        super().__init__(part_config, part_idx)
        tag_ids = part_config["ids"]

        # self.rel_pose_from_center[tag_ids[0]] = get_mat(
        #     [0, 0, -self.tag_offset], [0, 0, 0]
        # )
        # self.rel_pose_from_center[tag_ids[1]] = get_mat(
        #     [-self.tag_offset, 0, 0], [0, np.pi / 2, 0]
        # )
        # self.rel_pose_from_center[tag_ids[2]] = get_mat(
        #     [0, 0, self.tag_offset], [0, np.pi, 0]
        # )
        # self.rel_pose_from_center[tag_ids[3]] = get_mat(
        #     [self.tag_offset, 0, 0], [0, -np.pi / 2, 0]
        # )

        # 添加 reset_x_len、reset_y_len、reset_z_len 等属性
        self.reset_x_len = part_config.get("reset_x_len", 0)  # 这里的默认值可以根据实际情况调整
        self.reset_y_len = part_config.get("reset_y_len", 0)
        self.reset_z_len = part_config.get("reset_z_len", 0)


        self.done = False
        self.pos_error_threshold = 0.01
        self.ori_error_threshold = 0.25

        self.skill_complete_next_states = [
            "lift_up",
            "place",
        ]

        self.reset()

        self.part_attached_skill_idx = 4

    def reset(self):
        self.prev_pose = None
        self._state = "reach_cube_floor_xy"
        self.gripper_action = -1

    def fsm_step(
        self,
        ee_pos,
        ee_quat,
        gripper_width,
        rb_states,
        part_idxs,
        sim_to_april_mat,
        april_to_robot,
        place_to,
    ):
        def rot_mat_tensor(x, y, z, device):
            return torch.tensor(rot_mat([x, y, z], hom=True), device=device).float()

        def rel_rot_mat(s, t):
            s_inv = torch.linalg.inv(s)
            return t @ s_inv

        next_state = self._state

        ee_pose = C.to_homogeneous(ee_pos, C.quat2mat(ee_quat))
        place_pose = C.to_homogeneous(
            rb_states[part_idxs[place_to]][0][:3],
            C.quat2mat(rb_states[part_idxs[place_to]][0][3:7]),
        )
        cube_pose = C.to_homogeneous(
            rb_states[part_idxs[self.name]][0][:3],
            C.quat2mat(rb_states[part_idxs[self.name]][0][3:7]),
        )

        place_pose = sim_to_april_mat @ place_pose
        cube_pose = sim_to_april_mat @ cube_pose

        margin = rot_mat_tensor(0, -np.pi / 5, 0, ee_pose.device)
        device = ee_pose.device

        # Example state transitions (you can modify based on your needs):
        if self._state == "reach_cube_floor_xy":
            target_pos = cube_pose[:3, 3]
            target_ori = ee_pose[:3, :3]
            if self.satisfy(ee_pose, C.to_homogeneous(target_pos, target_ori)):
                self.prev_pose = cube_pose.clone()
                next_state = "pick_cube"
        elif self._state == "pick_cube":
            self.gripper_action = 1  # Gripper closes to pick the cube
            if self.gripper_less(gripper_width, 0.05):  # Adjust as necessary
                next_state = "lift_up"
        elif self._state == "lift_up":
            target_pos = self.prev_pose[:3, 3] + torch.tensor([0, 0, 0.1], device=device)
            if self.satisfy(ee_pose, C.to_homogeneous(target_pos, ee_pose[:3, :3])):
                next_state = "move_center"
        # Add more states as needed...

        skill_complete = self.may_transit_state(next_state)

        return (
            target_pos,
            C.mat2quat(target_ori),
            torch.tensor([self.gripper_action], device=device),
            skill_complete,
        )