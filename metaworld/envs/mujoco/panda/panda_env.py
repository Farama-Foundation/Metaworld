import numpy as np
from gymnasium.spaces import Box

from metaworld.envs import reward_utils
from metaworld.envs.mujoco.arm_env import ArmEnv
from metaworld.envs.mujoco.mujoco_env import _assert_task_is_set


class PandaEnv(ArmEnv):
    _ACTION_DIM = 8
    _QPOS_SPACE = Box(
        np.array(
            [
                -2.9,
                -1.76,
                -2.9,
                -3.07,
                -2.9,
                -0.0175,
                -2.9,
                0,
                -0.04,
            ]
        ),
        np.array(
            [
                2.9,
                1.76,
                2.9,
                -0.0698,
                2.9,
                3.75,
                2.9,
                0.04,
                0,
            ]
        ),
        dtype=np.float64,
    )

    def __init__(
        self,
        model_name,
        hand_low=...,
        hand_high=...,
        mocap_low=None,
        mocap_high=None,
        render_mode=None,
    ):
        super().__init__(
            model_name=model_name,
            hand_low=hand_low,
            hand_high=hand_high,
            mocap_low=mocap_low,
            mocap_high=mocap_high,
            render_mode=render_mode,
        )

        self.hand_init_qpos = np.array(
            [0, np.pi / 16.0, 0.00, -np.pi / 2.0 - np.pi / 3.0, 0.00, np.pi - 0.2, np.pi / 4, 0.04, -0.04]
        )
        self.init_left_pad = self.left_pad
        self.init_right_pad = self.right_pad

        self.action_space = Box(
            np.array([-80, -80, -80, -80, -80, -12, -12, -0.04]),
            np.array([80, 80, 80, 80, 80, 12, 12, 0.04]),
            dtype=np.float64,
        )

        self.arm_col = [
            "link0_collision",
            "link1_collision",
            "link2_collision",
            "link3_collision",
            "link4_collision",
            "link5_collision",
            "link6_collision",
            "link7_collision",
        ]

        self.action_cost_coff = 1e-3

    @property
    def tcp_center(self):
        """The COM of the gripper's 2 fingers.

        Returns:
            (np.ndarray): 3-element position
        """
        left_finger_pos = self.data.body("finger_joint1_tip").xpos
        right_finger_pos = self.data.body("finger_joint1_tip").xpos
        tcp_center = (right_finger_pos + left_finger_pos) / 2.0
        return tcp_center
        # return self.data.site("grip_site").xpos

    @property
    def gripper_opened(self):
        return self.data.qpos[7] > 0.02 and self.data.qpos[8] < -0.02

    @property
    def left_pad(self):
        return self.get_body_com("finger_joint1_tip")

    @property
    def right_pad(self):
        return self.get_body_com("finger_joint2_tip")

    # @property
    # def gripper_distance_apart(self):
    #     finger_right, finger_left = (
    #         self.data.body("rightclaw"),
    #         self.data.body("leftclaw"),
    #     )
    #     # the gripper can be at maximum about ~0.1 m apart.
    #     # dividing by 0.1 normalized the gripper distance between
    #     # 0 and 1. Further, we clip because sometimes the grippers
    #     # are slightly more than 0.1m apart (~0.00045 m)
    #     # clipping removes the effects of this random extra distance
    #     # that is produced by mujoco

    #     gripper_distance_apart = np.linalg.norm(finger_right.xpos - finger_left.xpos)
    #     gripper_distance_apart = np.clip(gripper_distance_apart / 0.1, 0.0, 1.0)

    #     return gripper_distance_apart

    def set_action(self, action):
        """Applies the given action to the simulation.

        Args:
            action (np.ndarray): 8-element array of actions
        """

        parsed_action = np.hstack((action, -action[-1]))
        self.do_simulation(parsed_action, n_frames=self.frame_skip)

    def gripper_effort_from_action(self, action):
        # TODO
        # -0.01115~0.020833 -> 0~1
        margin = 0.020833 + 0.01115
        effort = (action[-1] + 0.01115) / margin
        return effort

    def get_action_penalty(self, action):
        action_norm = np.linalg.norm(action)
        contact = self.check_contact_table()

        penalty = self.action_cost_coff * action_norm
        if contact:
            penalty = 5

        return penalty

    @property
    def gripper_distance_apart(self):
        gripper_distance_apart = np.linalg.norm(self.left_pad - self.right_pad)
        # ic(self.left_pad, self.right_pad, gripper_distance_apart)
        gripper_distance_apart = (gripper_distance_apart - 0.0308) / (0.0775 - 0.0308)
        gripper_distance_apart = np.clip(gripper_distance_apart, 0.0, 1.0)
        return gripper_distance_apart

    def touching_object(self, object_geom_id):
        """Determines whether the gripper is touching the object with given id.

        Args:
            object_geom_id (int): the ID of the object in question

        Returns:
            (bool): whether the gripper is touching the object

        """

        leftpad_geom_id = self.data.geom("finger_joint1_tip").id
        rightpad_geom_id = self.data.geom("finger_joint2_tip").id

        leftpad_object_contacts = [
            x
            for x in self.unwrapped.data.contact
            if (
                leftpad_geom_id in (x.geom1, x.geom2)
                and object_geom_id in (x.geom1, x.geom2)
            )
        ]

        rightpad_object_contacts = [
            x
            for x in self.unwrapped.data.contact
            if (
                rightpad_geom_id in (x.geom1, x.geom2)
                and object_geom_id in (x.geom1, x.geom2)
            )
        ]

        leftpad_object_contact_force = sum(
            self.unwrapped.data.efc_force[x.efc_address]
            for x in leftpad_object_contacts
        )

        rightpad_object_contact_force = sum(
            self.unwrapped.data.efc_force[x.efc_address]
            for x in rightpad_object_contacts
        )

        return 0 < leftpad_object_contact_force and 0 < rightpad_object_contact_force

    def _gripper_caging_reward(  # TODO
        self,
        action,
        obj_pos,
        obj_radius,
        pad_success_thresh,
        object_reach_radius,
        xz_thresh,
        desired_gripper_effort=1.0,
        high_density=False,
        medium_density=False,
    ):
        """Reward for agent grasping obj.

        Args:
            action(np.ndarray): (4,) array representing the action
                delta(x), delta(y), delta(z), gripper_effort
            obj_pos(np.ndarray): (3,) array representing the obj x,y,z
            obj_radius(float):radius of object's bounding sphere
            pad_success_thresh(float): successful distance of gripper_pad
                to object
            object_reach_radius(float): successful distance of gripper center
                to the object.
            xz_thresh(float): successful distance of gripper in x_z axis to the
                object. Y axis not included since the caging function handles
                    successful grasping in the Y axis.
            desired_gripper_effort(float): desired gripper effort, defaults to 1.0.
            high_density(bool): flag for high-density. Cannot be used with medium-density.
            medium_density(bool): flag for medium-density. Cannot be used with high-density.
        """
        if high_density and medium_density:
            raise ValueError("Can only be either high_density or medium_density")
        # MARK: Left-right gripper information for caging reward----------------
        left_pad = self.left_pad
        right_pad = self.right_pad

        # get current positions of left and right pads (Y axis)
        pad_y_lr = np.hstack((left_pad[1], right_pad[1]))
        # compare *current* pad positions with *current* obj position (Y axis)
        pad_to_obj_lr = np.abs(pad_y_lr - obj_pos[1])
        # compare *current* pad positions with *initial* obj position (Y axis)
        pad_to_objinit_lr = np.abs(pad_y_lr - self.obj_init_pos[1])

        # Compute the left/right caging rewards. This is crucial for success,
        # yet counterintuitive mathematically because we invented it
        # accidentally.
        #
        # Before touching the object, `pad_to_obj_lr` ("x") is always separated
        # from `caging_lr_margin` ("the margin") by some small number,
        # `pad_success_thresh`.
        #
        # When far away from the object:
        #       x = margin + pad_success_thresh
        #       --> Thus x is outside the margin, yielding very small reward.
        #           Here, any variation in the reward is due to the fact that
        #           the margin itself is shifting.
        # When near the object (within pad_success_thresh):
        #       x = pad_success_thresh - margin
        #       --> Thus x is well within the margin. As long as x > obj_radius,
        #           it will also be within the bounds, yielding maximum reward.
        #           Here, any variation in the reward is due to the gripper
        #           moving *too close* to the object (i.e, blowing past the
        #           obj_radius bound).
        #
        # Therefore, before touching the object, this is very nearly a binary
        # reward -- if the gripper is between obj_radius and pad_success_thresh,
        # it gets maximum reward. Otherwise, the reward very quickly falls off.
        #
        # After grasping the object and moving it away from initial position,
        # x remains (mostly) constant while the margin grows considerably. This
        # penalizes the agent if it moves *back* toward `obj_init_pos`, but
        # offers no encouragement for leaving that position in the first place.
        # That part is left to the reward functions of individual environments.
        caging_lr_margin = np.abs(pad_to_objinit_lr - pad_success_thresh)
        caging_lr = [
            reward_utils.tolerance(
                pad_to_obj_lr[i],  # "x" in the description above
                bounds=(obj_radius, pad_success_thresh),
                margin=caging_lr_margin[i],  # "margin" in the description above
                sigmoid="long_tail",
            )
            for i in range(2)
        ]
        caging_y = reward_utils.hamacher_product(*caging_lr)
        # ic(
        #     pad_y_lr,
        #     obj_pos[1],
        #     pad_to_obj_lr,
        #     pad_to_objinit_lr,
        #     caging_lr_margin,
        #     caging_lr,
        #     caging_y,
        # )

        # MARK: X-Z gripper information for caging reward-----------------------
        tcp = self.tcp_center
        xz = [0, 2]

        # Compared to the caging_y reward, caging_xz is simple. The margin is
        # constant (something in the 0.3 to 0.5 range) and x shrinks as the
        # gripper moves towards the object. After picking up the object, the
        # reward is maximized and changes very little
        caging_xz_margin = np.linalg.norm(self.obj_init_pos[xz] - self.init_tcp[xz])
        caging_xz_margin -= xz_thresh
        caging_xz = reward_utils.tolerance(
            np.linalg.norm(tcp[xz] - obj_pos[xz]),  # "x" in the description above
            bounds=(0, xz_thresh),
            margin=caging_xz_margin,  # "margin" in the description above
            sigmoid="long_tail",
        )

        # MARK: Closed-extent gripper information for caging reward-------------
        gripper_closed = (
            min(max(0, action[-1]), desired_gripper_effort) / desired_gripper_effort
        )

        # MARK: Combine components----------------------------------------------
        caging = reward_utils.hamacher_product(caging_y, caging_xz)
        gripping = gripper_closed if caging > 0.97 else 0.0
        caging_and_gripping = reward_utils.hamacher_product(caging, gripping)

        if high_density:
            caging_and_gripping = (caging_and_gripping + caging) / 2
        if medium_density:
            tcp = self.tcp_center
            tcp_to_obj = np.linalg.norm(obj_pos - tcp)
            tcp_to_obj_init = np.linalg.norm(self.obj_init_pos - self.init_tcp)
            # Compute reach reward
            # - We subtract `object_reach_radius` from the margin so that the
            #   reward always starts with a value of 0.1
            reach_margin = abs(tcp_to_obj_init - object_reach_radius)
            reach = reward_utils.tolerance(
                tcp_to_obj,
                bounds=(0, object_reach_radius),
                margin=reach_margin,
                sigmoid="long_tail",
            )
            caging_and_gripping = (caging_and_gripping + reach) / 2

        return caging_and_gripping
