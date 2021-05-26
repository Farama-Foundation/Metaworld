import numpy as np

from metaworld.envs.mujoco.sawyer_xyz.sawyer_xyz_state import SawyerXYZState

# The value returned by tolerance() at `margin` distance from `bounds` interval.
_DEFAULT_VALUE_AT_MARGIN = 0.1


def _sigmoids(x, value_at_1, sigmoid):
    """Returns 1 when `x` == 0, between 0 and 1 otherwise.

    Args:
        x: A scalar or numpy array.
        value_at_1: A float between 0 and 1 specifying the output when `x` == 1.
        sigmoid: String, choice of sigmoid type.

    Returns:
        A numpy array with values between 0.0 and 1.0.

    Raises:
        ValueError: If not 0 < `value_at_1` < 1, except for `linear`, `cosine` and
        `quadratic` sigmoids which allow `value_at_1` == 0.
        ValueError: If `sigmoid` is of an unknown type.
    """
    if sigmoid in ('cosine', 'linear', 'quadratic'):
        if not 0 <= value_at_1 < 1:
            raise ValueError(
                '`value_at_1` must be nonnegative and smaller than 1, '
                'got {}.'.format(value_at_1))
    else:
        if not 0 < value_at_1 < 1:
            raise ValueError('`value_at_1` must be strictly between 0 and 1, '
                             'got {}.'.format(value_at_1))

    if sigmoid == 'gaussian':
        scale = np.sqrt(-2 * np.log(value_at_1))
        return np.exp(-0.5 * (x * scale)**2)

    elif sigmoid == 'hyperbolic':
        scale = np.arccosh(1 / value_at_1)
        return 1 / np.cosh(x * scale)

    elif sigmoid == 'long_tail':
        scale = np.sqrt(1 / value_at_1 - 1)
        return 1 / ((x * scale)**2 + 1)

    elif sigmoid == 'reciprocal':
        scale = 1 / value_at_1 - 1
        return 1 / (abs(x) * scale + 1)

    elif sigmoid == 'cosine':
        scale = np.arccos(2 * value_at_1 - 1) / np.pi
        scaled_x = x * scale
        return np.where(
            abs(scaled_x) < 1, (1 + np.cos(np.pi * scaled_x)) / 2, 0.0)

    elif sigmoid == 'linear':
        scale = 1 - value_at_1
        scaled_x = x * scale
        return np.where(abs(scaled_x) < 1, 1 - scaled_x, 0.0)

    elif sigmoid == 'quadratic':
        scale = np.sqrt(1 - value_at_1)
        scaled_x = x * scale
        return np.where(abs(scaled_x) < 1, 1 - scaled_x**2, 0.0)

    elif sigmoid == 'tanh_squared':
        scale = np.arctanh(np.sqrt(1 - value_at_1))
        return 1 - np.tanh(x * scale)**2

    else:
        raise ValueError('Unknown sigmoid type {!r}.'.format(sigmoid))


def tolerance(x,
              bounds=(0.0, 0.0),
              margin=0.0,
              sigmoid='gaussian',
              value_at_margin=_DEFAULT_VALUE_AT_MARGIN):
    """Returns 1 when `x` falls inside the bounds, between 0 and 1 otherwise.

    Args:
        x: A scalar or numpy array.
        bounds: A tuple of floats specifying inclusive `(lower, upper)` bounds for
        the target interval. These can be infinite if the interval is unbounded
        at one or both ends, or they can be equal to one another if the target
        value is exact.
        margin: Float. Parameter that controls how steeply the output decreases as
        `x` moves out-of-bounds.
        * If `margin == 0` then the output will be 0 for all values of `x`
            outside of `bounds`.
        * If `margin > 0` then the output will decrease sigmoidally with
            increasing distance from the nearest bound.
        sigmoid: String, choice of sigmoid type. Valid values are: 'gaussian',
        'linear', 'hyperbolic', 'long_tail', 'cosine', 'tanh_squared'.
        value_at_margin: A float between 0 and 1 specifying the output value when
        the distance from `x` to the nearest bound is equal to `margin`. Ignored
        if `margin == 0`.

    Returns:
        A float or numpy array with values between 0.0 and 1.0.

    Raises:
        ValueError: If `bounds[0] > bounds[1]`.
        ValueError: If `margin` is negative.
    """
    lower, upper = bounds
    if lower > upper:
        raise ValueError('Lower bound must be <= upper bound.')
    if margin < 0:
        raise ValueError('`margin` must be non-negative. Current value: {}'.format(margin))

    in_bounds = np.logical_and(lower <= x, x <= upper)
    if margin == 0:
        value = np.where(in_bounds, 1.0, 0.0)
    else:
        d = np.where(x < lower, lower - x, x - upper) / margin
        value = np.where(in_bounds, 1.0, _sigmoids(d, value_at_margin,
                                                   sigmoid))

    return float(value) if np.isscalar(x) else value


def inverse_tolerance(x,
                      bounds=(0.0, 0.0),
                      margin=0.0,
                      sigmoid='reciprocal'):
    """Returns 0 when `x` falls inside the bounds, between 1 and 0 otherwise.

    Args:
        x: A scalar or numpy array.
        bounds: A tuple of floats specifying inclusive `(lower, upper)` bounds for
        the target interval. These can be infinite if the interval is unbounded
        at one or both ends, or they can be equal to one another if the target
        value is exact.
        margin: Float. Parameter that controls how steeply the output decreases as
        `x` moves out-of-bounds.
        * If `margin == 0` then the output will be 0 for all values of `x`
            outside of `bounds`.
        * If `margin > 0` then the output will decrease sigmoidally with
            increasing distance from the nearest bound.
        sigmoid: String, choice of sigmoid type. Valid values are: 'gaussian',
        'linear', 'hyperbolic', 'long_tail', 'cosine', 'tanh_squared'.
        value_at_margin: A float between 0 and 1 specifying the output value when
        the distance from `x` to the nearest bound is equal to `margin`. Ignored
        if `margin == 0`.

    Returns:
        A float or numpy array with values between 0.0 and 1.0.

    Raises:
        ValueError: If `bounds[0] > bounds[1]`.
        ValueError: If `margin` is negative.
    """
    bound = tolerance(x,
                      bounds=bounds,
                      margin=margin,
                      sigmoid=sigmoid,
                      value_at_margin=0)
    return 1 - bound


def rect_prism_tolerance(curr, zero, one):
    """Computes a reward if curr is inside a rectangular prism region.

    The 3d points curr and zero specify 2 diagonal corners of a rectangular
    prism that represents the decreasing region.

    one represents the corner of the prism that has a reward of 1.
    zero represents the diagonal opposite corner of the prism that has a reward
        of 0.
    Curr is the point that the prism reward region is being applied for.

    Args:
        curr(np.ndarray): The point who's reward is being assessed.
            shape is (3,).
        zero(np.ndarray): One corner of the rectangular prism, with reward 0.
            shape is (3,)
        one(np.ndarray): The diagonal opposite corner of one, with reward 1.
            shape is (3,)
    """
    in_range = lambda a, b, c: float(b <= a <=c) if c >= b else float(c <= a <= b)
    in_prism = (in_range(curr[0], zero[0], one[0]) and
                in_range(curr[1], zero[1], one[1]) and
                in_range(curr[2], zero[2], one[2]))
    if in_prism:
        diff = one - zero
        x_scale = (curr[0] - zero[0]) / diff[0]
        y_scale = (curr[1] - zero[1]) / diff[1]
        z_scale = (curr[2] - zero[2]) / diff[2]
        return x_scale * y_scale * z_scale
        # return 0.01
    else:
        return 1.


def hamacher_product(a, b):
    """The hamacher (t-norm) product of a and b.

    computes (a * b) / ((a + b) - (a * b))

    Args:
        a (float): 1st term of hamacher product.
        b (float): 2nd term of hamacher product.
    Raises:
        ValueError: a and b must range between 0 and 1

    Returns:
        float: The hammacher product of a and b
    """
    if not ((0. <= a <= 1.) and (0. <= b <= 1.)):
        raise ValueError("a and b must range between 0 and 1")

    denominator = a + b - (a * b)
    h_prod = ((a * b) / denominator) if denominator > 0 else 0

    assert 0. <= h_prod <= 1.
    return h_prod


def gripper_caging_reward(
        state: SawyerXYZState,
        initial_pos_obj: np.ndarray,
        initial_pos_pads_center: np.ndarray,
        obj_radius: float,
        pad_success_thresh: float,
        xz_thresh: float,
        desired_gripper_effort=1.0,
        include_reach_reward=False,
        reach_reward_radius=0.0):
    """Reward for grasping the main object. The main object's position should
    be at state.pos_objs[:3]

    Args:
        state: a state-based observation (NOT visual obs)
        obj_radius: radius of object's bounding sphere
        pad_success_thresh: successful distance of l/r pad to object
        xz_thresh: successful distance of gripper in x_z axis to the
            object. Y axis not included since the caging function handles
            successful grasping in the Y axis.
        object_reach_radius: successful distance of gripper center to the object

    """
    obj = state.pos_objs[:3]

    # MARK: Left-right gripper information for caging reward----------------
    # get current positions of left and right pads (Y axis)
    pad_y_lr = np.hstack((state.pos_pad_l[1], state.pos_pad_r[1]))
    # compare *current* pad positions with *current* obj position (Y axis)
    pad_to_obj_lr = np.abs(pad_y_lr - obj[1])
    # compare *current* pad positions with *initial* obj position (Y axis)
    pad_to_objinit_lr = np.abs(pad_y_lr - initial_pos_obj[1])

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
    caging_lr = [tolerance(
        pad_to_obj_lr[i],  # "x" in the description above
        bounds=(obj_radius, pad_success_thresh),
        margin=caging_lr_margin[i],  # "margin" in the description above
        sigmoid='long_tail',
    ) for i in range(2)]
    caging_y = hamacher_product(*caging_lr)

    # MARK: X-Z gripper information for caging reward-----------------------
    tcp = state.pos_pads_center
    xz = [0, 2]

    # Compared to the caging_y reward, caging_xz is simple. The margin is
    # constant (something in the 0.3 to 0.5 range) and x shrinks as the
    # gripper moves towards the object. After picking up the object, the
    # reward is maximized and changes very little
    caging_xz_margin = np.linalg.norm(
        initial_pos_obj[xz] -
        initial_pos_pads_center[xz]
    )
    caging_xz_margin -= xz_thresh
    caging_xz = tolerance(
        np.linalg.norm(tcp[xz] - obj[xz]),  # "x" in the description above
        bounds=(0, xz_thresh),
        margin=caging_xz_margin,  # "margin" in the description above
        sigmoid='long_tail',
    )

    # MARK: Closed-extent gripper information for caging reward-------------
    gripper_closed = min(
        max(0, state.action[-1]),
        desired_gripper_effort
    ) / desired_gripper_effort

    # MARK: Combine components----------------------------------------------
    caging = hamacher_product(caging_y, caging_xz)
    gripping = gripper_closed if caging > 0.97 else 0.
    caging_and_gripping = hamacher_product(caging, gripping)

    if not include_reach_reward:
        return (caging_and_gripping + caging) / 2
    else:
        assert reach_reward_radius > 0.0, 'When reach reward is enabled, reach_reward_radius must be > 0'

        tcp_to_obj = np.linalg.norm(obj - tcp)
        tcp_to_obj_init = np.linalg.norm(
            initial_pos_obj -
            initial_pos_pads_center
        )
        # Compute reach reward
        # - We subtract `object_reach_radius` from the margin so that the
        #   reward always starts with a value of 0.1
        reach_margin = abs(tcp_to_obj_init - reach_reward_radius)
        reach = tolerance(
            tcp_to_obj,
            bounds=(0, reach_reward_radius),
            margin=reach_margin,
            sigmoid='long_tail',
        )
        return (caging_and_gripping + reach) / 2
