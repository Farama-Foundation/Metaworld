"""A set of reward utilities written by the authors of dm_control."""

import numpy as np

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
    if sigmoid in ("cosine", "linear", "quadratic"):
        if not 0 <= value_at_1 < 1:
            raise ValueError(
                "`value_at_1` must be nonnegative and smaller than 1, "
                "got {}.".format(value_at_1)
            )
    else:
        if not 0 < value_at_1 < 1:
            raise ValueError(
                "`value_at_1` must be strictly between 0 and 1, "
                "got {}.".format(value_at_1)
            )

    if sigmoid == "gaussian":
        scale = np.sqrt(-2 * np.log(value_at_1))
        return np.exp(-0.5 * (x * scale) ** 2)

    elif sigmoid == "hyperbolic":
        scale = np.arccosh(1 / value_at_1)
        return 1 / np.cosh(x * scale)

    elif sigmoid == "long_tail":
        scale = np.sqrt(1 / value_at_1 - 1)
        return 1 / ((x * scale) ** 2 + 1)

    elif sigmoid == "reciprocal":
        scale = 1 / value_at_1 - 1
        return 1 / (abs(x) * scale + 1)

    elif sigmoid == "cosine":
        scale = np.arccos(2 * value_at_1 - 1) / np.pi
        scaled_x = x * scale
        return np.where(abs(scaled_x) < 1, (1 + np.cos(np.pi * scaled_x)) / 2, 0.0)

    elif sigmoid == "linear":
        scale = 1 - value_at_1
        scaled_x = x * scale
        return np.where(abs(scaled_x) < 1, 1 - scaled_x, 0.0)

    elif sigmoid == "quadratic":
        scale = np.sqrt(1 - value_at_1)
        scaled_x = x * scale
        return np.where(abs(scaled_x) < 1, 1 - scaled_x**2, 0.0)

    elif sigmoid == "tanh_squared":
        scale = np.arctanh(np.sqrt(1 - value_at_1))
        return 1 - np.tanh(x * scale) ** 2

    else:
        raise ValueError(f"Unknown sigmoid type {sigmoid!r}.")


def tolerance(
    x,
    bounds=(0.0, 0.0),
    margin=0.0,
    sigmoid="gaussian",
    value_at_margin=_DEFAULT_VALUE_AT_MARGIN,
):
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
        raise ValueError("Lower bound must be <= upper bound.")
    if margin < 0:
        raise ValueError(f"`margin` must be non-negative. Current value: {margin}")

    in_bounds = np.logical_and(lower <= x, x <= upper)
    if margin == 0:
        value = np.where(in_bounds, 1.0, 0.0)
    else:
        d = np.where(x < lower, lower - x, x - upper) / margin
        value = np.where(in_bounds, 1.0, _sigmoids(d, value_at_margin, sigmoid))

    return float(value) if np.isscalar(x) else value


def inverse_tolerance(x, bounds=(0.0, 0.0), margin=0.0, sigmoid="reciprocal"):
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
    bound = tolerance(
        x, bounds=bounds, margin=margin, sigmoid=sigmoid, value_at_margin=0
    )
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
        curr(np.ndarray): The point whose reward is being assessed.
            shape is (3,).
        zero(np.ndarray): One corner of the rectangular prism, with reward 0.
            shape is (3,)
        one(np.ndarray): The diagonal opposite corner of one, with reward 1.
            shape is (3,)
    """

    def in_range(a, b, c):
        return float(b <= a <= c) if c >= b else float(c <= a <= b)

    in_prism = (
        in_range(curr[0], zero[0], one[0])
        and in_range(curr[1], zero[1], one[1])
        and in_range(curr[2], zero[2], one[2])
    )
    if in_prism:
        diff = one - zero
        x_scale = (curr[0] - zero[0]) / diff[0]
        y_scale = (curr[1] - zero[1]) / diff[1]
        z_scale = (curr[2] - zero[2]) / diff[2]
        return x_scale * y_scale * z_scale
        # return 0.01
    else:
        return 1.0


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
    if not ((0.0 <= a <= 1.0) and (0.0 <= b <= 1.0)):
        raise ValueError("a and b must range between 0 and 1")

    denominator = a + b - (a * b)
    h_prod = ((a * b) / denominator) if denominator > 0 else 0

    assert 0.0 <= h_prod <= 1.0
    return h_prod
