from __future__ import annotations

import math

import numpy as np

from local.docs.utils.axis_transforms._base import AxisTransform


# =================================================================================================
#  UpperLogTransform
# =================================================================================================
class UpperLogTransform(AxisTransform):
    """
    A class that defines a transform f(x) as an 'inverse logarithmic' transform (i.e. log(x_ref - x) instead of log(x)).

    The main aim is transform data to make them more uniformly distributed over their range, for data that
      asymptotically approaches x_ref from below.

    We define the transform as follows:

        f(x)      = c0 + c1*(log(x_ref - x))
        f_inv(x') = x_ref - exp((x' - c0) / c1)

    With 3 parameters c0, c1, x_ref to be provided to the constructor.

    A factory method is provided to optimally choose these parameters based on data, such that uniformity is maximized.
    """

    # -------------------------------------------------------------------------
    #  Constructor
    # -------------------------------------------------------------------------
    def __init__(self, c0, c1, x_ref):
        self.c0 = c0
        self.c1 = c1
        self.x_ref = x_ref

    # -------------------------------------------------------------------------
    #  Main API
    # -------------------------------------------------------------------------
    def f(self, x: float | np.ndarray | list[float]) -> float | np.ndarray:
        if isinstance(x, list | np.ndarray):
            return self.c0 + self.c1 * np.log(self.x_ref - np.array(x))
        else:
            return float(self.c0 + self.c1 * np.log(self.x_ref - x))

    def f_inv(self, x_transformed: float | np.ndarray | list[float]) -> float | np.ndarray:
        if isinstance(x_transformed, list | np.ndarray):
            return self.x_ref - np.exp((np.array(x_transformed) - self.c0) / self.c1)
        else:
            return float(self.x_ref - np.exp((x_transformed - self.c0) / self.c1))

    def get_ticks_and_labels(self, n_max: int = 20, add_missing_ticks: bool = False) -> tuple[list[float], list[str]]:
        """
        Returns roughly equally spaced (on the transformed axis) ticks & labels.  Labels are formatted
        such as to only show the relevant precision.
        :param n_max: (int, default=20) maximum number of labels to return.
        :param add_missing_ticks: (bool, default=False) whether to add missing ticks in case there's a jump in labels:
                                  e.g. 1.414 -> 1.4144,
                                       in which case 1.4141, 1.4142, 1.4143 would be added with empty labels
        :return: (ticks, tick_labels), which is a tuple of list[float] and list[str].
        """

        # --- determine ticks, tick_labels ----------------
        ticks = []
        tick_labels = []
        for n_max_internal in range(n_max, 10 * n_max):
            _ticks, _tick_labels = self._get_raw_ticks_and_labels(n_max_internal)
            if len(_ticks) == n_max:
                ticks, tick_labels = _ticks, _tick_labels
                break
            elif len(_ticks) <= n_max:
                ticks, tick_labels = _ticks, _tick_labels
            elif len(_ticks) >= 2 * n_max:
                break

        # --- post-processing -----------------------------
        # if there's a jump in length (# of chars) between 2 labels, make sure the logical ticks are inserted,
        # if they are missing.  E.g.:
        #
        #  if 1.414  -> 1.4144   -> insert 1.4141, 1.4142, 1.4143 with empty labels
        if add_missing_ticks and len(ticks) >= 2:
            new_ticks = []
            new_labels = []
            for i in range(len(tick_labels)):
                new_ticks.append(ticks[i])
                new_labels.append(tick_labels[i])

                if i < len(tick_labels) - 1:
                    curr_label = tick_labels[i]
                    next_label = tick_labels[i + 1]

                    # check if there's a jump in length (more digits in next label)
                    if len(next_label) > len(curr_label) and "." in next_label:
                        # determine the step size based on the current label's precision
                        next_decimals = len(next_label.split(".")[1])
                        step = 10**-next_decimals

                        # generate intermediate ticks from curr_tick + step up to (but not including) next_tick
                        intermediate_tick = ticks[i] + step
                        while intermediate_tick < ticks[i + 1] - step / 2:
                            new_ticks.append(intermediate_tick)
                            new_labels.append("")  # empty label for intermediate ticks
                            intermediate_tick += step

            ticks = new_ticks
            tick_labels = new_labels

        # --- return final result -------------------------
        return ticks, tick_labels

    # -------------------------------------------------------------------------
    #  Internal
    # -------------------------------------------------------------------------
    def _get_raw_ticks_and_labels(self, n_max: int) -> tuple[list[float], list[str]]:
        """
        Internal method similar to get_ticks_and_labels, but such that typically a significantly lower number
        of ticks & labels is returned.  The public method adjusts n_max to get the right number.

        Returns roughly equally spaced (on the transformed axis) ticks & labels.  Labels are formatted
        such as to only show the relevant precision.
        :param n_max: (int, default=20) maximum number of labels to return.
        :return: (ticks, tick_labels), which is a tuple of list[float] and list[str].
        """

        # get raw numbers
        n_max = max(2, n_max)
        y_trans_delta = 1.0 / (n_max - 1)
        y_trans_values = np.linspace(0.0, 1.0, n_max)
        y_values = self.f_inv(y_trans_values)

        # labels with minimal relevant precision
        y_labels: list[str] = []
        for y, y_trans in zip(y_values, y_trans_values):
            y_delta = self.f_inv(y_trans + 0.5 * y_trans_delta) - self.f_inv(y_trans - 0.5 * y_trans_delta)
            n_digits = max(1, -int(math.floor(math.log10(y_delta)))) - 1
            y_delta_rounded = 10**-n_digits
            y_rounded = int(y / y_delta_rounded) * y_delta_rounded
            y_label = f"{y_rounded:.{n_digits}f}"
            y_labels.append(y_label)

        # de-duplicate & re-sort
        y_labels = sorted(set(y_labels), key=lambda x: float(x))

        # remove labels for which f(y) is significantly outside [0,1] (can happen due to rounding)
        y_labels = [y_label for y_label in y_labels if -0.05 < self.f(float(y_label)) < 1.05]

        # remove spurious trailing 0s where appropriate
        def _remove_trailing_zeros(_label: str) -> str:
            if "." not in _label:
                # only remove trailing 0s if they're after a decimal point, otherwise they change the value
                return _label
            elif _label.endswith(".0"):
                return _label[:-2]
            elif _label.endswith("0"):
                return _label[:-1]
            else:
                return _label

        y_labels[0] = _remove_trailing_zeros(y_labels[0])
        for i in range(1, len(y_labels) - 1):
            if len(y_labels[i]) > len(y_labels[i - 1]):
                y_labels[i] = _remove_trailing_zeros(y_labels[i])

        # re-sync y_values to y_labels
        y_values = [float(y_label) for y_label in y_labels]  # sync back to y_values

        # return
        return y_values, y_labels

    # -------------------------------------------------------------------------
    #  Factory
    # -------------------------------------------------------------------------
    @classmethod
    def from_values(cls, x: list[float] | np.ndarray, n: int = 100) -> UpperLogTransform:
        """
        Factory method to create an UpperLogTransform instance from the given (x,y) data.

        This method computes the parameters c0, c1, x_ref based on the provided data, such that...
            - x_ref > max(x)
            - f(max(x)) = 1.0
            - f(min(x)) = 0.0
            - f(x) are as uniformly distributed over [0,1] as possible

        :param x: list[float] | np.ndarray - the data values to use
        :param n: int - the number of x_ref values to consider for fitting the transform (default: 100)
        """

        # --- prep ------------------------------
        x = np.sort(np.array(x))  # sorting helps eliminate some impact from noise
        x_min, x_max = np.min(x), np.max(x)

        if x_min == x_max:
            return UpperLogTransform(c0=0.5, c1=1.0, x_ref=x_max + 1.0)

        # --- x_ref candidates ------------------
        q_values = 1.0 - np.linspace(1 / n, 1.0, n) ** 2
        x_ref_candidates = sorted(
            {x_max + (x_max - np.quantile(x, q)) for q in q_values}
        )  # x_ref candidates are quantiles mirrored wrt y_max
        x_ref_candidates = [x_ref for x_ref in x_ref_candidates if x_ref > x_max]

        # --- fit x_ref -------------------------
        best_x_ref, best_c0, best_c1 = 0.0, 0.0, 0.0
        best_linearity = -math.inf

        for x_ref in x_ref_candidates:
            # determine c0, c1 for this x_ref
            # (such that in the end (y_min, y_max) are mapped to (0.0, 1.0))
            c1 = 1.0 / (np.log(x_ref - x_max) - np.log(x_ref - x_min))
            c0 = -c1 * np.log(x_ref - x_min)

            # apply transform
            x_transform = c0 + c1 * np.log(x_ref - x)

            # evaluate (y_transform should be evenly distributed in [0,1])
            q_values = np.linspace(0.0, 1.0, 101)
            y_values = np.quantile(x_transform, q_values)
            linearity = 1.0 - np.sum((q_values - y_values) ** 2)  # 1.0 - sum-squared-error
            if linearity > best_linearity:
                best_linearity = linearity
                best_x_ref = x_ref
                best_c0 = c0
                best_c1 = c1

        # --- return transform instance ---------
        return UpperLogTransform(
            c0=float(best_c0),
            c1=float(best_c1),
            x_ref=float(best_x_ref),
        )


if __name__ == "__main__":
    x_values = [0, 0.5, 0.7, 0.79, 0.799, 0.7999, 0.79999, 0.79999, 0.799999, 0.7999999, 0.79999999, 0.799999999]

    transform = UpperLogTransform.from_values(x_values)

    print(f"x_min -> {transform.f(min(x_values)):.6f}")
    print(f"x_max -> {transform.f(max(x_values)):.6f}")

    print(f"-0.05 -> {transform.f_inv(-0.05):25.20f}")
    print(f" 0.00 -> {transform.f_inv(0.00):25.20f}")
    print(f" 1.00 -> {transform.f_inv(1.00):25.20f}")
    print(f" 1.05 -> {transform.f_inv(1.05):25.20f}")

    print(transform.f(x_values))
    print(transform.f_inv(np.linspace(0.0, 1.0, 21)))

    ticks, tick_labels = transform.get_ticks_and_labels(n_max=20, add_missing_ticks=True)
    print(ticks)
    print(tick_labels)

    pass
