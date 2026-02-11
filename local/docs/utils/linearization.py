from __future__ import annotations

import math

import numpy as np


# =================================================================================================
#  ExponentialLinearization
# =================================================================================================
class UpperExponentialTransform:
    """
    A class that defines a transform f(y), such that provided (x,y) data are transformed into more linear (x,f(y)) data.

    We define f as follows:

        f(y) = c0 + c1*(log10(y_ref - y))

    With 3 parameters c0, c1, y_ref to be provided to the constructor.

    The underlying assumption is that we are dealing with (x,y) data that represent samples from the relationship:

        y = y_ref - exp(d0 - d1*x + noise)      (with d1>0)

    Meaning that y asymptotically approaches y_ref (from below) as x increases.
    """

    # -------------------------------------------------------------------------
    #  Constructor
    # -------------------------------------------------------------------------
    def __init__(self, c0, c1, y_ref):
        self.c0 = c0
        self.c1 = c1
        self.y_ref = y_ref

    # -------------------------------------------------------------------------
    #  Main API
    # -------------------------------------------------------------------------
    def f(self, y: float | np.ndarray | list[float]) -> float | np.ndarray:
        """
        Apply the transform to the given x-value(s).
        """
        if isinstance(y, list):
            y = np.array(y)

        return self.c0 + self.c1 * np.log10(self.y_ref - y)

    def f_inv(self, y_transformed: float | np.ndarray | list[float]) -> float | np.ndarray:
        """
        Get the inverse transform, which maps f(y) back to y.
        """
        if isinstance(y_transformed, list):
            y_transformed = np.array(y_transformed)

        return self.y_ref - 10 ** ((y_transformed - self.c0) / self.c1)

    def get_ticks_and_labels(self, n_max: int = 20, add_missing_ticks: bool = False) -> tuple[list[float], list[str]]:
        """
        Returns roughly equally spaced (on the transformed axis) ticks & labels.  Labels are formatted
        such as to only show the relevant precision.
        :param n_max: (int, default=20) maximum number of labels to return.
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
        #  if 1.414  -> 1.4146   -> insert 1.4141, 1.4142, 1.4143, 1.4144, 1.4145 with empty labels
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
                    if len(next_label) > len(curr_label) and "." in curr_label and "." in next_label:
                        # determine the step size based on the current label's precision
                        curr_decimals = len(curr_label.split(".")[1]) if "." in curr_label else 0
                        step = 10 ** -(curr_decimals + 1)

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
    def from_y_values(cls, y_values: list[float] | np.ndarray) -> UpperExponentialTransform:
        """
        Shorthand factory method similar to from_values, but only accepting y-values.
        y-values are simply assumed to correspond to a uniform x-range (not necessarily in order), and x-values
        are generated accordingly, after which from_values is called.
        """
        y_values = sorted(y_values)
        x_values = np.linspace(0.0, 1.0, len(y_values))
        return cls.from_values(x_values, y_values)

    @classmethod
    def from_values(
        cls, x: list[float] | np.ndarray, y: list[float] | np.ndarray, n: int = 100
    ) -> UpperExponentialTransform:
        """
        Factory method to create an UpperExponentialTransform instance from the given (x,y) data.

        This method computes the parameters c0, c1, y_ref based on the provided data, such that...
            - y_ref > max(y)
            - (x, f(y)) are as linear as possible
            - f(max(y)) = 1.0
            - f(min(y)) = 0.0

        :param x: list[float] | np.ndarray - the x-values of the data
        :param y: list[float] | np.ndarray - the y-values of the data
        :param n: int - the number of y_ref values to consider for fitting the transform (default: 100)
        """

        # --- prep ------------------------------
        x = np.sort(np.array(x))  # sorting helps eliminate some impact from noise
        y = np.sort(np.array(y))  # sorting helps eliminate some impact from noise
        y_min, y_max = np.min(y), np.max(y)

        if y_min == y_max:
            return UpperExponentialTransform(c0=0.5, c1=1.0, y_ref=y_max + 1.0)

        # --- y_ref candidates ------------------
        q_values = 1.0 - np.linspace(1 / n, 1.0, n) ** 2
        y_ref_candidates = sorted(
            {y_max + (y_max - np.quantile(y, q)) for q in q_values}
        )  # y_ref candidates are quantiles mirrored wrt y_max
        y_ref_candidates = [y_ref for y_ref in y_ref_candidates if y_ref > y_max]

        # --- fit y_ref -------------------------
        best_y_ref = max(y_ref_candidates)
        best_c0 = 0.0
        best_c1 = 1.0
        best_linearity = -math.inf

        for y_ref in y_ref_candidates:
            # determine c0, c1 for this y_ref
            # (such that in the end (y_min, y_max) are mapped to (0.0, 1.0))
            c1 = 1.0 / (np.log10(y_ref - y_max) - np.log10(y_ref - y_min))
            c0 = -c1 * np.log10(y_ref - y_min)

            # apply transform
            y_transform = c0 + c1 * np.log10(y_ref - y)

            # evaluate (y_transform should be evenly distributed in [0,1])
            q_values = np.linspace(0.0, 1.0, 11)
            y_values = np.quantile(y_transform, q_values)
            linearity = 1.0 - np.sum((q_values - y_values) ** 2)
            if linearity > best_linearity:
                best_linearity = linearity
                best_y_ref = y_ref
                best_c0 = c0
                best_c1 = c1

        # --- return transform instance ---------
        return UpperExponentialTransform(
            c0=float(best_c0),
            c1=float(best_c1),
            y_ref=float(best_y_ref),
        )


if __name__ == "__main__":
    y_values = [0, 0.5, 0.7, 0.79, 0.799, 0.7999, 0.79999, 0.79999, 0.799999, 0.7999999, 0.79999999, 0.799999999]

    transform = UpperExponentialTransform.from_y_values(y_values)

    print(f"y_min -> {transform.f(min(y_values)):.6f}")
    print(f"y_max -> {transform.f(max(y_values)):.6f}")

    print(f"-0.05 -> {transform.f_inv(-0.05):25.20f}")
    print(f" 0.00 -> {transform.f_inv(0.00):25.20f}")
    print(f" 1.00 -> {transform.f_inv(1.00):25.20f}")
    print(f" 1.05 -> {transform.f_inv(1.05):25.20f}")

    print(transform.f(y_values))
    print(transform.f_inv(np.linspace(0.0, 1.0, 21)))

    ticks, tick_labels = transform.get_ticks_and_labels(n_max=20)

    pass
