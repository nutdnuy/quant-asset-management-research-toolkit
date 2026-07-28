from __future__ import annotations

from collections.abc import Hashable

import numpy as np

from qamr.contracts import LabeledMatrix, LabeledVector
from qamr.risk import CovarianceEstimate


def estimate(
    *,
    labels: tuple[Hashable, ...] = ("a", "b"),
    covariance: np.ndarray | None = None,
) -> CovarianceEstimate:
    resolved_covariance = (
        np.array([[0.04, 0.0], [0.0, 0.09]])
        if covariance is None
        else np.array(covariance, copy=True)
    )
    volatility = np.sqrt(np.diag(resolved_covariance))
    correlation = resolved_covariance / np.outer(volatility, volatility)
    return CovarianceEstimate(
        covariance=LabeledMatrix(
            resolved_covariance,
            labels,
            labels,
            "instrument",
            "instrument",
        ),
        correlation=LabeledMatrix(
            correlation,
            labels,
            labels,
            "instrument",
            "instrument",
        ),
        volatility=LabeledVector(volatility, labels, "instrument"),
        observation_count=24,
    )


def four_asset_estimate(order: np.ndarray) -> CovarianceEstimate:
    labels = ("a", "b", "c", "d")
    covariance = np.array(
        [
            [0.040, 0.006, 0.002, -0.001],
            [0.006, 0.090, 0.012, 0.003],
            [0.002, 0.012, 0.160, 0.020],
            [-0.001, 0.003, 0.020, 0.250],
        ]
    )
    volatility = np.sqrt(np.diag(covariance))
    correlation = covariance / np.outer(volatility, volatility)
    ordered_labels = tuple(labels[index] for index in order)
    return CovarianceEstimate(
        covariance=LabeledMatrix(
            covariance[np.ix_(order, order)],
            ordered_labels,
            ordered_labels,
            "instrument",
            "instrument",
        ),
        correlation=LabeledMatrix(
            correlation[np.ix_(order, order)],
            ordered_labels,
            ordered_labels,
            "instrument",
            "instrument",
        ),
        volatility=LabeledVector(
            volatility[order],
            ordered_labels,
            "instrument",
        ),
        observation_count=60,
    )
