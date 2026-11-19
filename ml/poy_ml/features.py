"""The feature contract shared with web/src/features.js. Change both together (and FEATURE_SPEC_VERSION).

21 landmarks (x, y, z) -> 63 floats: subtract the wrist (landmark 0) from every point, then divide
all values by the largest wrist-to-point distance measured in x and y only.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

N_LANDMARKS = 21
N_FEATURES = N_LANDMARKS * 3

FEATURE_SPEC = {
    "name": "wrist_origin_max_xy_distance",
    "n_landmarks": N_LANDMARKS,
    "n_features": N_FEATURES,
    "layout": "x0,y0,z0,x1,y1,z1,...",
}


def landmarks_to_features(landmarks: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """(21, 3) landmarks in MediaPipe's normalized image coordinates -> (63,) features."""
    pts = np.asarray(landmarks, dtype=np.float64).reshape(N_LANDMARKS, 3)
    rel = pts - pts[0]
    scale = float(np.max(np.hypot(rel[:, 0], rel[:, 1])))
    if scale == 0.0:
        return np.zeros(N_FEATURES)
    return (rel / scale).reshape(N_FEATURES)


def mirror_features(features: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """Left/right mirror of feature rows (..., 63): negate every x. Equals mirroring the image."""
    out = np.array(features, dtype=np.float64)
    out[..., 0::3] *= -1
    return out
