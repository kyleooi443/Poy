/**
 * Feature contract shared with ml/poy_ml/features.py. Change both together.
 * 21 landmarks {x, y, z} -> 63 numbers: wrist (landmark 0) moved to the origin, everything
 * divided by the largest wrist-to-point distance measured in x and y only.
 */

/** @typedef {{ x: number, y: number, z: number }} Landmark */

export const N_LANDMARKS = 21;
export const N_FEATURES = N_LANDMARKS * 3;
export const FEATURE_SPEC_VERSION = 1;

/**
 * @param {ArrayLike<Landmark | number[]>} landmarks objects {x,y,z} (MediaPipe) or [x,y,z] arrays
 * @returns {number[]}
 */
export function landmarksToFeatures(landmarks) {
  if (landmarks.length !== N_LANDMARKS)
    throw new Error(`expected 21 landmarks, got ${landmarks.length}`);
  const pts = Array.from(landmarks, (p) => (Array.isArray(p) ? p : [p.x, p.y, p.z]));
  const [wx, wy, wz] = pts[0];
  const rel = pts.map(([x, y, z]) => [x - wx, y - wy, z - wz]);
  const scale = Math.max(...rel.map(([x, y]) => Math.hypot(x, y)));
  if (scale === 0) return new Array(N_FEATURES).fill(0);
  return rel.flat().map((v) => v / scale);
}

/** Left/right mirror: negate every x. @param {number[]} f @returns {number[]} */
export const mirrorFeatures = (f) => f.map((v, i) => (i % 3 === 0 ? -v : v));
