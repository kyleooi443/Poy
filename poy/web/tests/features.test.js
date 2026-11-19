import { describe, expect, it } from "vitest";
import { N_FEATURES, landmarksToFeatures, mirrorFeatures } from "../src/features.js";

const hand = (seed = 1) => {
  let s = seed;
  const r = () => (s = (s * 16807) % 2147483647) / 2147483647;
  return Array.from({ length: 21 }, () => ({ x: r(), y: r(), z: r() }));
};

describe("landmarksToFeatures", () => {
  it("puts the wrist at the origin and max xy distance at 1", () => {
    const f = landmarksToFeatures(hand());
    expect(f).toHaveLength(N_FEATURES);
    expect(f.slice(0, 3)).toEqual([0, 0, 0]);
    let max = 0;
    for (let i = 0; i < 63; i += 3) max = Math.max(max, Math.hypot(f[i], f[i + 1]));
    expect(max).toBeCloseTo(1, 12);
  });

  it("accepts [x,y,z] arrays and objects alike", () => {
    const h = hand(2);
    expect(landmarksToFeatures(h.map((p) => [p.x, p.y, p.z]))).toEqual(landmarksToFeatures(h));
  });

  it("is translation and scale invariant", () => {
    const h = hand(3);
    const moved = h.map((p) => ({ x: p.x * 0.5 + 0.2, y: p.y * 0.5 + 0.1, z: p.z * 0.5 }));
    landmarksToFeatures(moved).forEach((v, i) =>
      expect(v).toBeCloseTo(landmarksToFeatures(h)[i], 12),
    );
  });

  it("returns zeros for a degenerate hand and rejects wrong lengths", () => {
    expect(landmarksToFeatures(Array(21).fill([0.5, 0.5, 0]))).toEqual(Array(63).fill(0));
    expect(() => landmarksToFeatures(hand().slice(1))).toThrow();
  });
});

it("mirror negates x only and is its own inverse", () => {
  const f = landmarksToFeatures(hand(4));
  const m = mirrorFeatures(f);
  expect(m[3]).toBe(-f[3]);
  expect(m[4]).toBe(f[4]);
  expect(mirrorFeatures(m)).toEqual(f.map((v) => v + 0)); // +0 turns -0 into 0
});
