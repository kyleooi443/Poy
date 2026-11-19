import { expect, it } from "vitest";
import { describeCameraError } from "../src/camera.js";

it.each([
  ["NotAllowedError", /denied/],
  ["NotFoundError", /No camera/],
  ["NotReadableError", /busy/],
  ["Weird", /Weird/],
])("%s gets a clear message", (name, re) => expect(describeCameraError({ name })).toMatch(re));
