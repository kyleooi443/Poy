/** MediaPipe Tasks Vision HandLandmarker, loaded from files served by this site (public/). */
import { FilesetResolver, HandLandmarker } from "@mediapipe/tasks-vision";

/** @typedef {import("./features.js").Landmark} Landmark */

/**
 * @param {string} base site base URL (import.meta.env.BASE_URL)
 * @returns {Promise<{ detect: (video: HTMLVideoElement, timeMs: number) => Landmark[] | null }>}
 */
export async function createHandTracker(base) {
  const modelUrl = `${base}models/hand_landmarker.task`;
  const head = await fetch(modelUrl, { method: "HEAD" }).catch(() => null);
  if (!head?.ok) throw new Error(`Hand model file missing: ${modelUrl}. Run "make setup".`);

  const vision = await FilesetResolver.forVisionTasks(`${base}wasm`);
  const options = {
    baseOptions: { modelAssetPath: modelUrl, delegate: /** @type {"GPU" | "CPU"} */ ("GPU") },
    runningMode: /** @type {"VIDEO"} */ ("VIDEO"),
    numHands: 1,
  };
  let landmarker;
  try {
    landmarker = await HandLandmarker.createFromOptions(vision, options);
  } catch {
    options.baseOptions.delegate = "CPU"; // some machines have no usable WebGL
    landmarker = await HandLandmarker.createFromOptions(vision, options);
  }
  let lastTime = -1;
  return {
    detect(video, timeMs) {
      if (video.readyState < 2) return null;
      const t = Math.max(timeMs, lastTime + 1); // timestamps must strictly increase
      lastTime = t;
      return landmarker.detectForVideo(video, t).landmarks[0] ?? null;
    },
  };
}
