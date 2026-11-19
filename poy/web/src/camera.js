/** Webcam access. The stream stays in this page; nothing is uploaded or recorded. */

/** Error with a message that can be shown to the player as-is. */
export class CameraError extends Error {}

/**
 * Turn a getUserMedia failure into a friendly message.
 * @param {unknown} err @returns {string}
 */
export function describeCameraError(err) {
  const name = /** @type {{ name?: string }} */ (err)?.name;
  if (name === "NotAllowedError" || name === "SecurityError")
    return "Camera permission was denied. Allow camera access in your browser's address bar, then press Start camera.";
  if (name === "NotFoundError" || name === "OverconstrainedError")
    return "No camera was found. Connect a webcam and press Start camera.";
  if (name === "NotReadableError")
    return "The camera is busy in another app. Close that app and press Start camera.";
  return `Could not start the camera (${name ?? String(err)}).`;
}

/**
 * Start the front camera and play it in `video`.
 * @param {HTMLVideoElement} video @returns {Promise<MediaStream>}
 */
export async function startCamera(video) {
  if (!navigator.mediaDevices?.getUserMedia)
    throw new CameraError(
      "This browser cannot use a camera here. Use a recent browser over https or localhost.",
    );
  let stream;
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: "user", width: { ideal: 640 }, height: { ideal: 480 } },
      audio: false,
    });
  } catch (err) {
    throw new CameraError(describeCameraError(err));
  }
  video.srcObject = stream;
  await video.play();
  return stream;
}

/** @param {MediaStream | null} stream */
export function stopCamera(stream) {
  stream?.getTracks().forEach((t) => t.stop());
}
