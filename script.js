// ====== CONFIG — EDIT THESE ======
const MODEL_PATH   = "best.onnx";
const CLASS_NAMES  = ["fresh", "expired"];   // ⚠️ match your model.names order!
const EXPIRED_CLASS = "expired";             // class that triggers alarm
const INPUT_SIZE   = 640;
const CONF_THRESH  = 0.4;
const IOU_THRESH   = 0.45;
// =================================

let session = null;
let running = false;
let rafId = null;

const video      = document.getElementById("video");
const canvas     = document.getElementById("canvas");
const ctx        = canvas.getContext("2d");
const statusText = document.getElementById("statusText");
const alertBox   = document.getElementById("alertBox");
const alarm      = document.getElementById("alarm");

// ---- Load model ----
(async () => {
  try {
    session = await ort.InferenceSession.create(MODEL_PATH, {
      executionProviders: ["wasm"]
    });
    statusText.textContent = "✅ Model loaded. Choose an input.";
  } catch (e) {
    statusText.textContent = "❌ Failed to load model: " + e;
  }
})();

// ---- Input handlers ----
document.getElementById("imageInput").onchange = e => {
  stopAll();
  const img = new Image();
  img.onload = async () => {
    sizeCanvas(img.width, img.height);
    ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
    const dets = await detect(img.width, img.height);
    drawDetections(dets);
  };
  img.src = URL.createObjectURL(e.target.files[0]);
};

document.getElementById("videoInput").onchange = e => {
  stopAll();
  video.src = URL.createObjectURL(e.target.files[0]);
  video.hidden = false;
  video.play();
  running = true;
  video.onloadeddata = () => { sizeCanvas(video.videoWidth, video.videoHeight); loop(); };
};

document.getElementById("webcamBtn").onclick = async () => {
  stopAll();
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ video: true });
    video.srcObject = stream;
    video.hidden = false;
    video.play();
    running = true;
    video.onloadeddata = () => { sizeCanvas(video.videoWidth, video.videoHeight); loop(); };
  } catch (err) {
    statusText.textContent = "❌ Webcam access denied.";
  }
};

document.getElementById("stopBtn").onclick = stopAll;

function stopAll() {
  running = false;
  if (rafId) cancelAnimationFrame(rafId);
  if (video.srcObject) video.srcObject.getTracks().forEach(t => t.stop());
  video.srcObject = null;
  stopAlarm();
}

function sizeCanvas(w, h) { canvas.width = w; canvas.height = h; }

// ---- Loop for video/webcam ----
async function loop() {
  if (!running) return;
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
  const dets = await detect(video.videoWidth, video.videoHeight);
  drawDetections(dets);
  rafId = requestAnimationFrame(loop);
}

// ---- Preprocess: letterbox to 640x640, normalize ----
function preprocess(srcW, srcH) {
  const tmp = document.createElement("canvas");
  tmp.width = INPUT_SIZE; tmp.height = INPUT_SIZE;
  const tctx = tmp.getContext("2d");
  tctx.fillStyle = "#727272";
  tctx.fillRect(0, 0, INPUT_SIZE, INPUT_SIZE);

  const r = Math.min(INPUT_SIZE / srcW, INPUT_SIZE / srcH);
  const nw = srcW * r, nh = srcH * r;
  const dx = (INPUT_SIZE - nw) / 2, dy = (INPUT_SIZE - nh) / 2;
  tctx.drawImage(canvas, dx, dy, nw, nh);

  const data = tctx.getImageData(0, 0, INPUT_SIZE, INPUT_SIZE).data;
  const arr = new Float32Array(INPUT_SIZE * INPUT_SIZE * 3);
  const area = INPUT_SIZE * INPUT_SIZE;
  for (let i = 0; i < area; i++) {
    arr[i]            = data[i * 4]     / 255; // R
    arr[i + area]     = data[i * 4 + 1] / 255; // G
    arr[i + area * 2] = data[i * 4 + 2] / 255; // B
  }
  return { tensor: new ort.Tensor("float32", arr, [1, 3, INPUT_SIZE, INPUT_SIZE]),
           r, dx, dy };
}

// ---- Run inference + postprocess ----
async function detect(srcW, srcH) {
  if (!session) return [];
  const { tensor, r, dx, dy } = preprocess(srcW, srcH);
  const out = await session.run({ [session.inputNames[0]]: tensor });
  const output = out[session.outputNames[0]];
  return postprocess(output, r, dx, dy);
}

// YOLOv8 output: [1, 4+numClasses, 8400]
function postprocess(output, r, dx, dy) {
  const data = output.data;
  const dims = output.dims;           // [1, ch, num]
  const ch = dims[1], num = dims[2];
  const nc = ch - 4;
  const boxes = [];

  for (let i = 0; i < num; i++) {
    let best = 0, bestId = 0;
    for (let c = 0; c < nc; c++) {
      const score = data[(4 + c) * num + i];
      if (score > best) { best = score; bestId = c; }
    }
    if (best < CONF_THRESH) continue;

    const cx = data[i], cy = data[num + i];
    const w  = data[2 * num + i], h = data[3 * num + i];
    let x1 = (cx - w / 2 - dx) / r;
    let y1 = (cy - h / 2 - dy) / r;
    let x2 = (cx + w / 2 - dx) / r;
    let y2 = (cy + h / 2 - dy) / r;
    boxes.push({ x1, y1, x2, y2, score: best, classId: bestId });
  }
  return nms(boxes);
}

function nms(boxes) {
  boxes.sort((a, b) => b.score - a.score);
  const keep = [];
  while (boxes.length) {
    const b = boxes.shift();
    keep.push(b);
    boxes = boxes.filter(o => iou(b, o) < IOU_THRESH);
  }
  return keep;
}

function iou(a, b) {
  const x1 = Math.max(a.x1, b.x1), y1 = Math.max(a.y1, b.y1);
  const x2 = Math.min(a.x2, b.x2), y2 = Math.min(a.y2, b.y2);
  const inter = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
  const areaA = (a.x2 - a.x1) * (a.y2 - a.y1);
  const areaB = (b.x2 - b.x1) * (b.y2 - b.y1);
  return inter / (areaA + areaB - inter);
}

// ---- Draw + alarm ----
function drawDetections(dets) {
  let expiredFound = false;
  ctx.lineWidth = 3;
  ctx.font = "18px Segoe UI";

  dets.forEach(d => {
    const label = CLASS_NAMES[d.classId] ?? d.classId;
    const isExpired = label === EXPIRED_CLASS;
    if (isExpired) expiredFound = true;

    ctx.strokeStyle = isExpired ? "#ef4444" : "#22c55e";
    ctx.fillStyle   = isExpired ? "#ef4444" : "#22c55e";
    ctx.strokeRect(d.x1, d.y1, d.x2 - d.x1, d.y2 - d.y1);

    const text = `${label} ${(d.score * 100).toFixed(0)}%`;
    const tw = ctx.measureText(text).width;
    ctx.fillRect(d.x1, d.y1 - 22, tw + 10, 22);
    ctx.fillStyle = "#000";
    ctx.fillText(text, d.x1 + 5, d.y1 - 5);
  });

  if (expiredFound) triggerAlarm(); else stopAlarm();
  statusText.textContent = `Detections: ${dets.length}`;
}

function triggerAlarm() {
  alertBox.classList.remove("hidden");
  if (alarm.paused) alarm.play().catch(() => {});
}
function stopAlarm() {
  alertBox.classList.add("hidden");
  alarm.pause(); alarm.currentTime = 0;
}
