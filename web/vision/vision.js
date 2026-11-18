// Camera + hand-seal recognition in the browser.
//
// Same two steps as server.py, run with ONNX Runtime Web instead of PyTorch:
//   1. hand.onnx (YOLO11n) finds the most confident hand in a mirrored 320x240 frame.
//   2. that box, padded by 20 px, is cropped and gesture.onnx (YOLO11l) names the seal.
//
// The Godot game reads window.shinobiVision.state each frame and calls
// start(), setCamRect() and debugGesture(). Nothing leaves the browser.

import * as ort from './ort/ort.webgpu.min.mjs';

const BASE = new URL('.', import.meta.url).href;
const FRAME_W = 320;
const FRAME_H = 240;
const HAND_SIZE = 640;      // hand model input: the frame scaled x2 -> 640x480, like ultralytics
const GESTURE_SIZE = 640;   // seal model input: square letterbox
const HAND_CONF = 0.35;
const CROP_PADDING = 20;
const MIN_INTERVAL_MS = 80; // at most ~12 checks a second, like server.py
// Class order of best_gesture.pt.
const NAMES = ['Chen', 'Chou', 'Hai', 'Mou', 'Shen', 'Si', 'Wei', 'Wu', 'Xu', 'Yin', 'You', 'Zi'];

const state = {
  phase: 'idle',      // idle | loading | camera | ready | error
  progress: 0,        // 0..1 while loading
  error: '',          // error code: no-camera | camera-denied | load-failed | no-webgl
  backend: '',
  seq: 0,             // bumps once per finished check
  gesture: '',
  conf: 0,
  hand: false,
  ms: 0,              // last check time
};

let video = null;
let camCanvas = null;
let camCtx = null;
let lastBox = null;
let handSession = null;
let gestureSession = null;
let running = false;
let startPromise = null;

const frame = makeCanvas(FRAME_W, FRAME_H);
const handIn = makeCanvas(HAND_SIZE, HAND_SIZE * FRAME_H / FRAME_W);
const crop = makeCanvas(1, 1);
const gestureIn = makeCanvas(GESTURE_SIZE, GESTURE_SIZE);

function makeCanvas(w, h) {
  const c = document.createElement('canvas');
  c.width = w;
  c.height = h;
  return { canvas: c, ctx: c.getContext('2d', { willReadFrequently: true }) };
}

// ---------- downloads ----------

async function fetchBytes(urls, onBytes) {
  const out = [];
  for (const url of urls) {
    const res = await fetch(url);
    if (!res.ok) throw new Error(`${url}: ${res.status}`);
    const reader = res.body.getReader();
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      out.push(value);
      onBytes(value.byteLength);
    }
  }
  const total = out.reduce((n, c) => n + c.byteLength, 0);
  const joined = new Uint8Array(total);
  let at = 0;
  for (const c of out) { joined.set(c, at); at += c.byteLength; }
  return joined;
}

async function gunzip(bytes) {
  const stream = new Blob([bytes]).stream().pipeThrough(new DecompressionStream('gzip'));
  return new Uint8Array(await new Response(stream).arrayBuffer());
}

async function loadModels() {
  const manifest = await (await fetch(BASE + 'models/manifest.json')).json();
  const total = manifest.totalBytes;
  let got = 0;
  const tick = (n) => { got += n; state.progress = Math.min(0.99, got / total); };

  const wasm = await gunzip(await fetchBytes([BASE + manifest.ortWasm], tick));
  ort.env.wasm.wasmPaths = BASE + 'ort/';
  ort.env.wasm.wasmBinary = wasm.buffer;
  ort.env.wasm.numThreads = self.crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 1) : 1;
  ort.env.logLevel = 'error';

  const hand = await fetchBytes(manifest.hand.map((f) => BASE + 'models/' + f), tick);
  const gesture = await fetchBytes(manifest.gesture.map((f) => BASE + 'models/' + f), tick);

  const tries = navigator.gpu ? ['webgpu', 'wasm'] : ['wasm'];
  let lastErr = null;
  for (const ep of tries) {
    try {
      const opts = { executionProviders: [ep], graphOptimizationLevel: 'all' };
      handSession = await ort.InferenceSession.create(hand, opts);
      gestureSession = await ort.InferenceSession.create(gesture, opts);
      state.backend = ep;
      break;
    } catch (err) {
      lastErr = err;
      console.warn('[vision] backend failed', ep, err);
    }
  }
  if (!gestureSession) throw lastErr;
  // Warm up so the first real check is not slow.
  await runHand();
}

// ---------- camera ----------

async function openCamera() {
  if (!navigator.mediaDevices?.getUserMedia) throw Object.assign(new Error('no camera api'), { code: 'no-camera' });
  let stream;
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: 'user' },
      audio: false,
    });
  } catch (err) {
    const denied = err && (err.name === 'NotAllowedError' || err.name === 'SecurityError');
    throw Object.assign(err, { code: denied ? 'camera-denied' : 'no-camera' });
  }
  video = document.createElement('video');
  video.playsInline = true;
  video.muted = true;
  video.srcObject = stream;
  await video.play();
}

function ensureCamCanvas() {
  if (camCanvas) return;
  camCanvas = document.createElement('canvas');
  camCanvas.id = 'shinobi-cam';
  camCanvas.width = 640;
  camCanvas.height = 480;
  Object.assign(camCanvas.style, {
    position: 'fixed', left: '0', top: '0', width: '0', height: '0',
    pointerEvents: 'none', zIndex: '5', display: 'none', borderRadius: '4px',
  });
  document.body.appendChild(camCanvas);
  camCtx = camCanvas.getContext('2d');
  requestAnimationFrame(drawCam);
}

// Draw the mirrored, 4:3 centre of the video into ctx at w x h.
function drawMirrored(ctx, w, h) {
  const vw = video.videoWidth;
  const vh = video.videoHeight;
  const target = 4 / 3;
  let sw = vw, sh = vh;
  if (vw / vh > target) sw = vh * target; else sh = vw / target;
  const sx = (vw - sw) / 2;
  const sy = (vh - sh) / 2;
  ctx.save();
  ctx.translate(w, 0);
  ctx.scale(-1, 1);
  ctx.drawImage(video, sx, sy, sw, sh, 0, 0, w, h);
  ctx.restore();
}

function drawCam() {
  requestAnimationFrame(drawCam);
  if (!video || camCanvas.style.display === 'none' || video.readyState < 2) return;
  drawMirrored(camCtx, camCanvas.width, camCanvas.height);
  if (lastBox) {
    const s = camCanvas.width / FRAME_W;
    camCtx.strokeStyle = '#00ff00';
    camCtx.lineWidth = 3;
    camCtx.strokeRect(lastBox[0] * s, lastBox[1] * s, (lastBox[2] - lastBox[0]) * s, (lastBox[3] - lastBox[1]) * s);
  }
}

// ---------- inference ----------

function toTensor({ ctx, canvas }) {
  const { width: w, height: h } = canvas;
  const px = ctx.getImageData(0, 0, w, h).data;
  const plane = w * h;
  const data = new Float32Array(3 * plane);
  for (let i = 0, p = 0; i < plane; i++, p += 4) {
    data[i] = px[p] / 255;
    data[i + plane] = px[p + 1] / 255;
    data[i + 2 * plane] = px[p + 2] / 255;
  }
  return new ort.Tensor('float32', data, [1, 3, h, w]);
}

// YOLO detect output is [1, 4 + classes, anchors]. Return the single best
// anchor (what server.py keeps: the highest confidence box).
function bestAnchor(out) {
  const [, rows, n] = out.dims;
  const d = out.data;
  let best = -1, bestCls = 0, bestScore = -1;
  for (let c = 4; c < rows; c++) {
    const off = c * n;
    for (let i = 0; i < n; i++) {
      if (d[off + i] > bestScore) { bestScore = d[off + i]; best = i; bestCls = c - 4; }
    }
  }
  return { cx: d[best], cy: d[n + best], w: d[2 * n + best], h: d[3 * n + best], cls: bestCls, score: bestScore };
}

async function runHand() {
  const { ctx, canvas } = handIn;
  ctx.drawImage(frame.canvas, 0, 0, canvas.width, canvas.height);
  const out = await handSession.run({ [handSession.inputNames[0]]: toTensor(handIn) });
  return bestAnchor(out[handSession.outputNames[0]]);
}

async function runGesture(x1, y1, x2, y2) {
  const cw = x2 - x1, ch = y2 - y1;
  crop.canvas.width = cw;
  crop.canvas.height = ch;
  crop.ctx.drawImage(frame.canvas, x1, y1, cw, ch, 0, 0, cw, ch);
  // server.py drew its green box on the frame before cropping, so the crop
  // the model was used with carries a thin green edge. Keep that.
  crop.ctx.strokeStyle = '#00ff00';
  crop.ctx.lineWidth = 2;
  crop.ctx.strokeRect(1, 1, cw - 2, ch - 2);

  const { ctx } = gestureIn;
  ctx.fillStyle = 'rgb(114,114,114)';
  ctx.fillRect(0, 0, GESTURE_SIZE, GESTURE_SIZE);
  const r = Math.min(GESTURE_SIZE / cw, GESTURE_SIZE / ch);
  const nw = Math.round(cw * r), nh = Math.round(ch * r);
  ctx.drawImage(crop.canvas, 0, 0, cw, ch, Math.floor((GESTURE_SIZE - nw) / 2), Math.floor((GESTURE_SIZE - nh) / 2), nw, nh);
  const out = await gestureSession.run({ [gestureSession.inputNames[0]]: toTensor(gestureIn) });
  return bestAnchor(out[gestureSession.outputNames[0]]);
}

const clamp = (v, lo, hi) => Math.max(lo, Math.min(v, hi));

async function check() {
  const t0 = performance.now();
  const hand = await runHand();
  let gesture = '', conf = 0, box = null;
  if (hand.score >= HAND_CONF) {
    const s = FRAME_W / HAND_SIZE;
    const x1 = Math.trunc((hand.cx - hand.w / 2) * s), y1 = Math.trunc((hand.cy - hand.h / 2) * s);
    const x2 = Math.trunc((hand.cx + hand.w / 2) * s), y2 = Math.trunc((hand.cy + hand.h / 2) * s);
    box = [
      clamp(x1 - CROP_PADDING, 0, FRAME_W - 1), clamp(y1 - CROP_PADDING, 0, FRAME_H - 1),
      clamp(x2 + CROP_PADDING, 0, FRAME_W - 1), clamp(y2 + CROP_PADDING, 0, FRAME_H - 1),
    ];
    if (box[2] - box[0] > 1 && box[3] - box[1] > 1) {
      const g = await runGesture(...box);
      gesture = NAMES[g.cls];
      conf = g.score;
    }
  }
  return { gesture, conf, box, ms: performance.now() - t0 };
}

function publish(r) {
  lastBox = r.box;
  state.gesture = r.gesture;
  state.conf = Math.round(r.conf * 1000) / 1000;
  state.hand = !!r.box;
  state.ms = Math.round(r.ms);
  state.seq++;
}

async function loop() {
  while (running) {
    const t0 = performance.now();
    if (video.readyState >= 2) {
      drawMirrored(frame.ctx, FRAME_W, FRAME_H);
      try {
        publish(await check());
      } catch (err) {
        console.error('[vision] check failed', err);
      }
    }
    const wait = MIN_INTERVAL_MS - (performance.now() - t0);
    await new Promise((r) => setTimeout(r, Math.max(wait, 0)));
  }
}

// ---------- public API ----------

async function start() {
  if (startPromise) return startPromise;
  startPromise = (async () => {
    try {
      state.phase = 'loading';
      state.error = '';
      const models = loadModels();
      models.catch(() => {});
      state.phase = 'camera';
      await openCamera();
      state.phase = 'loading';
      await models;
      ensureCamCanvas();
      state.progress = 1;
      state.phase = 'ready';
      running = true;
      loop();
    } catch (err) {
      console.error('[vision]', err);
      state.phase = 'error';
      state.error = err.code || 'load-failed';
      startPromise = null;
    }
  })();
  return startPromise;
}

// x, y, w, h are fractions of the game canvas.
function setCamRect(x, y, w, h, visible) {
  ensureCamCanvas();
  const gc = document.getElementById('canvas');
  const r = gc ? gc.getBoundingClientRect() : { left: 0, top: 0, width: innerWidth, height: innerHeight };
  Object.assign(camCanvas.style, {
    left: `${r.left + x * r.width}px`, top: `${r.top + y * r.height}px`,
    width: `${w * r.width}px`, height: `${h * r.height}px`,
    display: visible && video ? 'block' : 'none',
  });
}

// For testing without a camera: pretend the model saw `name`.
function debugGesture(name, conf = 0.9) {
  publish({ gesture: name, conf, box: null, ms: 0 });
}

// Run both models on an image (for bench.html). wholeFrame skips the hand
// step and reads the seal from the whole picture.
async function checkImage(img, wholeFrame = false) {
  if (!gestureSession) await loadModels();
  frame.ctx.drawImage(img, 0, 0, FRAME_W, FRAME_H);
  if (!wholeFrame) return check();
  const t0 = performance.now();
  const h = await runHand();
  const g = await runGesture(0, 0, FRAME_W - 1, FRAME_H - 1);
  return { hand: h.score, gesture: NAMES[g.cls], conf: g.score, ms: performance.now() - t0 };
}

window.shinobiVision = { state, start, setCamRect, debugGesture, checkImage, loadModels: () => loadModels() };
