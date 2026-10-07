// Phát hiện buồn ngủ: DrowsinessMonitor của src/copilot/drowsiness.py viết lại bằng JS, cùng tham số.
import { play } from "./audio.js";

const MP = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.35";
const FPS = 10;                                   // cabin_source(fps=10)
const LEFT_EYE = [33, 160, 158, 133, 153, 144];   // p1..p6 của công thức EAR
const RIGHT_EYE = [362, 385, 387, 263, 373, 380];
const MOUTH = [78, 308, 13, 14];                  // khoé trái, khoé phải, môi trên, môi dưới
const LABEL = { calibrating: "đang hiệu chỉnh", ok: "tỉnh táo", warn: "nhắc nghỉ", alarm: "báo động", no_face: "không thấy mặt" };

const dist = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1]);
function ear(p) { return (dist(p[1], p[5]) + dist(p[2], p[4])) / (2 * dist(p[0], p[3]) + 1e-6); }
function mar(p) { return dist(p[2], p[3]) / (dist(p[0], p[1]) + 1e-6); }

export class Monitor {
  constructor(fps = FPS, { calibSec = 10, perclosWinSec = 60, closedRatio = 0.75, eyesClosedAlarmSec = 1.5,
    yawnMar = 0.6, yawnSec = 1.5, perclosWarn = 0.15, perclosAlarm = 0.3 } = {}) {
    Object.assign(this, { fps, closedRatio, yawnMar, perclosWarn, perclosAlarm });
    this.calibN = Math.trunc(calibSec * fps);
    this.calib = [];
    this.thr = null;
    this.winMax = Math.trunc(perclosWinSec * fps);
    this.window = [];
    this.closedRun = 0;
    this.yawnRun = 0;
    this.yawns = [];
    this.alarmFrames = Math.trunc(eyesClosedAlarmSec * fps);
    this.yawnFrames = Math.trunc(yawnSec * fps);
    this.frame = 0;
  }

  update(lm) {
    this.frame++;
    if (!lm) return { level: "no_face", reason: "không thấy mặt" };
    const e = (ear(LEFT_EYE.map((i) => lm[i])) + ear(RIGHT_EYE.map((i) => lm[i]))) / 2;
    const m = mar(MOUTH.map((i) => lm[i]));
    if (this.thr === null) {
      this.calib.push(e);
      if (this.calib.length < this.calibN) return { level: "calibrating", reason: "", ear: e, progress: this.calib.length / this.calibN };
      // trung vị: vài lần chớp mắt lúc hiệu chỉnh không kéo ngưỡng xuống
      const s = [...this.calib].sort((a, b) => a - b), n = s.length;
      this.thr = (n % 2 ? s[(n - 1) / 2] : (s[n / 2 - 1] + s[n / 2]) / 2) * this.closedRatio;
    }
    const closed = e < this.thr;
    this.window.push(closed);
    if (this.window.length > this.winMax) this.window.shift();
    this.closedRun = closed ? this.closedRun + 1 : 0;
    if (m > this.yawnMar) {
      this.yawnRun++;
      if (this.yawnRun === this.yawnFrames) { this.yawns.push(this.frame); if (this.yawns.length > 10) this.yawns.shift(); }
    } else this.yawnRun = 0;
    const perclos = this.window.filter(Boolean).length / this.window.length;
    const recentYawns = this.yawns.filter((f) => this.frame - f < 120 * this.fps).length;
    const base = { ear: e, perclos, yawns: recentYawns, closed: this.closedRun / this.fps };

    if (this.closedRun >= this.alarmFrames) return { ...base, level: "alarm", reason: `nhắm mắt ${(this.closedRun / this.fps).toFixed(1)} s` };
    const full = this.window.length >= Math.floor(this.winMax / 2);
    if (full && perclos >= this.perclosAlarm) return { ...base, level: "alarm", reason: `PERCLOS ${Math.round(perclos * 100)}%` };
    if ((full && perclos >= this.perclosWarn) || recentYawns >= 3) {
      const why = perclos >= this.perclosWarn ? `PERCLOS ${Math.round(perclos * 100)}%` : `ngáp ${recentYawns} lần / 2 phút`;
      return { ...base, level: "warn", reason: why };
    }
    return { ...base, level: "ok", reason: "" };
  }
}

const $ = (id) => document.getElementById(id);
let landmarker = null, mon = null, raf = 0, lastT = -1, history = [], prev = "ok", lastSaid = -1e9, msgs = null;

async function loadLandmarker() {
  if (landmarker) return landmarker;
  $("cam-status").textContent = "đang tải MediaPipe…";
  const { FaceLandmarker, FilesetResolver } = await import(`${MP}/vision_bundle.mjs`);
  const fileset = await FilesetResolver.forVisionTasks(`${MP}/wasm`);
  landmarker = await FaceLandmarker.createFromOptions(fileset, {
    baseOptions: { modelAssetPath: "model/face_landmarker.task" },
    runningMode: "VIDEO", numFaces: 1,
  });
  return landmarker;
}

export async function initDrowsy() {
  msgs = await fetch("data/drowsy.json").then((r) => r.json());
  $("cam-start").addEventListener("click", async () => {
    const btn = $("cam-start");
    btn.disabled = true;
    try {
      await loadLandmarker();
      const stream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480, facingMode: "user" }, audio: false });
      start(stream, null, true);
    } catch (e) {
      $("cam-status").textContent = e.name === "NotAllowedError"
        ? "Trình duyệt chưa cho dùng webcam. Bấm biểu tượng camera trên thanh địa chỉ để cho phép, hoặc chọn video."
        : `Không mở được webcam (${e.name || e}). Thử chọn video có mặt người.`;
      btn.disabled = false;
    }
  });
  document.querySelector(".cam").addEventListener("click", () => {
    const v = $("cam-video");
    if (!v.src || v.srcObject) return;  // chỉ video file mới tạm dừng được
    v.paused ? v.play() : v.pause();
    $("cam-status").textContent = "";
  });
  $("cam-file").addEventListener("change", async (e) => {
    const f = e.target.files[0];
    if (!f) return;
    await loadLandmarker();
    start(null, URL.createObjectURL(f), false);
  });
}

function start(stream, url, mirror) {
  const v = $("cam-video");
  if (v.srcObject) v.srcObject.getTracks().forEach((t) => t.stop());
  v.srcObject = stream;
  if (url) { v.src = url; v.loop = false; } else v.removeAttribute("src");
  v.play().catch(() => { $("cam-status").textContent = "Bấm vào khung hình để phát video."; });
  $("cam-empty").hidden = true;
  document.querySelector(".cam").classList.toggle("mirror", mirror);
  $("cam-start").textContent = stream ? "Webcam đang bật" : "Bật webcam";
  $("cam-start").disabled = !!stream;
  $("cam-status").textContent = "";
  mon = new Monitor();
  history = [];
  prev = "ok";
  lastSaid = -1e9;
  $("alert").hidden = true;
  cancelAnimationFrame(raf);
  lastT = -1;
  raf = requestAnimationFrame(loop);
}

function loop(now) {
  raf = requestAnimationFrame(loop);
  const v = $("cam-video");
  if (v.readyState < 2 || v.paused || v.ended) return;
  if (lastT >= 0 && now - lastT < 1000 / FPS) return;
  lastT = now;
  const t0 = performance.now();
  const res = landmarker.detectForVideo(v, now);
  const ms = performance.now() - t0;
  const W = v.videoWidth, H = v.videoHeight;
  const lm = res.faceLandmarks?.[0]?.map((p) => [p.x * W, p.y * H]) ?? null;
  const st = mon.update(lm);
  overlay(lm, W, H, st);
  meters(st, ms);
  history.push({ e: st.ear ?? null, thr: mon.thr });
  if (history.length > 20 * FPS) history.shift();
  chart();
  speak(st, now / 1000);
}

function overlay(lm, W, H, st) {
  const cv = $("cam-overlay");
  cv.width = W; cv.height = H;
  const g = cv.getContext("2d");
  if (!lm) return;
  const col = st.level === "alarm" ? "#ff5a4c" : st.level === "warn" ? "#ffc531" : "#4cc283";
  g.strokeStyle = col; g.fillStyle = col; g.lineWidth = 2;
  for (const eye of [LEFT_EYE, RIGHT_EYE]) {
    g.beginPath();
    eye.forEach((i, j) => (j ? g.lineTo(lm[i][0], lm[i][1]) : g.moveTo(lm[i][0], lm[i][1])));
    g.closePath(); g.stroke();
  }
  for (const i of MOUTH) { g.beginPath(); g.arc(lm[i][0], lm[i][1], 2.5, 0, 7); g.fill(); }
}

function meters(st, ms) {
  const lv = $("lv");
  lv.className = "level " + st.level;
  $("lv-text").textContent = st.level === "calibrating" ? `hiệu chỉnh ${Math.round((st.progress || 0) * 100)}%` : LABEL[st.level];
  $("m-ear").textContent = st.ear != null ? st.ear.toFixed(3) : "–";
  $("m-thr").textContent = mon.thr != null ? mon.thr.toFixed(3) : "đang đo";
  $("m-perclos").textContent = st.perclos != null ? `${Math.round(st.perclos * 100)}%` : "–";
  $("m-closed").textContent = st.closed != null ? `${st.closed.toFixed(1)} s` : "–";
  $("m-yawn").textContent = st.yawns != null ? st.yawns : "–";
  $("m-ms").textContent = `${ms.toFixed(0)} ms`;
}

function chart() {
  const cv = $("ear-chart");
  const w = cv.clientWidth, h = 120, dpr = devicePixelRatio || 1;
  cv.width = w * dpr; cv.height = h * dpr;
  const g = cv.getContext("2d");
  g.scale(dpr, dpr);
  const css = getComputedStyle(document.documentElement);
  const lo = 0, hi = 0.45, n = 20 * FPS;
  const X = (i) => (i / (n - 1)) * w, Y = (v) => h - 4 - ((v - lo) / (hi - lo)) * (h - 8);
  g.strokeStyle = css.getPropertyValue("--line"); g.lineWidth = 1;
  g.strokeRect(0.5, 0.5, w - 1, h - 1);
  const off = n - history.length;
  const thr = mon.thr;
  if (thr) {
    g.fillStyle = "rgba(210,43,31,.15)";
    history.forEach((p, i) => { if (p.e != null && p.e < thr) g.fillRect(X(i + off) - 1, 0, w / n + 1, h); });
    g.setLineDash([5, 4]); g.strokeStyle = css.getPropertyValue("--stop");
    g.beginPath(); g.moveTo(0, Y(thr)); g.lineTo(w, Y(thr)); g.stroke(); g.setLineDash([]);
  }
  g.strokeStyle = css.getPropertyValue("--plate"); g.lineWidth = 2;
  g.beginPath();
  let pen = false;
  history.forEach((p, i) => {
    if (p.e == null) { pen = false; return; }
    const x = X(i + off), y = Y(Math.min(p.e, hi));
    pen ? g.lineTo(x, y) : g.moveTo(x, y);
    pen = true;
  });
  g.stroke();
}

function speak(st, t) {
  // cabin_source chỉ phát sự kiện khi mức đổi sang warn/alarm; Copilot.on_drowsy cooldown 60 s (báo động 20 s)
  const changed = st.level !== prev;
  prev = st.level;
  if (!changed || (st.level !== "warn" && st.level !== "alarm")) return;
  if (t - lastSaid < (st.level === "alarm" ? 20 : 60)) return;
  lastSaid = t;
  const m = msgs[st.level];
  const box = $("alert");
  box.hidden = false;
  box.className = "alert " + st.level;
  $("alert-title").textContent = st.level === "alarm" ? `Báo động · ${st.reason}` : `Nhắc nghỉ · ${st.reason}`;
  $("alert-text").textContent = m.text;
  play(m.audio);
}
