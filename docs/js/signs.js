// Nhận diện biển báo trong trình duyệt: cùng các bước với dashcam/yolo_onnx.py + preprocess.py (numpy -> JS).
import { playButton } from "./audio.js";

const IMGSZ = 640, CONF = 0.15, IOU = 0.5;          // SignPipeline.load: detector conf 0.15, NMS 0.5
const CROP = 64, PAD = 0.15;                         // preprocess.py
const MEAN = [0.485, 0.456, 0.406].map((v) => v * 255);
const STD = [0.229, 0.224, 0.225].map((v) => v * 255);
const MIN_CONF = 0.6;                                // TrackVoter: dưới mức này không phát sự kiện
const MIN_DET = 0.25;

let det, cls, names, signs, samples;
let kind = "ô tô";
let last = null;                                     // kết quả ảnh đang hiển thị, để đổi loại xe không phải chạy lại

const $ = (id) => document.getElementById(id);

export async function initSigns() {
  const meta = $("sign-meta");
  [names, signs, samples] = await Promise.all(
    ["model/names.json", "data/signs.json", "data/samples.json"].map((u) => fetch(u).then((r) => r.json())));
  renderThumbs();
  ort.env.wasm.numThreads = 1;
  const t0 = performance.now();
  [det, cls] = await Promise.all([
    ort.InferenceSession.create("model/detector_1cls.int8.onnx"),
    ort.InferenceSession.create("model/signnet.int8.onnx"),
  ]);
  meta.textContent = `đã tải model (${((performance.now() - t0) / 1000).toFixed(1)} s)`;

  document.querySelectorAll("#signs .seg button").forEach((b) => b.addEventListener("click", () => {
    kind = b.dataset.kind;
    document.querySelectorAll("#signs .seg button").forEach((x) => x.setAttribute("aria-checked", x === b));
    if (last) renderWarnings(last);
  }));
  $("sign-file").addEventListener("change", async (e) => {
    const f = e.target.files[0];
    if (!f) return;
    document.querySelectorAll("#thumbs button").forEach((x) => x.setAttribute("aria-pressed", false));
    await run(URL.createObjectURL(f), null);
  });
  // ?anh=3 mở thẳng ảnh mẫu thứ 3 (để gửi link)
  const n = parseInt(new URLSearchParams(location.search).get("anh"), 10);
  await pick(n >= 1 && n <= samples.length ? n - 1 : 0);
}

function renderThumbs() {
  const box = $("thumbs");
  samples.forEach((s, i) => {
    const b = document.createElement("button");
    b.type = "button";
    b.setAttribute("aria-pressed", false);
    b.setAttribute("aria-label", `Ảnh mẫu ${i + 1}`);
    b.innerHTML = `<img src="${s.file}" alt="" loading="lazy">`;
    b.addEventListener("click", () => pick(i));
    box.appendChild(b);
  });
}

async function pick(i) {
  document.querySelectorAll("#thumbs button").forEach((x, j) => x.setAttribute("aria-pressed", i === j));
  await run(samples[i].file, samples[i]);
}

function loadImage(src) {
  return new Promise((ok, fail) => {
    const img = new Image();
    img.onload = () => ok(img);
    img.onerror = fail;
    img.src = src;
  });
}

async function run(src, sample) {
  const img = await loadImage(src);
  const W = img.naturalWidth, H = img.naturalHeight;

  // 1. letterbox về 640x640, nền xám 114 như ultralytics
  const r = Math.min(IMGSZ / H, IMGSZ / W);
  const nw = Math.round(W * r), nh = Math.round(H * r);
  const px = Math.floor((IMGSZ - nw) / 2), py = Math.floor((IMGSZ - nh) / 2);
  const lb = new OffscreenCanvas(IMGSZ, IMGSZ).getContext("2d", { willReadFrequently: true });
  lb.fillStyle = "rgb(114,114,114)";
  lb.fillRect(0, 0, IMGSZ, IMGSZ);
  lb.drawImage(img, px, py, nw, nh);
  const px32 = lb.getImageData(0, 0, IMGSZ, IMGSZ).data;
  const N = IMGSZ * IMGSZ, x = new Float32Array(3 * N);
  for (let i = 0; i < N; i++) {
    x[i] = px32[4 * i] / 255;
    x[N + i] = px32[4 * i + 1] / 255;
    x[2 * N + i] = px32[4 * i + 2] / 255;
  }
  const t0 = performance.now();
  const out = (await det.run({ [det.inputNames[0]]: new ort.Tensor("float32", x, [1, 3, IMGSZ, IMGSZ]) }))[det.outputNames[0]];
  const t1 = performance.now();

  // 2. decode (1, 4+1, A): cx, cy, w, h, điểm; YOLO11 không có objectness
  const A = out.dims[2], d = out.data;
  let boxes = [];
  for (let a = 0; a < A; a++) {
    const s = d[4 * A + a];
    if (s <= CONF) continue;
    const cx = d[a], cy = d[A + a], w = d[2 * A + a], h = d[3 * A + a];
    boxes.push({ b: [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], s });
  }
  boxes = nms(boxes).map(({ b, s }) => ({
    s,
    b: [clip((b[0] - px) / r, W), clip((b[1] - py) / r, H), clip((b[2] - px) / r, W), clip((b[3] - py) / r, H)],
  }));

  // 3. crop nới 15% mỗi phía -> 64x64 -> SignNet
  const crops = [], kept = [];
  const cc = new OffscreenCanvas(CROP, CROP).getContext("2d", { willReadFrequently: true });
  cc.imageSmoothingQuality = "high";
  for (const box of boxes) {
    const [x1, y1, x2, y2] = box.b;
    const bw = (x2 - x1) * (1 + 2 * PAD), bh = (y2 - y1) * (1 + 2 * PAD);
    const cx = (x1 + x2) / 2, cy = (y1 + y2) / 2;
    const a = Math.trunc(Math.max(cx - bw / 2, 0)), b = Math.trunc(Math.max(cy - bh / 2, 0));
    const c = Math.trunc(Math.min(cx + bw / 2, W)), e = Math.trunc(Math.min(cy + bh / 2, H));
    if (c - a < 4 || e - b < 4) continue;
    cc.drawImage(img, a, b, c - a, e - b, 0, 0, CROP, CROP);
    crops.push(cc.getImageData(0, 0, CROP, CROP).data);
    kept.push(box);
  }
  let t2 = t1;
  if (crops.length) {
    const M = CROP * CROP, xb = new Float32Array(crops.length * 3 * M);
    crops.forEach((p, k) => {
      for (let i = 0; i < M; i++)
        for (let ch = 0; ch < 3; ch++) xb[k * 3 * M + ch * M + i] = (p[4 * i + ch] - MEAN[ch]) / STD[ch];
    });
    const logits = (await cls.run({ [cls.inputNames[0]]: new ort.Tensor("float32", xb, [crops.length, 3, CROP, CROP]) }))[cls.outputNames[0]];
    t2 = performance.now();
    const C = logits.dims[1];
    kept.forEach((box, k) => {
      const row = logits.data.slice(k * C, (k + 1) * C);
      const m = Math.max(...row);
      const e = row.map((v) => Math.exp(v - m));
      const sum = e.reduce((p, q) => p + q, 0);
      let best = 0;
      for (let j = 1; j < C; j++) if (e[j] > e[best]) best = j;
      box.code = names[best];
      box.p = e[best] / sum;
    });
  }
  const found = kept.filter((k) => k.s >= MIN_DET);
  last = { img, found, sample, ms: [t1 - t0, t2 - t1] };
  draw(last);
  renderWarnings(last);
}

function clip(v, hi) { return Math.min(Math.max(v, 0), hi); }

function iou(a, b) {
  const w = Math.max(0, Math.min(a[2], b[2]) - Math.max(a[0], b[0]));
  const h = Math.max(0, Math.min(a[3], b[3]) - Math.max(a[1], b[1]));
  const i = w * h;
  return i / Math.max((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i, 1e-9);
}

function nms(boxes) {
  boxes.sort((p, q) => q.s - p.s);
  const keep = [];
  for (const bx of boxes) if (keep.every((k) => iou(k.b, bx.b) < IOU)) keep.push(bx);
  return keep;
}

const group = (code) => code[0];
function color(code) {
  const g = group(code);
  const css = getComputedStyle(document.documentElement);
  return g === "P" || g === "B" ? css.getPropertyValue("--stop") : g === "W" ? css.getPropertyValue("--amber") : css.getPropertyValue("--plate");
}

function draw({ img, found, ms }) {
  const cv = $("sign-canvas");
  cv.width = img.naturalWidth;
  cv.height = img.naturalHeight;
  const g = cv.getContext("2d");
  g.drawImage(img, 0, 0);
  const lw = Math.max(2, cv.width / 400), fs = Math.max(13, cv.width / 60);
  g.font = `600 ${fs}px "JetBrains Mono", monospace`;
  for (const k of found) {
    const [x1, y1, x2, y2] = k.b;
    const sure = k.p >= MIN_CONF;
    g.strokeStyle = sure ? color(k.code) : "#9aa5b0";
    g.lineWidth = lw;
    g.setLineDash(sure ? [] : [6, 4]);
    g.strokeRect(x1, y1, x2 - x1, y2 - y1);
    const label = k.code;
    const tw = g.measureText(label).width + 8;
    const ty = y1 - fs - 6 < 0 ? y2 + 2 : y1 - fs - 6;
    g.fillStyle = g.strokeStyle;
    g.fillRect(x1 - lw / 2, ty, tw, fs + 6);
    g.fillStyle = group(k.code) === "W" && sure ? "#1c2127" : "#fff";
    g.fillText(label, x1 + 4 - lw / 2, ty + fs);
  }
  g.setLineDash([]);
  $("sign-meta").textContent = `${found.length} biển · tìm biển ${ms[0].toFixed(0)} ms · phân loại ${ms[1].toFixed(0)} ms ` +
    `(onnxruntime-web, WASM 1 luồng, trên máy bạn)`;
}

function sayHtml(text) {
  // tô đậm mức phạt cho dễ thấy
  return text.replace(/(\d[\d,]* (?:triệu|nghìn) đến \d[\d,]* (?:triệu|nghìn) đồng)/, '<span class="fine">$1</span>');
}

function renderWarnings({ found, sample }) {
  const box = $("warns");
  box.innerHTML = "";
  const byCode = new Map();
  for (const k of [...found].sort((p, q) => q.p - p.p)) if (k.p >= MIN_CONF && !byCode.has(k.code)) byCode.set(k.code, k);
  const unsure = found.filter((k) => k.p < MIN_CONF);

  // trợ lý nói biển có mức phạt trước
  const order = [...byCode.values()].sort((p, q) => rank(q.code) - rank(p.code));
  if (!order.length) box.innerHTML = `<p class="muted">Không thấy biển nào đủ chắc chắn.</p>`;
  for (const k of order) box.appendChild(card(k));
  if (unsure.length) {
    const h = document.createElement("h3");
    h.textContent = "Không chắc, trợ lý không đọc";
    box.appendChild(h);
    for (const k of unsure) box.appendChild(card(k, true));
  }
  if (sample) box.appendChild(compare(sample, found));
}

function rank(code) {
  const w = info(code).warn[kind];
  return !w ? 0 : w.cite ? 2 : 1;
}

function info(code) {
  return signs.find((s) => s.code === code) || { code, name: code, warn: {} };
}

function card(k, unsure = false) {
  const s = info(k.code), w = s.warn[kind];
  const el = document.createElement("article");
  el.className = "warn";
  el.innerHTML = `<header><span class="code ${group(k.code)}">${k.code}</span><span class="name">${s.name}</span>` +
    `<span class="conf">${Math.round(k.p * 100)}%</span></header>`;
  if (unsure) return el;
  if (!w) {
    el.insertAdjacentHTML("beforeend", `<p class="silent">Trợ lý im lặng: biển chỉ dẫn, không có hành vi bị phạt riêng.</p>`);
    return el;
  }
  const p = document.createElement("p");
  p.className = "say";
  p.innerHTML = sayHtml(w.text) + " ";
  p.appendChild(playButton(w.audio));
  el.appendChild(p);
  if (w.cite) {
    el.insertAdjacentHTML("beforeend",
      `<details><summary>${w.cite}</summary><p>${w.law.replace(/</g, "&lt;")}${w.law.length >= 600 ? "…" : ""}</p></details>`);
  }
  return el;
}

function compare(sample, found) {
  // so với nhãn của VNTS: một biển nhãn coi là đúng nếu có box IoU >= 0.5 cùng mã
  const W = last.img.naturalWidth, H = last.img.naturalHeight;
  const used = new Set();
  let hit = 0, loose = 0;
  let left = sample.gt.map((g) => {
    const [cx, cy, w, h] = g.box;
    return { code: g.code, b: [(cx - w / 2) * W, (cy - h / 2) * H, (cx + w / 2) * W, (cy + h / 2) * H] };
  });
  // lượt 1 theo chuẩn mAP@0.5; lượt 2: đúng tên nhưng khung lệch (biển nhỏ ~25 px, lệch vài px là IoU < 0.5)
  for (const thr of [0.5, 0.3]) {
    left = left.filter((g) => {
      const m = found.findIndex((k, i) => !used.has(i) && k.code === g.code && k.p >= MIN_CONF && iou(k.b, g.b) >= thr);
      if (m < 0) return true;
      used.add(m);
      thr === 0.5 ? hit++ : loose++;
      return false;
    });
  }
  const miss = left.map((g) => g.code);
  const extra = found.filter((k, i) => !used.has(i) && k.p >= MIN_CONF).map((k) => k.code);
  const p = document.createElement("p");
  p.className = "gt " + (!miss.length && !extra.length ? "ok" : "bad");
  p.textContent = `So với nhãn VNTS: đúng ${hit}/${sample.gt.length} biển` +
    (loose ? ` · ${loose} biển đúng tên nhưng khung lệch (IoU < 0,5)` : "") +
    (miss.length ? ` · bỏ sót ${miss.join(", ")}` : "") + (extra.length ? ` · thừa ${extra.join(", ")}` : "");
  return p;
}
