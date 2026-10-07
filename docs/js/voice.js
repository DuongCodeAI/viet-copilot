// Lệnh giọng nói: hiển thị data/voice.json (output thật của scripts/build_web_demo.py trên laptop).
import { playButton } from "./audio.js";

const KIND = { call: "gọi tool", ask: "hỏi lại", refuse: "từ chối", reply: "trả lời" };
const STEPS = [["stt", "STT"], ["restore", "thêm dấu"], ["brain", "LLM"], ["tools", "tool"], ["tts", "TTS"]];
const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);

function callText(c) {
  const args = Object.entries(c.arguments).map(([k, v]) => `${k}=${JSON.stringify(v)}`).join(", ");
  return `${c.name}(${args})`;
}

function norm(s) {
  return s.toLowerCase().normalize("NFC").replace(/[.,!?]/g, "").replace(/\s+/g, " ").trim();
}

export async function initVoice() {
  const [turns, meta] = await Promise.all(["data/voice.json", "data/voice_meta.json"].map(
    (u) => fetch(u).then((r) => (r.ok ? r.json() : null))));
  const box = document.getElementById("turns");
  const maxTotal = Math.max(...turns.map((t) => Object.values(t.ms).reduce((a, b) => a + b, 0)));
  for (const t of turns) box.appendChild(turn(t, maxTotal));
  if (meta?.note) document.getElementById("voice-note").textContent = meta.note;
}

function turn(t, maxTotal) {
  const el = document.createElement("article");
  el.className = "turn";
  const spoken = !!t.audio_in;

  // cột 1: tài xế nói / gõ gì, trợ lý nghe được gì
  const c1 = document.createElement("div");
  c1.innerHTML = `<h4>${spoken ? "Tài xế nói" : "Tài xế gõ"}</h4>`;
  const p = document.createElement("p");
  p.textContent = `“${spoken ? t.said : t.input}” `;
  if (spoken) p.appendChild(playButton(t.audio_in));
  c1.appendChild(p);
  if (spoken) {
    const same = norm(t.said) === norm(t.input);
    c1.insertAdjacentHTML("beforeend", `<p class="heard${same ? "" : " diff"}">STT nghe được: <em>${esc(t.input)}</em></p>`);
  } else if (t.model_input && t.model_input !== t.input) {
    c1.insertAdjacentHTML("beforeend", `<p class="heard diff">thêm dấu: <em>${esc(t.model_input)}</em></p>`);
  }
  c1.insertAdjacentHTML("beforeend", `<div class="chips"><span class="chip">${t.speed} km/h</span>` +
    `<span class="chip">${t.night ? "ban đêm" : "ban ngày"}</span></div>`);

  // cột 2: model quyết định gì, guard có chặn không
  const a = t.action;
  const c2 = document.createElement("div");
  c2.innerHTML = `<h4>Qwen3-1.7B quyết định</h4><span class="kind ${a.kind}">${KIND[a.kind] || a.kind}</span>`;
  for (const c of a.calls) c2.insertAdjacentHTML("beforeend", `<div class="call">${esc(callText(c))}</div>`);
  if (a.model) {
    const wanted = a.model.calls.map(callText).join(", ");
    const rule = (a.notes.find((n) => n.startsWith("guard:")) || "").replace("guard: ", "");
    c2.insertAdjacentHTML("beforeend", `<div class="guard">Model gọi <code>${esc(wanted)}</code>, guard luật an toàn ` +
      `đổi thành ${KIND[a.kind]} (<code>${esc(rule)}</code>)</div>`);
  }

  // cột 3: trợ lý trả lời
  const c3 = document.createElement("div");
  c3.innerHTML = `<h4>Trợ lý nói</h4>`;
  const r = document.createElement("p");
  r.textContent = t.reply + " ";
  r.appendChild(playButton(t.audio_out));
  c3.appendChild(r);

  // latency từng bước
  const total = Object.values(t.ms).reduce((x, y) => x + y, 0);
  const lat = document.createElement("div");
  lat.className = "lat";
  const segs = STEPS.filter(([k]) => t.ms[k] != null)
    .map(([k]) => `<span class="c-${k}" style="width:${(t.ms[k] / maxTotal) * 100}%" title="${k} ${t.ms[k]} ms"></span>`).join("");
  const keys = STEPS.filter(([k]) => t.ms[k] != null && t.ms[k] >= 1)
    .map(([k, name]) => `<span class="k"><i class="c-${k}"></i>${name} ${fmt(t.ms[k])}</span>`).join(" ");
  lat.innerHTML = `<span>${fmt(total)}</span><div class="bar" aria-hidden="true">${segs}</div><span>${keys}</span>`;

  el.append(c1, c2, c3, lat);
  return el;
}

function fmt(ms) {
  return ms >= 1000 ? `${(ms / 1000).toFixed(1).replace(".", ",")} s` : `${Math.round(ms)} ms`;
}
