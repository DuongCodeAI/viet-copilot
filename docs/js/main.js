import { initSigns } from "./signs.js";
import { initDrowsy } from "./drowsy.js";
import { initVoice } from "./voice.js";

const inits = { signs: initSigns, drowsy: initDrowsy, voice: initVoice };
const started = new Set();

function show(id) {
  for (const tab of document.querySelectorAll('[role="tab"]')) {
    const on = tab.getAttribute("aria-controls") === id;
    tab.setAttribute("aria-selected", on);
    tab.tabIndex = on ? 0 : -1;
    document.getElementById(tab.getAttribute("aria-controls")).hidden = !on;
  }
  if (!started.has(id)) {
    started.add(id);
    inits[id]().catch((e) => console.error(id, e));
  }
  history.replaceState(null, "", "#/" + id);  // "#/": không trùng id của section nên trang không tự cuộn
}

const tabs = [...document.querySelectorAll('[role="tab"]')];
tabs.forEach((tab, i) => {
  tab.addEventListener("click", () => show(tab.getAttribute("aria-controls")));
  tab.addEventListener("keydown", (e) => {
    const d = e.key === "ArrowRight" ? 1 : e.key === "ArrowLeft" ? -1 : 0;
    if (!d) return;
    const next = tabs[(i + d + tabs.length) % tabs.length];
    next.focus();
    show(next.getAttribute("aria-controls"));
  });
});
const want = location.hash.replace(/^#\/?/, "");
show(inits[want] ? want : "signs");
addEventListener("hashchange", () => {
  const id = location.hash.replace(/^#\/?/, "");
  if (inits[id]) show(id);
});
