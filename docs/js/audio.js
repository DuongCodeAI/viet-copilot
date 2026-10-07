// Một trình phát dùng chung: bấm nghe câu khác thì câu đang đọc dừng lại, như trợ lý chỉ nói một câu một lúc.
const player = new Audio();
let current = null;

function reset() {
  if (current) {
    current.classList.remove("on");
    current.textContent = current.dataset.label;
  }
  current = null;
}
player.addEventListener("ended", reset);

export function playButton(src, label = "Nghe") {
  const b = document.createElement("button");
  b.className = "play";
  b.type = "button";
  b.dataset.label = label;
  b.textContent = label;
  b.addEventListener("click", () => {
    const same = current === b;
    player.pause();
    reset();
    if (same) return;
    play(src, b);
  });
  return b;
}

export function play(src, button = null) {
  player.pause();
  reset();
  player.src = src;
  player.play().catch(() => {});
  if (button) {
    current = button;
    button.classList.add("on");
    button.textContent = "Dừng";
  }
}
