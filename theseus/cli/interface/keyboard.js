document.addEventListener("focusin", (e) => {
  const screen = e.target.closest(".screen");
  if (screen) screen.dataset.lastFocus = e.target.id;
});
window.addEventListener("popstate", () =>
  emitEvent("locationchanged", window.location.search),
);
document.addEventListener("keydown", (e) => {
  if (e.ctrlKey || e.metaKey || e.altKey || e.isComposing) return;
  if (document.querySelector(".q-dialog")) return;
  const target = e.target;
  if (target.matches("input, textarea, select") || target.isContentEditable)
    return;
  const list = target.closest(".key-list");
  if (
    list &&
    !list.classList.contains("run-tree") &&
    ["ArrowUp", "ArrowDown", "Home", "End"].includes(e.key)
  ) {
    const items = [...list.querySelectorAll("button")];
    const index = items.indexOf(target.closest("button"));
    const next =
      e.key === "Home"
        ? 0
        : e.key === "End"
          ? items.length - 1
          : Math.max(
              0,
              Math.min(
                items.length - 1,
                index + (e.key === "ArrowUp" ? -1 : 1),
              ),
            );
    items[next]?.focus();
    e.preventDefault();
    return;
  }
  if (
    [
      "Escape",
      "q",
      "m",
      "u",
      "ArrowLeft",
      "ArrowRight",
      "ArrowUp",
      "ArrowDown",
      "Home",
      "End",
      "PageUp",
      "PageDown",
      "h",
      "l",
    ].includes(e.key)
  ) {
    e.preventDefault();
    emitEvent("terminalkey", { key: e.key, shiftKey: e.shiftKey });
  }
});
