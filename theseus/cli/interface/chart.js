/* Plotly owns rendering and axes; this overlay preserves the terminal gestures. */
window.theseusChart = {
  resize(plot) {
    if (!plot || plot.theseusResizeObserver) return;
    // Plotly.Plots.resize queues work which rejects after a screen is hidden.
    // Set dimensions directly, only while the retained chart is displayed.
    plot.theseusResizeObserver = new ResizeObserver(([entry]) => {
      if (!plot.isConnected) {
        plot.theseusResizeObserver.disconnect();
        return;
      }
      const width = Math.round(entry.contentRect.width);
      const height = Math.round(entry.contentRect.height);
      if (!width || !height || !plot._fullLayout) return;
      if (
        plot._fullLayout.width === width &&
        plot._fullLayout.height === height
      )
        return;
      getElement(plot.id.slice(1)).Plotly.relayout(plot, { width, height });
    });
    plot.theseusResizeObserver.observe(plot.parentElement);
  },
  mount(host) {
    const plot = host?.querySelector(".js-plotly-plot");
    if (!plot?._fullLayout) return;
    if (host.chartMounted) {
      host.chartResize();
      return;
    }
    host.chartMounted = true;
    const overlay = document.createElement("div");
    overlay.className = "chart-gestures";
    const box = document.createElement("div");
    box.className = "zoom-box";
    const line = document.createElement("div");
    line.className = "inspect-line";
    overlay.append(box, line);
    host.append(overlay);
    host.chartResize = () => {
      const { xaxis: x, yaxis: y } = plot._fullLayout;
      Object.assign(overlay.style, {
        left: x._offset + "px",
        top: y._offset + "px",
        width: x._length + "px",
        height: y._length + "px",
      });
    };
    host.chartResize();
    let start = null,
      last = null,
      timer = null,
      held = false,
      hovering = false;
    const send = (detail) =>
      host.dispatchEvent(new CustomEvent("chartgesture", { detail }));
    const point = (e) => {
      const bounds = overlay.getBoundingClientRect();
      const px = Math.max(0, Math.min(bounds.width, e.clientX - bounds.left));
      const py = Math.max(0, Math.min(bounds.height, e.clientY - bounds.top));
      const { xaxis: x, yaxis: y } = plot._fullLayout;
      return {
        px,
        py,
        x: x.range[0] + (px / bounds.width) * (x.range[1] - x.range[0]),
        y: y.range[1] - (py / bounds.height) * (y.range[1] - y.range[0]),
      };
    };
    const checkpoint = (p) => {
      const { xaxis: x, yaxis: y } = plot._fullLayout;
      for (const trace of plot.data.filter((t) => t.name === "checkpoint")) {
        for (let i = 0; i < trace.x.length; i++) {
          const tx = x.type === "log" ? Math.log10(trace.x[i]) : trace.x[i];
          const ty = y.type === "log" ? Math.log10(trace.y[i]) : trace.y[i];
          const px =
            ((tx - x.range[0]) / (x.range[1] - x.range[0])) * x._length;
          const py =
            ((y.range[1] - ty) / (y.range[1] - y.range[0])) * y._length;
          if (Math.hypot(px - p.px, py - p.py) <= 6)
            return { ...p, x: tx, px, target: trace.customdata[i] };
        }
      }
      return null;
    };
    const inspect = (p) => {
      line.style.display = "block";
      line.style.left = p.px + "px";
      send({ kind: "inspect", x: p.x });
    };
    overlay.addEventListener("pointerdown", (e) => {
      if (e.button !== 0) return;
      start = last = point(e);
      e.preventDefault();
      overlay.setPointerCapture(e.pointerId);
      held = false;
      line.style.display = "none";
      timer = setTimeout(() => {
        held = true;
        inspect(last);
      }, 350);
    });
    overlay.addEventListener("pointermove", (e) => {
      const p = point(e);
      if (!start) {
        const hit = checkpoint(p);
        if (hit) {
          hovering = true;
          inspect(hit);
        } else if (hovering) {
          hovering = false;
          line.style.display = "none";
          send({ kind: "leave" });
        }
        return;
      }
      last = p;
      if (held) {
        inspect(p);
        return;
      }
      if (Math.abs(p.px - start.px) >= 2 && Math.abs(p.py - start.py) >= 2)
        clearTimeout(timer);
      Object.assign(box.style, {
        display: "block",
        left: Math.min(start.px, p.px) + "px",
        top: Math.min(start.py, p.py) + "px",
        width: Math.abs(start.px - p.px) + "px",
        height: Math.abs(start.py - p.py) + "px",
      });
    });
    overlay.addEventListener("pointerup", (e) => {
      if (!start) return;
      clearTimeout(timer);
      const p = point(e),
        origin = start;
      start = null;
      overlay.releasePointerCapture(e.pointerId);
      box.style.display = "none";
      if (held) return;
      if (Math.abs(p.px - origin.px) >= 2 && Math.abs(p.py - origin.py) >= 2) {
        send({
          kind: "zoom",
          box: [
            Math.min(origin.x, p.x),
            Math.max(origin.x, p.x),
            Math.min(origin.y, p.y),
            Math.max(origin.y, p.y),
          ],
        });
      } else {
        const hit = checkpoint(p);
        if (hit) inspect(hit);
        else {
          line.style.display = "none";
          send({ kind: "reset" });
        }
      }
    });
    overlay.addEventListener("dblclick", (e) => {
      const hit = checkpoint(point(e));
      if (hit) {
        e.preventDefault();
        send({ kind: "open", target: hit.target });
      }
    });
    overlay.addEventListener("pointercancel", () => {
      clearTimeout(timer);
      start = null;
      box.style.display = "none";
    });
    overlay.addEventListener("pointerleave", () => {
      if (hovering) {
        hovering = false;
        line.style.display = "none";
        send({ kind: "leave" });
      }
    });
  },
};
