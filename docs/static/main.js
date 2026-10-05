/* DVLA-RL++ project page: navigation highlighting, result tabs, ablation chart, equation rendering. */
(function () {
  "use strict";

  /* ---------------------------------------------------------- equations */
  function renderMath() {
    if (typeof window.renderMathInElement !== "function") return;
    window.renderMathInElement(document.body, {
      delimiters: [
        { left: "\\[", right: "\\]", display: true },
        { left: "\\(", right: "\\)", display: false },
      ],
      throwOnError: false,
      ignoredClasses: ["citation", "eq-fallback"],
    });
  }

  /* ---------------------------------------------------------- sticky nav */
  function initNav() {
    var links = Array.prototype.slice.call(document.querySelectorAll(".nav-inner a[href^='#']"));
    if (!links.length) return;
    var sections = links.map(function (a) { return document.getElementById(a.getAttribute("href").slice(1)); });
    var ticking = false;
    function update() {
      ticking = false;
      var probe = window.scrollY + window.innerHeight * 0.35;
      var current = null;
      sections.forEach(function (s, i) {
        if (s && s.offsetTop <= probe) current = links[i];
      });
      if (window.scrollY < sections[0].offsetTop - window.innerHeight * 0.35) current = null;
      links.forEach(function (a) { a.classList.toggle("is-active", a === current); });
    }
    window.addEventListener("scroll", function () {
      if (!ticking) { ticking = true; window.requestAnimationFrame(update); }
    }, { passive: true });
    window.addEventListener("resize", update);
    update();
  }

  /* ---------------------------------------------------------- result tabs */
  function initTabs() {
    document.querySelectorAll("[data-tabs]").forEach(function (root) {
      var tabs = root.querySelectorAll(".tab");
      var panels = root.querySelectorAll(".tab-panel");
      function select(name) {
        tabs.forEach(function (t) { t.setAttribute("aria-selected", t.dataset.tab === name ? "true" : "false"); });
        panels.forEach(function (p) { p.hidden = p.dataset.panel !== name; });
      }
      tabs.forEach(function (t) { t.addEventListener("click", function () { select(t.dataset.tab); }); });
      select(tabs[0].dataset.tab);
    });
  }

  /* ---------------------------------------------------------- ablation chart (Table 4) */
  var ABLATION = {
    miniImageNet: [
      { name: "DVLA-RL", clean: 81.7, shift: 75.8 },
      { name: "+ CSP", clean: 82.6, shift: 79.4 },
      { name: "+ CFG", clean: 82.4, shift: 78.4 },
      { name: "DVLA-RL++", clean: 83.4, shift: 81.2 },
    ],
    CUB: [
      { name: "DVLA-RL", clean: 91.9, shift: 85.0 },
      { name: "+ CSP", clean: 93.4, shift: 89.4 },
      { name: "+ CFG", clean: 93.0, shift: 88.1 },
      { name: "DVLA-RL++", clean: 94.3, shift: 92.0 },
    ],
  };

  function svgEl(tag, attrs, text) {
    var el = document.createElementNS("http://www.w3.org/2000/svg", tag);
    Object.keys(attrs || {}).forEach(function (k) { el.setAttribute(k, attrs[k]); });
    if (text !== undefined) el.textContent = text;
    return el;
  }

  function drawChart(svg, rows, tip) {
    while (svg.firstChild) svg.removeChild(svg.firstChild);
    var W = 860, H = 300, padL = 54, padR = 16, padT = 24, padB = 46;
    var innerW = W - padL - padR, innerH = H - padT - padB;
    var values = [];
    rows.forEach(function (r) { values.push(r.clean, r.shift); });
    var lo = Math.floor(Math.min.apply(null, values) / 2) * 2 - 2;
    var hi = Math.ceil(Math.max.apply(null, values) / 2) * 2 + 2;
    var y = function (v) { return padT + innerH * (1 - (v - lo) / (hi - lo)); };
    svg.setAttribute("viewBox", "0 0 " + W + " " + H);

    for (var g = lo; g <= hi; g += 2) {
      svg.appendChild(svgEl("line", { x1: padL, x2: W - padR, y1: y(g), y2: y(g), stroke: "#edf1f5" }));
      svg.appendChild(svgEl("text", { x: padL - 8, y: y(g) + 4, "text-anchor": "end", "font-size": 12, fill: "#7a8794" }, g.toFixed(0)));
    }
    svg.appendChild(svgEl("text", { x: 14, y: padT + innerH / 2, "font-size": 12, fill: "#7a8794", transform: "rotate(-90 14 " + (padT + innerH / 2) + ")", "text-anchor": "middle" }, "5-way 1-shot accuracy (%)"));

    var groupW = innerW / rows.length, barW = Math.min(46, groupW * 0.3), gap = 8;
    rows.forEach(function (r, i) {
      var cx = padL + groupW * (i + 0.5);
      [["clean", "#7ca6c9", "Clean"], ["shift", "#d1848c", "Shift"]].forEach(function (spec, j) {
        var key = spec[0], color = spec[1], label = spec[2];
        var x = cx - barW - gap / 2 + j * (barW + gap);
        var v = r[key];
        var rect = svgEl("rect", { class: "bar", x: x, y: y(v), width: barW, height: y(lo) - y(v), rx: 3, fill: color });
        rect.addEventListener("mousemove", function (ev) {
          tip.textContent = r.name + " · " + label + ": " + v.toFixed(1) + "%";
          tip.style.left = ev.clientX + 12 + "px";
          tip.style.top = ev.clientY - 30 + "px";
          tip.classList.add("is-visible");
        });
        rect.addEventListener("mouseleave", function () { tip.classList.remove("is-visible"); });
        var title = svgEl("title", {}, r.name + " " + label + " " + v.toFixed(1) + "%");
        rect.appendChild(title);
        svg.appendChild(rect);
        svg.appendChild(svgEl("text", { x: x + barW / 2, y: y(v) - 6, "text-anchor": "middle", "font-size": 12, fill: "#3e5265", "font-weight": 600 }, v.toFixed(1)));
      });
      var drop = (r.clean - r.shift).toFixed(1);
      svg.appendChild(svgEl("text", { x: cx, y: H - 24, "text-anchor": "middle", "font-size": 13, fill: "#1e2a36", "font-weight": r.name === "DVLA-RL++" ? 700 : 500 }, r.name));
      svg.appendChild(svgEl("text", { x: cx, y: H - 8, "text-anchor": "middle", "font-size": 11, fill: "#7a8794" }, "drop " + drop + " pp"));
    });
  }

  function initChart() {
    var card = document.querySelector("[data-ablation-chart]");
    if (!card) return;
    var svg = card.querySelector("svg");
    var tip = document.createElement("div");
    tip.className = "chart-tip";
    document.body.appendChild(tip);
    var buttons = card.querySelectorAll(".tab");
    function select(name) {
      buttons.forEach(function (b) { b.setAttribute("aria-selected", b.dataset.dataset === name ? "true" : "false"); });
      drawChart(svg, ABLATION[name], tip);
    }
    buttons.forEach(function (b) { b.addEventListener("click", function () { select(b.dataset.dataset); }); });
    select("miniImageNet");
  }

  /* ---------------------------------------------------------- paper button */
  function initPaperButton() {
    var btn = document.querySelector("[data-paper-button]");
    if (!btn || !window.fetch) return;
    var href = btn.dataset.paperHref;
    fetch(href, { method: "HEAD" }).then(function (r) {
      if (r.ok) {
        btn.classList.remove("is-disabled");
        btn.setAttribute("href", href);
        btn.removeAttribute("aria-disabled");
        btn.removeAttribute("title");
      }
    }).catch(function () { /* keep disabled */ });
  }

  document.addEventListener("DOMContentLoaded", function () {
    renderMath();
    initNav();
    initTabs();
    initChart();
    initPaperButton();
  });
})();
