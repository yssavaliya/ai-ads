(function () {
  "use strict";

  var toast = document.getElementById("toast");
  function say(msg) {
    if (!toast) return;
    toast.textContent = msg;
    toast.classList.add("show");
    clearTimeout(say.t);
    say.t = setTimeout(function () { toast.classList.remove("show"); }, 1600);
  }

  // ---- copy buttons -------------------------------------------------------
  function copyText(text) {
    if (navigator.clipboard && window.isSecureContext) return navigator.clipboard.writeText(text);
    return new Promise(function (resolve, reject) {
      var ta = document.createElement("textarea");
      ta.value = text; ta.style.position = "fixed"; ta.style.opacity = "0";
      document.body.appendChild(ta); ta.select();
      try { document.execCommand("copy") ? resolve() : reject(); } catch (e) { reject(e); }
      document.body.removeChild(ta);
    });
  }
  document.addEventListener("click", function (ev) {
    var btn = ev.target.closest(".copy");
    if (!btn) return;
    var target = document.querySelector(btn.getAttribute("data-copy"));
    if (!target) return;
    copyText(target.innerText.trim()).then(function () {
      btn.classList.add("done"); btn.textContent = "Copied ✓"; say("Copied to clipboard");
      setTimeout(function () { btn.classList.remove("done"); btn.textContent = "Copy"; }, 1500);
    }, function () { say("Copy failed — select the text manually"); });
  });

  // ---- posted tracker (this browser only) ---------------------------------
  var store = {
    get: function () { try { return JSON.parse(localStorage.getItem("hub-posted") || "{}"); } catch (e) { return {}; } },
    set: function (v) { try { localStorage.setItem("hub-posted", JSON.stringify(v)); } catch (e) { /* private mode */ } }
  };
  var posted = store.get();
  var cards = Array.prototype.slice.call(document.querySelectorAll(".card[data-id]"));
  cards.forEach(function (card) {
    var box = card.querySelector(".posted-box");
    if (!box) return;
    box.checked = !!posted[card.dataset.id];
    card.classList.toggle("is-posted", box.checked);
    box.addEventListener("change", function () {
      if (box.checked) posted[card.dataset.id] = 1; else delete posted[card.dataset.id];
      store.set(posted);
      card.classList.toggle("is-posted", box.checked);
      apply();
    });
  });

  // ---- filters ------------------------------------------------------------
  var q = document.getElementById("q");
  var hidePosted = document.getElementById("hide-posted");
  var chips = document.getElementById("type-chips");
  var activeType = "";

  function apply() {
    var needle = q ? q.value.trim().toLowerCase() : "";
    cards.forEach(function (card) {
      var show = true;
      if (activeType && card.dataset.type !== activeType) show = false;
      if (needle && card.dataset.search.indexOf(needle) === -1) show = false;
      if (hidePosted && hidePosted.checked && card.classList.contains("is-posted")) show = false;
      card.classList.toggle("hidden", !show);
    });
    Array.prototype.forEach.call(document.querySelectorAll(".day"), function (day) {
      var any = day.querySelector(".card:not(.hidden)");
      day.style.display = any ? "" : "none";
    });
  }
  if (q) q.addEventListener("input", apply);
  if (hidePosted) hidePosted.addEventListener("change", apply);
  if (chips) chips.addEventListener("click", function (ev) {
    var chip = ev.target.closest(".chip");
    if (!chip) return;
    activeType = chip.dataset.type || "";
    Array.prototype.forEach.call(chips.querySelectorAll(".chip"), function (c) { c.classList.toggle("active", c === chip); });
    apply();
  });
  apply();
})();
