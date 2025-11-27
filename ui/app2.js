// Assistant UI controller: chat, streaming, docs, pins, tooltip lookup + hover.
// ASCII only except emoji icons for tooltip categories (per user preference).

const state = {
  chatId: null,
  chats: [],
  docs: [],
  selectedDoc: null,
  pinnedPaths: [],
  tooltips: null,
  tooltipKeys: [],
  tooltipCache: [],
  tooltipLookup: {},
  tooltipRegex: null,
  tooltipPanelAnchor: null,
  activeTooltipKey: null,
  streamController: null,
  typingBubble: null,
};

const API_BASE_META_VALUE = (() => {
  if (typeof document === "undefined") return "";
  const meta = document.querySelector('meta[name="api-base"]');
  return meta?.content?.trim() || "";
})();

const API_BASE_URL = (() => {
  const origin =
    typeof window !== "undefined" && window.location && window.location.origin && window.location.origin !== "null"
      ? window.location.origin
      : "";
  if (!API_BASE_META_VALUE) {
    return origin;
  }
  if (typeof window !== "undefined" && window.location) {
    try {
      return new URL(API_BASE_META_VALUE, window.location.href).toString().replace(/\/$/, "");
    } catch {
      const trimmed = API_BASE_META_VALUE.replace(/\/$/, "");
      if (trimmed.startsWith("/")) {
        return origin ? `${origin}${trimmed}` : trimmed;
      }
      return trimmed;
    }
  }
  return API_BASE_META_VALUE.replace(/\/$/, "");
})();

function buildApiUrl(path) {
  const normalized = path.startsWith("/") ? path : `/${path}`;
  if (API_BASE_URL) {
    return `${API_BASE_URL}${normalized}`;
  }
  return normalized;
}

function apiFetch(path, options) {
  return fetch(buildApiUrl(path), options);
}

const STORAGE_KEYS = {
  backend: "rag.ui.backend",
  openaiKey: "rag.ui.openaiKey",
  openaiModel: "rag.ui.openaiModel",
  openrouterKey: "rag.ui.openrouterKey",
  openrouterModel: "rag.ui.openrouterModel",
  openaiCompatKey: "rag.ui.openaiCompatKey",
  openaiCompatModel: "rag.ui.openaiCompatModel",
  openaiCompatBase: "rag.ui.openaiCompatBase",
};

const STORAGE_ENABLED = (() => {
  try {
    if (typeof window === "undefined" || !window.localStorage) return false;
    const probe = "__rag_probe__";
    window.localStorage.setItem(probe, "1");
    window.localStorage.removeItem(probe);
    return true;
  } catch (_) {
    return false;
  }
})();

function cacheGet(key) {
  if (!STORAGE_ENABLED) return null;
  try {
    return window.localStorage.getItem(key);
  } catch (_) {
    return null;
  }
}

function cacheSet(key, value) {
  if (!STORAGE_ENABLED) return;
  try {
    if (value == null || value === "") {
      window.localStorage.removeItem(key);
    } else {
      window.localStorage.setItem(key, value);
    }
  } catch (_) {
    /* ignore storage failures */
  }
}

function cacheRemove(keys) {
  if (!STORAGE_ENABLED) return;
  (Array.isArray(keys) ? keys : [keys]).forEach((key) => {
    try {
      window.localStorage.removeItem(key);
    } catch (_) {
      /* ignore */
    }
  });
}

const DRAWER_IDS = ["historyDrawer", "settingsDrawer", "docsDrawer"];
let drawersBound = false;

function setDrawerVisibility(id, open) {
  const drawer = $(id);
  if (!drawer) return;
  drawer.setAttribute("aria-hidden", open ? "false" : "true");
}

function refreshDrawerLock() {
  const anyOpen = DRAWER_IDS.some((drawerId) => $(drawerId)?.getAttribute("aria-hidden") === "false");
  document.body.classList.toggle("drawer-open", anyOpen);
}

function openDrawer(id) {
  DRAWER_IDS.forEach((drawerId) => {
    setDrawerVisibility(drawerId, drawerId === id);
  });
  refreshDrawerLock();
}

function closeDrawer(id) {
  setDrawerVisibility(id, false);
  refreshDrawerLock();
}

function toggleDrawer(id) {
  const drawer = $(id);
  if (!drawer) return;
  const isOpen = drawer.getAttribute("aria-hidden") === "false";
  if (isOpen) {
    closeDrawer(id);
  } else {
    openDrawer(id);
  }
}

function closeAllDrawers() {
  DRAWER_IDS.forEach((drawerId) => setDrawerVisibility(drawerId, false));
  refreshDrawerLock();
}

function bindDrawerTriggers() {
  if (drawersBound) return;
  drawersBound = true;
  $("historyBtn")?.addEventListener("click", () => toggleDrawer("historyDrawer"));
  $("docsBtn")?.addEventListener("click", () => toggleDrawer("docsDrawer"));
  $("settingsBtn")?.addEventListener("click", () => toggleDrawer("settingsDrawer"));
  document.querySelectorAll("[data-close]").forEach((el) => {
    const target = el.getAttribute("data-close");
    if (!target) return;
    el.addEventListener("click", () => closeDrawer(target));
  });
  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape") closeAllDrawers();
  });
  refreshDrawerLock();
}

// ---------- Utilities ----------
const $ = (id) => document.getElementById(id);
const qs = (root, sel) => (root || document).querySelector(sel);
const create = (tag, className) => {
  const el = document.createElement(tag);
  if (className) el.className = className;
  return el;
};
const setText = (el, value) => { if (el) el.textContent = value; };
const debounce = (fn, delay) => {
  let to = 0;
  return (...args) => {
    clearTimeout(to);
    to = window.setTimeout(() => fn(...args), delay);
  };
};
const escapeRegex = (s) => s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");

// ---------- Status ----------
async function updateStatus() {
  try {
    const res = await apiFetch("/api/status");
    const data = await res.json();
    const backend = currentBackend();
    const tag = $("modelTag");
    if (tag) {
      let label = "";
      if (backend === "openai") {
        const custom = ($("openaiModel")?.value || "").trim();
        label = custom || "OpenAI (custom)";
      } else if (backend === "openrouter") {
        const custom = ($("openrouterModel")?.value || "").trim();
        label = custom || "OpenRouter (custom)";
      } else if (backend === "openai_compat") {
        const compatModel = ($("openaiCompatModel")?.value || "").trim();
        const compatBase = ($("openaiCompatBase")?.value || "").trim();
        label = compatModel || compatBase || "OpenAI-compatible endpoint";
      } else if (data.model) {
        label = data.model;
      } else if (backend) {
        label = backend;
      }
      tag.textContent = label ? `Model: ${label}` : "";
    }
    if (data.context_cap != null && $("capVal")) {
      setText($("capVal"), String(data.context_cap));
    }
  } catch (_) {
    /* ignore diagnostics failures */
  }
}

function currentBackend() {
  return ($("backendSel")?.value || "").toLowerCase();
}

function updateBackendVisibility() {
  const backend = currentBackend();
  const isOpenAI = backend === "openai";
  const isOpenRouter = backend === "openrouter";
  const isCompat = backend === "openai_compat";
  const setRow = (id, show) => {
    const node = $(id);
    if (!node) return;
    if (!node.dataset.defaultDisplay || node.dataset.defaultDisplay === "none") {
      const computed = typeof window !== "undefined" && window.getComputedStyle ? window.getComputedStyle(node).display : "";
      node.dataset.defaultDisplay = computed && computed !== "none" ? computed : "flex";
    }
    node.style.display = show ? node.dataset.defaultDisplay : "none";
  };
  setRow("openaiKeyRow", isOpenAI);
  setRow("openaiModelRow", isOpenAI);
  setRow("orKeyRow", isOpenRouter);
  setRow("orModelRow", isOpenRouter);
  setRow("compatBaseRow", isCompat);
  setRow("compatKeyRow", isCompat);
  setRow("compatModelRow", isCompat);
}

function toggleCacheBadge(id, shouldShow) {
  const el = $(id);
  if (!el) return;
  el.hidden = !(STORAGE_ENABLED && shouldShow);
}

function updateCacheStatusNote() {
  const note = $("cacheStatus");
  if (!note) return;
  if (!STORAGE_ENABLED) {
    note.textContent = "Browser storage unavailable; credentials reset on reload.";
    note.classList.add("warn");
  } else {
    note.textContent = "Credentials stay local to this browser (cleared on request).";
    note.classList.remove("warn");
  }
}

function applyCachedSettings() {
  updateCacheStatusNote();
  const backendSel = $("backendSel");
  const openaiInput = $("openaiKey");
  const openaiModelInput = $("openaiModel");
  const orKeyInput = $("openrouterKey");
  const orModelInput = $("openrouterModel");
  const compatKeyInput = $("openaiCompatKey");
  const compatModelInput = $("openaiCompatModel");
  const compatBaseInput = $("openaiCompatBase");
  if (STORAGE_ENABLED) {
    const backendVal = cacheGet(STORAGE_KEYS.backend);
    if (backendSel && backendVal != null) backendSel.value = backendVal;
    const openaiVal = cacheGet(STORAGE_KEYS.openaiKey);
    if (openaiInput && openaiVal != null) openaiInput.value = openaiVal;
    const openaiModelVal = cacheGet(STORAGE_KEYS.openaiModel);
    if (openaiModelInput && openaiModelVal != null) openaiModelInput.value = openaiModelVal;
    const orKeyVal = cacheGet(STORAGE_KEYS.openrouterKey);
    if (orKeyInput && orKeyVal != null) orKeyInput.value = orKeyVal;
    const orModelVal = cacheGet(STORAGE_KEYS.openrouterModel);
    if (orModelInput && orModelVal != null) orModelInput.value = orModelVal;
    const compatKeyVal = cacheGet(STORAGE_KEYS.openaiCompatKey);
    if (compatKeyInput && compatKeyVal != null) compatKeyInput.value = compatKeyVal;
    const compatModelVal = cacheGet(STORAGE_KEYS.openaiCompatModel);
    if (compatModelInput && compatModelVal != null) compatModelInput.value = compatModelVal;
    const compatBaseVal = cacheGet(STORAGE_KEYS.openaiCompatBase);
    if (compatBaseInput && compatBaseVal != null) compatBaseInput.value = compatBaseVal;
  }
  toggleCacheBadge("backendCached", !!(backendSel?.value || "").trim());
  toggleCacheBadge("openaiKeyCached", !!(openaiInput?.value || "").trim());
  toggleCacheBadge("openaiModelCached", !!(openaiModelInput?.value || "").trim());
  toggleCacheBadge("openrouterKeyCached", !!(orKeyInput?.value || "").trim());
  toggleCacheBadge("openrouterModelCached", !!(orModelInput?.value || "").trim());
  toggleCacheBadge("openaiCompatKeyCached", !!(compatKeyInput?.value || "").trim());
  toggleCacheBadge("openaiCompatModelCached", !!(compatModelInput?.value || "").trim());
  toggleCacheBadge("openaiCompatBaseCached", !!(compatBaseInput?.value || "").trim());
  updateBackendVisibility();
}

function setupCacheListeners() {
  const backendSel = $("backendSel");
  const openaiInput = $("openaiKey");
  const openaiModelInput = $("openaiModel");
  const orKeyInput = $("openrouterKey");
  const orModelInput = $("openrouterModel");
  const compatKeyInput = $("openaiCompatKey");
  const compatModelInput = $("openaiCompatModel");
  const compatBaseInput = $("openaiCompatBase");
  const clearBtn = $("clearCredsBtn");

  const syncBadge = (input, indicatorId) => {
    if (indicatorId) toggleCacheBadge(indicatorId, !!(input?.value || "").trim());
  };

  backendSel?.addEventListener("change", () => {
    const val = (backendSel.value || "").trim();
    cacheSet(STORAGE_KEYS.backend, val);
    toggleCacheBadge("backendCached", !!val);
    updateBackendVisibility();
    updateStatus();
  });

  const bindSecretField = (input, storageKey, indicatorId) => {
    if (!input) return;
    input.addEventListener("input", () => {
      cacheSet(storageKey, input.value.trim());
      syncBadge(input, indicatorId);
    });
    input.addEventListener("change", () => {
      const value = input.value.trim();
      input.value = value;
      cacheSet(storageKey, value);
      syncBadge(input, indicatorId);
    });
  };

  bindSecretField(openaiInput, STORAGE_KEYS.openaiKey, "openaiKeyCached");
  bindSecretField(openaiModelInput, STORAGE_KEYS.openaiModel, "openaiModelCached");
  bindSecretField(orKeyInput, STORAGE_KEYS.openrouterKey, "openrouterKeyCached");
  bindSecretField(orModelInput, STORAGE_KEYS.openrouterModel, "openrouterModelCached");
  bindSecretField(compatKeyInput, STORAGE_KEYS.openaiCompatKey, "openaiCompatKeyCached");
  bindSecretField(compatModelInput, STORAGE_KEYS.openaiCompatModel, "openaiCompatModelCached");
  bindSecretField(compatBaseInput, STORAGE_KEYS.openaiCompatBase, "openaiCompatBaseCached");
  openaiModelInput?.addEventListener("input", () => updateStatus());
  orModelInput?.addEventListener("input", () => updateStatus());
  compatModelInput?.addEventListener("input", () => updateStatus());
  compatBaseInput?.addEventListener("input", () => updateStatus());

  clearBtn?.addEventListener("click", () => {
    cacheRemove([
      STORAGE_KEYS.backend,
      STORAGE_KEYS.openaiKey,
      STORAGE_KEYS.openaiModel,
      STORAGE_KEYS.openrouterKey,
      STORAGE_KEYS.openrouterModel,
      STORAGE_KEYS.openaiCompatKey,
      STORAGE_KEYS.openaiCompatModel,
      STORAGE_KEYS.openaiCompatBase,
    ]);
    if (backendSel) backendSel.value = "";
    if (openaiInput) openaiInput.value = "";
    if (openaiModelInput) openaiModelInput.value = "";
    if (orKeyInput) orKeyInput.value = "";
    if (orModelInput) orModelInput.value = "";
    if (compatKeyInput) compatKeyInput.value = "";
    if (compatModelInput) compatModelInput.value = "";
    if (compatBaseInput) compatBaseInput.value = "";
    updateBackendVisibility();
    toggleCacheBadge("backendCached", false);
    toggleCacheBadge("openaiKeyCached", false);
    toggleCacheBadge("openaiModelCached", false);
    toggleCacheBadge("openrouterKeyCached", false);
    toggleCacheBadge("openrouterModelCached", false);
    toggleCacheBadge("openaiCompatKeyCached", false);
    toggleCacheBadge("openaiCompatModelCached", false);
    toggleCacheBadge("openaiCompatBaseCached", false);
    updateStatus();
    if (clearBtn.dataset.timeoutId) {
      window.clearTimeout(Number(clearBtn.dataset.timeoutId));
    }
    const original = clearBtn.dataset.originalText || clearBtn.textContent || "Clear cached values";
    clearBtn.dataset.originalText = original;
    clearBtn.textContent = "Cleared";
    const tid = window.setTimeout(() => {
      clearBtn.textContent = original;
      delete clearBtn.dataset.timeoutId;
    }, 1500);
    clearBtn.dataset.timeoutId = String(tid);
  });
}

// ---------- Chat Rendering ----------
function renderMessages(messages) {
  const wrap = $("chat");
  if (!wrap) return;
  wrap.innerHTML = "";
  const frag = document.createDocumentFragment();
  (messages || []).forEach((msg) => frag.appendChild(buildMessageBubble(msg)));
  wrap.appendChild(frag);
  highlightCode(wrap);
  annotateTooltips(wrap);
}

function buildMessageBubble({ role, content, time }) {
  const bubble = create("div", `msg ${role === "user" ? "user" : "assistant"}`);
  const header = create("div", "hdr");
  const author = create("span");
  author.textContent = role === "user" ? "You" : "assistant";
  const stamp = create("span");
  stamp.textContent = time ? new Date(time).toLocaleTimeString() : "";
  header.append(author, stamp);

  const body = create("div");
  let html = content || "";
  try {
    html = DOMPurify.sanitize(marked.parse(html));
  } catch (_) { /* keep raw */ }
  body.innerHTML = html;
  bubble.append(header, body);
  return bubble;
}

function appendUserBubble(text) {
  const wrap = $("chat");
  if (!wrap) return;
  const bubble = buildMessageBubble({
    role: "user",
    content: text,
    time: new Date().toISOString(),
  });
  wrap.appendChild(bubble);
  wrap.scrollTop = wrap.scrollHeight;
}

function appendTypingBubble() {
  const wrap = $("chat");
  if (!wrap) return;
  const bubble = create("div", "msg assistant typing");
  const header = create("div", "hdr");
  header.innerHTML = `<span>assistant</span><span>${new Date().toLocaleTimeString()}</span>`;
  const dots = create("div", "typing-dots");
  dots.innerHTML = "<span></span><span></span><span></span>";
  const body = create("div");
  body.appendChild(dots);
  bubble.append(header, body);
  wrap.appendChild(bubble);
  wrap.scrollTop = wrap.scrollHeight;
  state.typingBubble = bubble;
}

function replaceTypingBubble(html) {
  if (!state.typingBubble) return;
  const body = qs(state.typingBubble, ":scope > div:not(.hdr)");
  if (!body) return;
  let safe = html || "";
  try {
    safe = DOMPurify.sanitize(marked.parse(safe));
  } catch (_) { /* allow fallback */ }
  body.innerHTML = safe || "(no content)";
  highlightCode(state.typingBubble);
  annotateTooltips(state.typingBubble);
  state.typingBubble.classList.remove("typing");
  state.typingBubble = null;
}

function highlightCode(root) {
  try {
    (root || document).querySelectorAll("pre code").forEach((block) => hljs.highlightElement(block));
  } catch (_) { /* highlight.js optional */ }
  addCopyButtons(root);
}

function addCopyButtons(root) {
  (root || document).querySelectorAll("pre").forEach((pre) => {
    if (pre.querySelector(".copy-btn")) return;
    const btn = create("button", "copy-btn");
    btn.textContent = "copy";
    btn.onclick = () => {
      try {
        navigator.clipboard.writeText(pre.innerText);
        btn.textContent = "copied!";
        setTimeout(() => { btn.textContent = "copy"; }, 900);
      } catch (_) { /* silent */ }
    };
    pre.style.position = "relative";
    btn.style.position = "absolute";
    btn.style.top = "6px";
    btn.style.right = "6px";
    pre.appendChild(btn);
  });
}

// ---------- Payload ----------
function numericValue(id) {
  const input = $(id);
  if (!input) return undefined;
  const raw = input.value;
  if (raw === "" || raw == null) return undefined;
  const num = Number(raw);
  return Number.isFinite(num) ? num : undefined;
}

function buildPayload(message) {
  return {
    message,
    chat_id: state.chatId,
    temperature: Number(($("temp")?.value || "0.3")) || 0.3,
    plan_mode: !!$("plan")?.checked,
    self_check: !!$("selfCheck")?.checked,
    strict_mode: false,
    plan_depth: parseInt(($("planDepth")?.value || "2"), 10) || 2,
    top_k: numericValue("setTopK"),
    top_p: numericValue("setTopP"),
    repeat_penalty: numericValue("setRep"),
    freq_penalty: numericValue("setFreq"),
    retr_k: parseInt(($("setRetrK")?.value || "5"), 10) || 5,
    max_context_tokens: Math.max(10000, parseInt(($("setMaxContextTokens")?.value || "3000000"), 10) || 3000000),
    pinned_paths: state.pinnedPaths.slice(),
    pins_only: !!$("pinsOnly")?.checked,
    backend: ($("backendSel")?.value || "").trim(),
    openai_key: ($("openaiKey")?.value || "").trim(),
    openai_model: ($("openaiModel")?.value || "").trim(),
    openrouter_key: ($("openrouterKey")?.value || "").trim(),
    openrouter_model: ($("openrouterModel")?.value || "").trim(),
    openai_compat_key: ($("openaiCompatKey")?.value || "").trim(),
    openai_compat_model: ($("openaiCompatModel")?.value || "").trim(),
    openai_compat_base_url: ($("openaiCompatBase")?.value || "").trim(),
    _stream: !!$("stream")?.checked,
  };
}

// ---------- Chat API ----------
function renderChatHistory() {
  const list = $("chatList");
  if (!list) return;
  list.innerHTML = "";
  if (!state.chats.length) {
    const empty = create("div", "history-empty");
    empty.textContent = "No conversations yet.";
    list.appendChild(empty);
    return;
  }
  const frag = document.createDocumentFragment();
  state.chats.forEach((chat) => {
    const btn = create("button", "history-item");
    btn.type = "button";
    btn.dataset.chatId = chat.id;
    const when = new Date(chat.timestamp);
    const label = create("span", "history-label");
    label.textContent = when.toLocaleString();
    btn.title = label.textContent;
    btn.appendChild(label);
    if (chat.id === state.chatId) {
      btn.classList.add("active");
      btn.setAttribute("aria-current", "true");
      const pill = create("span", "history-pill");
      pill.textContent = "current";
      btn.appendChild(pill);
    }
    if (chat.id !== state.chatId) {
      btn.removeAttribute("aria-current");
    }
    frag.appendChild(btn);
  });
  list.appendChild(frag);
}

async function fetchChats() {
  try {
    const res = await apiFetch("/api/chats");
    const data = await res.json();
    state.chats = data.chats || [];
    renderChatHistory();
  } catch (_) { /* ignore */ }
}

async function loadChat(chatId) {
  if (!chatId) return;
  try {
    const res = await apiFetch(`/api/chats/${encodeURIComponent(chatId)}`);
    const data = await res.json();
    state.chatId = data.id;
    renderChatHistory();
    renderMessages(data.messages || []);
  } catch (_) { /* ignore */ }
}

async function startNewChat() {
  try {
    const res = await apiFetch("/api/new_chat", { method: "POST" });
    const data = await res.json();
    state.chatId = data.id;
    await Promise.all([fetchChats(), fetchDocs()]);
    updatePinsUI();
    renderMessages([]);
  } catch (_) { /* ignore */ }
}

async function clearChat() {
  if (!state.chatId) return;
  try {
    await apiFetch(`/api/clear_chat?chat_id=${encodeURIComponent(state.chatId)}`, { method: "POST" });
    await loadChat(state.chatId);
  } catch (_) { /* ignore */ }
}

function exportChat() {
  if (state.chatId) window.open(buildApiUrl(`/api/chats/${encodeURIComponent(state.chatId)}`), "_blank");
}

async function handleSend() {
  const input = $("msg");
  if (!input) return;
  const message = input.value.trim();
  if (!message) return;

  appendUserBubble(message);
  appendTypingBubble();

  const payload = buildPayload(message);
  const sendBtn = $("send");
  const stopBtn = $("stop");
  const previousLabel = sendBtn?.textContent || "Send";
  if (sendBtn) {
    sendBtn.disabled = true;
    sendBtn.textContent = "Sending...";
  }
  if (stopBtn) {
    stopBtn.disabled = false;
    stopBtn.style.display = "inline-block";
  }

  try {
    if (payload._stream) {
      await sendStream(payload);
    } else {
      await sendOnce(payload);
    }
    input.value = "";
  } finally {
    if (sendBtn) {
      sendBtn.disabled = false;
      sendBtn.textContent = previousLabel;
    }
  }
}

async function sendOnce(payload) {
  try {
    const res = await apiFetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const data = await res.json();
    state.chatId = data.id;
    await fetchChats();
    setText($("perf"), formatPerf(data.diagnostics?.perf));
    const last = data.messages?.[data.messages.length - 1]?.content || "";
    replaceTypingBubble(last);
  } catch (err) {
    replaceTypingBubble(`[error] ${err}`);
  }
  hideStopButton();
}

async function sendStream(payload) {
  const controller = new AbortController();
  state.streamController = controller;
  let accumulated = "";
  try {
    const res = await apiFetch("/api/chat_stream", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
      signal: controller.signal,
    });
    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      let newline;
      while ((newline = buffer.indexOf("\n")) >= 0) {
        const line = buffer.slice(0, newline).trim();
        buffer = buffer.slice(newline + 1);
        if (!line) continue;
        try {
          const event = JSON.parse(line);
          if (event.type === "chunk") {
            accumulated += event.text || "";
            const body = state.typingBubble && qs(state.typingBubble, ":scope > div:not(.hdr)");
            if (body) body.textContent = accumulated;
          } else if (event.type === "done") {
            state.chatId = event.id;
            await fetchChats();
            setText($("perf"), formatPerf(event.perf));
          }
        } catch (_) { /* ignore malformed */ }
      }
    }
  } catch (err) {
    accumulated = accumulated || `[error] ${err}`;
  }
  replaceTypingBubble(accumulated || "(no content)");
  hideStopButton();
}

function hideStopButton() {
  state.streamController = null;
  const stopBtn = $("stop");
  if (stopBtn) {
    stopBtn.disabled = true;
    stopBtn.style.display = "none";
  }
}

async function forceStop() {
  try {
    state.streamController?.abort();
    if (state.chatId) {
      await apiFetch("/api/cancel", {
        method: "POST",
        headers:{ "Content-Type": "application/json" },
        body: JSON.stringify({ chat_id: state.chatId }),
      });
    }
  } catch (_) { /* ignore */ }
  hideStopButton();
}

function formatPerf(perf) {
  if (!perf) return "";
  const parts = [
    `time ${perf.ms || 0} ms`,
    `in ${perf.tin || 0} tok`,
    `out ${perf.tout || 0} tok`,
  ];
  if (perf.tps != null) parts.push(`${Number(perf.tps).toFixed(1)} tokens/sec`);
  return parts.join(" | ");
}

// ---------- Docs ----------
async function fetchDocs() {
  try {
    const res = await apiFetch("/api/docs_list");
    const data = await res.json();
    state.docs = data.files || [];
    renderDocsTree(state.docs);
  } catch (_) { /* ignore */ }
}

function renderDocsTree(paths) {
  const container = $("docsList");
  if (!container) return;
  container.innerHTML = "";
  const tree = buildDocTree(paths);
  const root = create("details");
  root.open = true;
  const summary = create("summary", "folder");
  summary.textContent = "Docs";
  root.appendChild(summary);
  root.appendChild(renderTreeNode(tree, ""));
  container.appendChild(root);
}

function buildDocTree(paths) {
  const root = {};
  (paths || []).forEach((path) => {
    let normalized = (path || "").replace(/^[\\/]+/, "").replace(/\\/g, "/");
    if (normalized.toLowerCase().startsWith("docs/")) {
      normalized = normalized.slice(5);
    }
    const parts = normalized.split("/").filter(Boolean);
    let node = root;
    parts.forEach((part, idx) => {
      if (idx === parts.length - 1) {
        node[part] = null;
      } else {
        node[part] = node[part] || {};
        node = node[part];
      }
    });
  });
  return root;
}

function renderTreeNode(node, base) {
  const list = create("ul", "docs-tree");
  const folders = Object.keys(node).filter((key) => node[key] && typeof node[key] === "object").sort();
  const files = Object.keys(node).filter((key) => node[key] === null).sort();

  folders.forEach((folder) => {
    const li = create("li");
    const details = create("details");
    const summary = create("summary", "folder");
    summary.textContent = folder;
    details.appendChild(summary);
    details.appendChild(renderTreeNode(node[folder], `${base}${folder}/`));
    li.appendChild(details);
    list.appendChild(li);
  });

  files.forEach((file) => {
    const li = create("li");
    const button = create("button", "file");
    button.type = "button";
    button.textContent = file;
    const fullPath = `${base}${file}`;
    const docPath = `Docs/${fullPath}`;
    button.dataset.path = docPath;
    button.addEventListener("click", () => {
      state.selectedDoc = docPath;
      highlightSelectedDoc();
      loadDoc(docPath);
    });
    li.appendChild(button);
    list.appendChild(li);
  });
  return list;
}

function highlightSelectedDoc() {
  document.querySelectorAll(".docs-list button.file").forEach((btn) => {
    if (btn.dataset.path === state.selectedDoc) btn.classList.add("selected");
    else btn.classList.remove("selected");
  });
}

async function loadDoc(path) {
  try {
    const rel = path && path.startsWith("Docs/") ? path : `Docs/${path || ""}`;
    const res = await apiFetch(`/api/docs_read?path=${encodeURIComponent(rel)}`);
    const data = await res.json();
    const viewer = $("docView");
    if (!viewer) return;
    let html = data.content || "";
    try {
      html = DOMPurify.sanitize(marked.parse(html));
    } catch (_) { /* fallback */ }
    viewer.innerHTML = html;
    highlightCode(viewer);
    annotateTooltips(viewer);
  } catch (_) { /* ignore */ }
}

(function initDocSearch() {
  const input = $("docSearch");
  if (!input) return;
  input.addEventListener("input", () => {
    const q = input.value.trim().toLowerCase();
    const container = $("docsList");
    if (!container) return;
    container.querySelectorAll("button.file").forEach((btn) => {
      const name = (btn.textContent || "").toLowerCase();
      const full = (btn.dataset.path || "").toLowerCase();
      btn.style.display = !q || name.includes(q) || full.includes(q) ? "" : "none";
    });
  });
})();

(function initDocToggles() {
  $("docsExpandAll")?.addEventListener("click", () => {
    document.querySelectorAll("#docsList details").forEach((d) => { d.open = true; });
  });
  $("docsCollapseAll")?.addEventListener("click", () => {
    document.querySelectorAll("#docsList details").forEach((d) => { d.open = false; });
  });
})();

// ---------- Pins ----------
function updatePinsUI() {
  const count = state.pinnedPaths.length;
  setText($("pinnedBadge"), count ? `pinned ${count}` : "");
  setText($("pinStatus"), count ? "pins active" : "no pins");
  const list = $("pinList");
  if (!list) return;
  list.innerHTML = "";
  const frag = document.createDocumentFragment();
  state.pinnedPaths.forEach((path) => {
    const chip = create("span", "pin-chip");
    chip.textContent = path;
    const btn = create("button");
    btn.textContent = "x";
    btn.title = "remove";
    btn.onclick = () => {
      state.pinnedPaths = state.pinnedPaths.filter((p) => p !== path);
      updatePinsUI();
    };
    chip.appendChild(btn);
    frag.appendChild(chip);
  });
  list.appendChild(frag);
}

(function initPins() {
  $("pinBtn")?.addEventListener("click", () => {
    if (!state.selectedDoc) return;
    if (!state.pinnedPaths.includes(state.selectedDoc)) {
      state.pinnedPaths.push(state.selectedDoc);
      updatePinsUI();
    }
  });
  $("clearPinBtn")?.addEventListener("click", () => {
    state.pinnedPaths = [];
    updatePinsUI();
  });
})();

// ---------- Tooltips ----------
const TOOLTIP_ICONS = {
  service: "🏢",
  class: "📦",
  function: "🔧",
  datatype: "📐",
  enum: "📊",
  property: "🧷",
  global: "🌐",
  keyword: "🔑",
  symbol: "🔣",
};

const tooltipPanel = create("div", "tt-tooltip");
document.body.appendChild(tooltipPanel);

document.addEventListener("pointerover", (event) => {
  const target = event.target.closest(".tt-target");
  if (!target) {
    if (!event.relatedTarget || !event.relatedTarget.closest(".tt-target")) hideTooltipPanel();
    return;
  }
  const key = target.dataset.ttKey;
  if (!key) { hideTooltipPanel(); return; }
  state.tooltipPanelAnchor = target;
  showTooltipPanel(key, target);
  positionTooltipPanel(event.clientX, event.clientY);
});

document.addEventListener("pointermove", (event) => {
  if (!tooltipPanel.classList.contains("show")) return;
  positionTooltipPanel(event.clientX, event.clientY);
});

document.addEventListener("pointerout", (event) => {
  if (!tooltipPanel.classList.contains("show")) return;
  const toward = event.relatedTarget;
  if (toward && (toward === tooltipPanel || toward.closest(".tt-target"))) return;
  hideTooltipPanel();
});

function positionTooltipPanel(clientX, clientY) {
  const panel = tooltipPanel;
  const pad = 14;
  const viewportW = window.innerWidth;
  const viewportH = window.innerHeight;
  const rect = panel.getBoundingClientRect();
  let left = clientX + pad;
  let top = clientY + pad;
  if (left + rect.width > viewportW - 8) left = clientX - rect.width - pad;
  if (top + rect.height > viewportH - 8) top = viewportH - rect.height - 8;
  panel.style.left = `${Math.max(8, left)}px`;
  panel.style.top = `${Math.max(8, top)}px`;
}

function showTooltipPanel(key, anchor) {
  const entry = resolveTooltipEntry(key);
  if (!entry) { hideTooltipPanel(); return; }
  const meta = entry.meta || {};
  const bestPractice = $("bestPractice")?.checked;
  let html = `<div class="tt-title">${entry.icon || ""} ${entry.display || entry.key}</div>`;
  if (meta.summary) html += `<div class="tt-summary">${meta.summary}</div>`;
  if (meta.usage) html += `<div class="tt-row"><div class="tt-key">Usage</div><div class="tt-val">${meta.usage}</div></div>`;
  const useCase = meta.useCase || "";
  if (useCase) html += `<div class="tt-row"><div class="tt-key">Use case</div><div class="tt-val">${useCase}</div></div>`;
  const common = meta.mistake || "";
  if (common) html += `<div class="tt-row"><div class="tt-key">Common mistake</div><div class="tt-val">${common}</div></div>`;
  const tip = meta.tip || "";
  if (tip) html += `<div class="tt-row"><div class="tt-key">Tip</div><div class="tt-val">${tip}</div></div>`;
  const notes = meta.notes || [];
  if (notes.length) {
    if (bestPractice) {
      html += `<div class="tt-warn"><div class="tt-warn-title">Notes</div><ul>`
        + notes.map((item) => `<li>${item}</li>`).join("")
        + `</ul></div>`;
    } else {
      html += `<div class="tt-notes"><div class="tt-notes-title">Notes</div><ul>`
        + notes.map((item) => `<li>${item}</li>`).join("")
        + `</ul></div>`;
    }
  }
  tooltipPanel.innerHTML = html;
  tooltipPanel.dataset.cat = entry.category || "";
  tooltipPanel.classList.add("show");
  anchor.dataset.ttActive = "1";
}

function hideTooltipPanel() {
  tooltipPanel.classList.remove("show");
  if (state.tooltipPanelAnchor) {
    delete state.tooltipPanelAnchor.dataset.ttActive;
    state.tooltipPanelAnchor = null;
  }
}

function resolveTooltipEntry(keyOrLabel) {
  const lookup = state.tooltipLookup || {};
  const lower = keyOrLabel?.toLowerCase?.() || "";
  return lookup[lower] || lookup[keyOrLabel] || null;
}

async function loadTooltips() {
  if (state.tooltips) return;
  try {
    const res = await apiFetch("/api/tooltips");
    const data = await res.json();
    state.tooltips = data || {};
    state.tooltipKeys = [];
    state.tooltipCache = [];
    state.tooltipLookup = {};
    const groups = [
      ["services", "service"],
      ["classes", "class"],
      ["functions", "function"],
      ["datatypes", "datatype"],
      ["enums", "enum"],
      ["properties", "property"],
      ["globals", "global"],
      ["keywords", "keyword"],
      ["symbols", "symbol"],
    ];
    groups.forEach(([field, category]) => {
      const bucket = data[field] || {};
      Object.entries(bucket).forEach(([key, meta]) => {
        const lower = key.toLowerCase();
        const raw = meta || {};
        const normalized = {
          summary: raw.summary || raw.Summary || raw.description || raw.Description || "",
          usage: raw.usage || raw.Usage || raw.use || raw.Use || "",
          tip: raw.tip || raw.Tip || "",
          notes: Array.isArray(raw.notes) ? raw.notes : Array.isArray(raw.BestPractice) ? raw.BestPractice : [],
          raw,
        };
        const common = raw["Common mistake"] || raw["common mistake"] || raw.mistake || "";
        if (common) normalized.mistake = common;
        const useCase = raw["Use case"] || raw.Use || raw["use case"] || "";
        if (useCase) normalized.useCase = useCase;
        const entry = {
          key,
          lower,
          display: key,
          meta: normalized,
          category,
          icon: TOOLTIP_ICONS[category] || "",
        };
        state.tooltipKeys.push(key);
        state.tooltipCache.push({ key, lower, category });
        state.tooltipLookup[lower] = entry;
        state.tooltipLookup[key] = entry;
      });
    });
    state.tooltipRegex = null; // rebuild on demand
  } catch (_) {
    state.tooltips = {};
    state.tooltipKeys = [];
    state.tooltipCache = [];
    state.tooltipLookup = {};
    state.tooltipRegex = null;
  }
}

function buildTooltipRegex() {
  if (!state.tooltipKeys.length) {
    state.tooltipRegex = null;
    return;
  }
  const pattern = state.tooltipCache
    .map((entry) => escapeRegex(entry.key))
    .sort((a, b) => b.length - a.length)
    .join("|");
  state.tooltipRegex = pattern ? new RegExp(pattern, "gi") : null;
}

function annotateTooltips(root) {
  if (!state.tooltipKeys.length) return;
  if (!state.tooltipRegex) buildTooltipRegex();
  if (!state.tooltipRegex) return;
  const scope = root || document;
  clearTooltipSpans(scope);
  const codeOnly = !!$("tooltipsCodeOnly")?.checked;
  const selectors = codeOnly
    ? "pre code"
    : "p, li, code, td, th, span, strong, em, h1, h2, h3, h4, h5, h6";
  scope.querySelectorAll(selectors).forEach((element) => annotateElement(element));
}

function clearTooltipSpans(root) {
  (root || document).querySelectorAll(".tt-target").forEach((span) => {
    span.replaceWith(document.createTextNode(span.textContent));
  });
}

function annotateElement(element) {
  const walker = document.createTreeWalker(
    element,
    NodeFilter.SHOW_TEXT,
    {
      acceptNode(node) {
        if (!node.nodeValue || !node.nodeValue.trim()) return NodeFilter.FILTER_REJECT;
        if (node.parentNode && node.parentNode.closest(".tt-target")) return NodeFilter.FILTER_REJECT;
        return NodeFilter.FILTER_ACCEPT;
      },
    },
  );
  const textNodes = [];
  let current;
  while ((current = walker.nextNode())) textNodes.push(current);
  textNodes.forEach(processTextNode);
}

function processTextNode(textNode) {
  const original = textNode.nodeValue;
  const lower = original.toLowerCase();
  const matches = [];
  state.tooltipCache.forEach(({ key, lower: lowerKey }) => {
    let index = lower.indexOf(lowerKey);
    while (index !== -1) {
      matches.push({ start: index, end: index + key.length, key });
      index = lower.indexOf(lowerKey, index + key.length);
    }
  });
  if (!matches.length) return;
  matches.sort((a, b) => a.start - b.start || b.end - a.end);
  const filtered = [];
  let lastEnd = -1;
  matches.forEach((match) => {
    if (match.start >= lastEnd) {
      filtered.push(match);
      lastEnd = match.end;
    }
  });
  if (!filtered.length) return;
  const frag = document.createDocumentFragment();
  let cursor = 0;
  filtered.forEach((match) => {
    if (match.start > cursor) {
      frag.appendChild(document.createTextNode(original.slice(cursor, match.start)));
    }
    const span = create("span", "tt-target");
    span.textContent = original.slice(match.start, match.end);
    span.dataset.ttKey = match.key;
    const entry = resolveTooltipEntry(match.key);
    if (entry?.category) span.dataset.ttCategory = entry.category;
    frag.appendChild(span);
    cursor = match.end;
  });
  if (cursor < original.length) {
    frag.appendChild(document.createTextNode(original.slice(cursor)));
  }
  textNode.parentNode.replaceChild(frag, textNode);
}

function renderTooltipResults(query) {
  const container = $("ttResults");
  if (!container) return;
  container.innerHTML = "";
  if (!state.tooltipKeys.length) {
    container.innerHTML = "<div class=\"tt-item\">Tooltip data not loaded.</div>";
    return;
  }
  let keys;
  const trimmed = (query || "").trim();
  if (!trimmed) {
    container.innerHTML = "<div class=\"tt-item\">Type a term or \"!\" to list entries.</div>";
    updateTooltipDetail(null);
    return;
  } else if (trimmed === "!") {
    keys = state.tooltipCache.slice(0, 1000).map((entry) => entry.key);
  } else {
    const lower = trimmed.toLowerCase();
    keys = state.tooltipCache
      .filter((entry) => entry.lower.includes(lower))
      .slice(0, 1000)
      .map((entry) => entry.key);
  }
  if (!keys.length) {
    container.innerHTML = `<div class="tt-item">No tooltip entries match "${trimmed}".</div>`;
    updateTooltipDetail(null);
    return;
  }
  const frag = document.createDocumentFragment();
  keys.forEach((key) => {
    const entry = resolveTooltipEntry(key);
    if (!entry) return;
    const btn = create("button", `tt-chip tt-target cat-${entry.category || "generic"}`);
    btn.type = "button";
    btn.dataset.key = entry.key;
    btn.dataset.ttKey = entry.key;
    btn.textContent = `${entry.icon || ""} ${entry.display || entry.key}`;
    frag.appendChild(btn);
  });
  container.appendChild(frag);
  updateTooltipDetail(keys[0]);
}

function updateTooltipDetail(key) {
  const detail = $("ttDetail");
  state.activeTooltipKey = key || null;
  if (!detail) return;
  if (!key) {
    detail.innerHTML = "<div class=\"tt-name\">Select a tooltip entry to see details.</div>";
    return;
  }
  const entry = resolveTooltipEntry(key);
  if (!entry) {
    detail.innerHTML = "<div class=\"tt-name\">No details available.</div>";
    return;
  }
  const meta = entry.meta || {};
  let html = `<div class="tt-name">${entry.icon || ""} ${entry.display || entry.key}</div>`;
  if (meta.summary) html += `<div class="tt-sub">${meta.summary}</div>`;
  if (meta.usage) html += `<div class="tt-sub"><strong>Usage:</strong> ${meta.usage}</div>`;
  if (meta.useCase) html += `<div class="tt-sub"><strong>Use case:</strong> ${meta.useCase}</div>`;
  if (meta.mistake) html += `<div class="tt-sub"><strong>Common mistake:</strong> ${meta.mistake}</div>`;
  if (meta.tip) html += `<div class="tt-tip">${meta.tip}</div>`;
  const bestPractice = $("bestPractice")?.checked;
  const notes = meta.notes || [];
  if (notes.length) {
    if (bestPractice) {
      html += "<div class=\"tt-notes\"><div class=\"tt-notes-title\">Best practices</div><ul>"
        + notes.map((item) => `<li>${item}</li>`).join("")
        + "</ul></div>";
    } else {
      html += "<div class=\"tt-notes\"><div class=\"tt-notes-title\">Notes</div><ul>"
        + notes.map((item) => `<li>${item}</li>`).join("")
        + "</ul></div>";
    }
  }
  detail.innerHTML = html;
}

function reannotateAll() {
  annotateTooltips($("chat"));
  annotateTooltips($("docView"));
}

// ---------- Event Wiring ----------
function attachHandlers() {
  $("newChatBtn")?.addEventListener("click", startNewChat);
  $("clearChatBtn")?.addEventListener("click", clearChat);
  $("exportBtn")?.addEventListener("click", exportChat);
  $("send")?.addEventListener("click", handleSend);
  $("stop")?.addEventListener("click", forceStop);
  $("chatList")?.addEventListener("click", async (e) => {
    const btn = e.target.closest(".history-item");
    if (!btn) return;
    const id = btn.dataset.chatId;
    if (!id || id === state.chatId) return;
    await loadChat(id);
    closeDrawer("historyDrawer");
  });
  $("msg")?.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  });
  $("temp")?.addEventListener("input", () => {
    const val = Number($("temp")?.value || 0);
    setText($("tempVal"), val.toFixed(2));
  });
  $("ttResults")?.addEventListener("click", (e) => {
    const btn = e.target.closest(".tt-chip");
    if (!btn) return;
    const key = btn.dataset.key;
    updateTooltipDetail(key);
  });
  $("ttSearch")?.addEventListener("input", debounce((e) => {
    renderTooltipResults(e.target.value.trim());
  }, 180));
  $("tooltipsCodeOnly")?.addEventListener("change", () => reannotateAll());
  $("bestPractice")?.addEventListener("change", () => {
    if (state.activeTooltipKey) updateTooltipDetail(state.activeTooltipKey);
  });
  $("backendSel")?.addEventListener("change", updateBackendVisibility);
  setupCacheListeners();
  bindDrawerTriggers();
  updateBackendVisibility();
}

// ---------- Initialization ----------
async function initialize() {
  applyCachedSettings();
  attachHandlers();
  await Promise.all([fetchChats(), fetchDocs(), loadTooltips()]);
  updatePinsUI();
  renderTooltipResults(""); // show helper message
  updateTooltipDetail(null);
  await updateStatus();
  reannotateAll();
}

window.addEventListener("DOMContentLoaded", initialize);
