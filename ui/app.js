const state = {
  chatId: null,
  chats: [],
  docs: [],
  selectedDoc: null,
  pinnedPath: null,
};

function el(id) {
  return document.getElementById(id);
}

function getInputValue(id) {
  const node = el(id);
  return node ? (node.value || "").trim() : "";
}

function getNumericValue(id) {
  const value = getInputValue(id);
  if (!value) {
    return null;
  }
  const num = Number(value);
  return Number.isFinite(num) ? num : null;
}

async function setModelTag() {
  try {
    const res = await fetch("/api/status");
    const data = await res.json();
    const tag = el("modelTag");
    if (tag) {
      tag.textContent = data.model ? `Model: ${data.model}` : "";
    }
    if (data.context_cap && el("capVal")) {
      el("capVal").textContent = String(data.context_cap);
    }
    if (el("apiStatus")) {
      if (!data.api_dump || !data.api_dump.loaded) {
        try {
          await fetch("/api/ensure_api_dump", { method: "POST" });
          const follow = await fetch("/api/status");
          data.api_dump = (await follow.json()).api_dump;
        } catch {
          /* ignore */
        }
      }
      const info = data.api_dump;
      el("apiStatus").textContent = info && info.loaded ? `loaded (${info.property_count} props)` : "not detected";
    }
  } catch {
    /* best effort only */
  }
}

function sanitizeMarkdown(content) {
  try {
    const parsed = window.marked ? window.marked.parse(content || "") : content || "";
    return window.DOMPurify ? window.DOMPurify.sanitize(parsed) : parsed;
  } catch {
    return content || "";
  }
}

function highlightAll(root) {
  try {
    (root || document).querySelectorAll("pre code").forEach((block) => {
      if (window.hljs) {
        window.hljs.highlightElement(block);
      }
    });
  } catch {
    /* optional */
  }
}

function renderMessages(messages) {
  const wrap = el("chat");
  if (!wrap) return;
  wrap.innerHTML = "";
  (messages || []).forEach((msg) => {
    const container = document.createElement("div");
    container.className = `msg ${msg.role === "user" ? "user" : "assistant"}`;

    const header = document.createElement("div");
    header.className = "hdr";
    const when = msg.time ? new Date(msg.time).toLocaleTimeString() : "";
    header.innerHTML = `<span>${msg.role === "user" ? "You" : "assistant"}</span><span>${when}</span>`;
    container.appendChild(header);

    const body = document.createElement("div");
    body.innerHTML = sanitizeMarkdown(msg.content);
    container.appendChild(body);

    wrap.appendChild(container);
  });
  highlightAll(wrap);
  addCopyButtons();
  wrap.scrollTop = wrap.scrollHeight;
}

async function fetchChats() {
  try {
    const res = await fetch("/api/chats");
    const data = await res.json();
    state.chats = data.chats || [];
    const sel = el("chatList");
    if (sel) {
      sel.innerHTML = "";
      state.chats.forEach((chat) => {
        const opt = document.createElement("option");
        opt.value = chat.id;
        opt.textContent = new Date(chat.timestamp).toLocaleString();
        sel.appendChild(opt);
      });
    }
  } catch {
    state.chats = [];
  }
}

async function loadChat(id) {
  if (!id) return;
  try {
    const res = await fetch(`/api/chats/${encodeURIComponent(id)}`);
    const data = await res.json();
    state.chatId = data.id;
    renderMessages(data.messages || []);
  } catch {
    /* ignore */
  }
}

async function newChat() {
  try {
    const res = await fetch("/api/new_chat", { method: "POST" });
    const data = await res.json();
    state.chatId = data.id;
    await fetchChats();
    await fetchDocs();
    updatePinnedUI();
    if (el("chatList")) {
      el("chatList").value = state.chatId;
    }
    renderMessages([]);
  } catch {
    /* ignore */
  }
}

function collectBackendOptions(payload) {
  const backend = getInputValue("backendSel");
  if (backend) payload.backend = backend;
  const openaiKey = getInputValue("openaiKey");
  if (openaiKey) payload.openai_key = openaiKey;
  const orKey = getInputValue("openrouterKey");
  if (orKey) payload.openrouter_key = orKey;
  const orModel = getInputValue("openrouterModel");
  if (orModel) payload.openrouter_model = orModel;
  const compatBase = getInputValue("openaiCompatBase");
  const compatKey = getInputValue("openaiCompatKey");
  const compatModel = getInputValue("openaiCompatModel");
  if (compatBase) payload.openai_compat_base_url = compatBase;
  if (compatKey) payload.openai_compat_key = compatKey;
  if (compatModel) payload.openai_compat_model = compatModel;
}

function buildPayload(message) {
  const payload = {
    message,
    chat_id: state.chatId,
    temperature: parseFloat(el("temp")?.value || "0.4") || 0.4,
    plan_mode: !!el("plan")?.checked,
    self_check: !!el("selfCheck")?.checked,
    show_thinking: !!el("showThinking")?.checked,
    plan_depth: parseInt(el("planDepth")?.value || "2", 10) || 2,
    top_k: getNumericValue("setTopK"),
    top_p: getNumericValue("setTopP"),
    repeat_penalty: getNumericValue("setRep"),
    freq_penalty: getNumericValue("setFreq"),
    retr_k: parseInt(el("setRetrK")?.value || "5", 10) || 5,
    pinned_path: state.pinnedPath || null,
    pinned_paths: state.pinnedPath ? [state.pinnedPath] : undefined,
    pins_only: !!el("pinsOnly")?.checked,
  };
  collectBackendOptions(payload);
  return payload;
}

function appendUser(content) {
  const wrap = el("chat");
  if (!wrap) return;
  const container = document.createElement("div");
  container.className = "msg user";
  const header = document.createElement("div");
  header.className = "hdr";
  header.innerHTML = `<span>You</span><span>${new Date().toLocaleTimeString()}</span>`;
  container.appendChild(header);
  const body = document.createElement("div");
  body.innerHTML = sanitizeMarkdown(content);
  container.appendChild(body);
  wrap.appendChild(container);
  wrap.scrollTop = wrap.scrollHeight;
}

function appendTyping() {
  const wrap = el("chat");
  if (!wrap) return "";
  const id = `typing-${Math.random().toString(36).slice(2)}`;
  const node = document.createElement("div");
  node.className = "msg assistant typing";
  node.id = id;
  node.innerHTML = '<div class="typing-dots"><span></span><span></span><span></span></div><span>assistant is responding…</span>';
  wrap.appendChild(node);
  wrap.scrollTop = wrap.scrollHeight;
  return id;
}

function replaceTyping(typingId, messages) {
  const node = document.getElementById(typingId);
  if (node) node.remove();
  renderMessages(messages);
}

function updateTyping(typingId, text) {
  const node = document.getElementById(typingId);
  if (!node) return;
  node.innerHTML = `<div class="typing-dots"><span></span><span></span><span></span></div><div>${sanitizeMarkdown(text)}</div>`;
  highlightAll(node);
}

function addCopyButtons() {
  document.querySelectorAll(".msg pre").forEach((pre) => {
    if (pre.querySelector(".copy-btn")) return;
    const btn = document.createElement("button");
    btn.className = "copy-btn";
    btn.textContent = "Copy";
    Object.assign(btn.style, { position: "absolute", right: "8px", top: "8px", fontSize: "12px" });
    btn.addEventListener("click", () => {
      const code = pre.querySelector("code");
      if (!code) return;
      navigator.clipboard.writeText(code.innerText || code.textContent || "");
      btn.textContent = "Copied!";
      setTimeout(() => {
        btn.textContent = "Copy";
      }, 1200);
    });
    pre.style.position = "relative";
    pre.appendChild(btn);
  });
}

async function sendOnce(payload, typingId) {
  const res = await fetch("/api/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  const data = await res.json();
  state.chatId = data.id;
  replaceTyping(typingId, data.messages || []);
  if (payload.show_thinking && data.thinking) {
    const tgt = el("thinking");
    if (tgt) {
      tgt.textContent = JSON.stringify(data.thinking, null, 2);
      el("thinkingPanel")?.setAttribute("open", "open");
    }
  }
  await fetchChats();
  if (el("chatList")) {
    el("chatList").value = state.chatId;
  }
}

function extractReferences(text) {
  const refs = [];
  if (!text) return refs;
  const lines = text.split(/\r?\n/);
  let inRefs = false;
  for (const line of lines) {
    if (/^\s*references\s*:?\s*$/i.test(line)) {
      inRefs = true;
      continue;
    }
    if (!inRefs) continue;
    if (!line.trim()) break;
    const match = line.match(/^\s*[-*]\s+(.*)$/);
    if (match) refs.push(match[1].trim());
  }
  return refs;
}

function renderRefs(refs) {
  const box = el("refs");
  if (!box) return;
  box.innerHTML = "";
  [...new Set(refs || [])].forEach((path) => {
    const chip = document.createElement("span");
    chip.className = "ref";
    chip.textContent = path.length > 48 ? `…${path.slice(-48)}` : path;
    chip.title = path;
    chip.addEventListener("click", () => loadSource(path));
    box.appendChild(chip);
  });
}

async function loadSource(path) {
  if (!path) return;
  try {
    const res = await fetch(`/api/source?path=${encodeURIComponent(path)}`);
    const data = await res.json();
    if (el("previewTitle")) el("previewTitle").textContent = data.path || "Reference Preview";
    const container = el("previewContent");
    if (!container) return;
    if (data.kind === "markdown") {
      container.innerHTML = sanitizeMarkdown(data.content || "");
      highlightAll(container);
    } else if (data.kind === "code") {
      container.innerHTML = `<pre><code class="language-${data.language || "plaintext"}"></code></pre>`;
      const code = container.querySelector("code");
      if (code) code.textContent = data.content || "";
      highlightAll(container);
    } else {
      container.textContent = data.content || "";
    }
  } catch {
    if (el("previewTitle")) el("previewTitle").textContent = "Preview unavailable";
    if (el("previewContent")) el("previewContent").textContent = "Failed to load document.";
  }
}

async function sendStream(payload, typingId) {
  const res = await fetch("/api/chat_stream", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!res.body) {
    await sendOnce(payload, typingId);
    return;
  }
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let acc = "";
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    let idx;
    while ((idx = buffer.indexOf("\n")) >= 0) {
      const line = buffer.slice(0, idx).trim();
      buffer = buffer.slice(idx + 1);
      if (!line) continue;
      let payloadObj;
      try {
        payloadObj = JSON.parse(line);
      } catch {
        continue;
      }
      if (payloadObj.type === "chunk") {
        acc += payloadObj.text || "";
        updateTyping(typingId, acc);
      } else if (payloadObj.type === "done") {
        state.chatId = payloadObj.id;
        await fetchChats();
        if (el("chatList")) {
          el("chatList").value = state.chatId;
        }
        const refs = payloadObj.diagnostics?.sources || extractReferences(acc);
        renderRefs(refs);
        if (refs && refs.length) {
          loadSource(refs[0]);
        }
        if (payload.show_thinking) {
          const thinking = payloadObj.diagnostics?.plan_steps || [];
          const node = el("thinking");
          if (node) {
            node.textContent = JSON.stringify({ steps: thinking }, null, 2);
            el("thinkingPanel")?.setAttribute("open", "open");
          }
        }
      }
    }
  }
  if (buffer.trim()) {
    try {
      const obj = JSON.parse(buffer.trim());
      if (obj.type === "done") {
        state.chatId = obj.id;
      }
    } catch {
      /* ignore trailing fragment */
    }
  }
  if (state.chatId) {
    const follow = await fetch(`/api/chats/${encodeURIComponent(state.chatId)}`);
    const data = await follow.json();
    replaceTyping(typingId, data.messages || []);
  } else {
    replaceTyping(typingId, []);
  }
}

async function send() {
  const textarea = el("msg");
  if (!textarea) return;
  const message = textarea.value.trim();
  if (!message) return;
  appendUser(message);
  const typingId = appendTyping();
  const payload = buildPayload(message);
  textarea.value = "";
  const sendButton = el("send");
  const prevText = sendButton?.textContent;
  if (sendButton) {
    sendButton.disabled = true;
    sendButton.textContent = "Sending...";
  }
  try {
    if (el("stream")?.checked) {
      await sendStream(payload, typingId);
    } else {
      await sendOnce(payload, typingId);
    }
  } finally {
    if (sendButton) {
      sendButton.disabled = false;
      sendButton.textContent = prevText || "Send";
    }
  }
}

async function clearChat() {
  if (!state.chatId) return;
  const res = await fetch("/api/clear_chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ chat_id: state.chatId }),
  });
  const data = await res.json();
  state.chatId = data.id;
  renderMessages([]);
  await fetchChats();
  if (el("chatList")) {
    el("chatList").value = state.chatId;
  }
}

async function exportChat() {
  if (!state.chatId) return;
  const res = await fetch(`/api/chats/${encodeURIComponent(state.chatId)}`);
  const data = await res.json();
  const blob = new Blob([JSON.stringify(data, null, 2)], { type: "application/json" });
  const anchor = document.createElement("a");
  anchor.href = URL.createObjectURL(blob);
  anchor.download = `${state.chatId}.json`;
  document.body.appendChild(anchor);
  anchor.click();
  anchor.remove();
}

function resetSettings() {
  const defaults = {
    temp: "0.4",
    plan: true,
    selfCheck: false,
    showThinking: false,
    planDepth: "2",
    setTopK: "",
    setTopP: "",
    setRep: "",
    setFreq: "",
    setRetrK: "5",
    stream: true,
  };
  Object.entries(defaults).forEach(([id, value]) => {
    const node = el(id);
    if (!node) return;
    if (typeof value === "boolean") {
      node.checked = value;
    } else {
      node.value = value;
    }
  });
  if (el("tempVal")) el("tempVal").textContent = defaults.temp;
}

async function fetchDocs() {
  try {
    const res = await fetch("/api/docs_list");
    const data = await res.json();
    state.docs = data.files || [];
    renderDocsList(state.docs);
  } catch {
    state.docs = [];
    renderDocsList([]);
  }
}

function renderDocsList(files) {
  const list = el("docsList");
  if (!list) return;
  const searchTerm = (el("docSearch")?.value || "").toLowerCase();
  list.innerHTML = "";
  files
    .filter((file) => !searchTerm || file.toLowerCase().includes(searchTerm))
    .slice(0, 500)
    .forEach((path) => {
      const row = document.createElement("div");
      row.className = "doc";
      if (state.selectedDoc === path) {
        row.classList.add("active");
      }
      row.textContent = path.replace(/^Docs\//, "");
      row.title = path;
      row.addEventListener("click", () => {
        state.selectedDoc = path;
        loadDoc(path);
        renderDocsList(files);
      });
      list.appendChild(row);
    });
}

async function loadDoc(path) {
  if (!path) return;
  state.selectedDoc = path;
  try {
    const res = await fetch(`/api/docs_read?path=${encodeURIComponent(path)}`);
    const data = await res.json();
    const view = el("docView");
    if (view) {
      view.innerHTML = sanitizeMarkdown(data.content || "");
      highlightAll(view);
    }
  } catch {
    const view = el("docView");
    if (view) view.textContent = "Failed to load document.";
  }
}

function updatePinnedUI() {
  const status = el("pinStatus");
  const badge = el("pinnedBadge");
  const text = state.pinnedPath ? `Pinned: ${state.pinnedPath.replace(/^Docs\//, "")}` : "";
  if (status) status.textContent = text;
  if (badge) badge.textContent = text;
}

function updateBackendVisibility() {
  const backend = (getInputValue("backendSel") || "").toLowerCase();
  const toggle = (id, show) => {
    const node = el(id);
    if (node) node.style.display = show ? "" : "none";
  };
  toggle("openaiKeyRow", backend === "openai");
  toggle("openaiModelRow", backend === "openai");
  toggle("orKeyRow", backend === "openrouter");
  toggle("orModelRow", backend === "openrouter");
  toggle("compatBaseRow", backend === "openai_compat");
  toggle("compatKeyRow", backend === "openai_compat");
  toggle("compatModelRow", backend === "openai_compat");
}

function bindUI() {
  setModelTag();
  if (el("temp") && el("tempVal")) {
    el("tempVal").textContent = el("temp").value;
    el("temp").addEventListener("input", (e) => {
      el("tempVal").textContent = e.target.value || "0.4";
    });
  }
  el("send")?.addEventListener("click", send);
  el("msg")?.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      send();
    }
  });
  el("newChatBtn")?.addEventListener("click", newChat);
  el("clearChatBtn")?.addEventListener("click", clearChat);
  el("exportBtn")?.addEventListener("click", exportChat);
  el("resetSettingsBtn")?.addEventListener("click", resetSettings);
  el("chatList")?.addEventListener("change", (e) => loadChat(e.target.value));
  el("pinBtn")?.addEventListener("click", () => {
    if (state.selectedDoc) {
      state.pinnedPath = state.selectedDoc;
      updatePinnedUI();
    }
  });
  el("clearPinBtn")?.addEventListener("click", () => {
    state.pinnedPath = null;
    updatePinnedUI();
  });
  el("docSearch")?.addEventListener("input", () => renderDocsList(state.docs));
  el("backendSel")?.addEventListener("change", updateBackendVisibility);
  updateBackendVisibility();
}

document.addEventListener("DOMContentLoaded", async () => {
  bindUI();
  await fetchChats();
  await fetchDocs();
  updatePinnedUI();
  if (state.chats.length && el("chatList")) {
    el("chatList").value = state.chats[0].id;
    loadChat(state.chats[0].id);
  }
});
