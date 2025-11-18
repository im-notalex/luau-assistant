const state = { chatId: null, chats: [], selectedDoc: null, pinnedPath: null, docs: [] };


function el(id){ return document.getElementById(id); }

function setModelTag(){
  fetch('/api/status').then(r=>r.json()).then(d=>{
    el('modelTag').textContent = d.model ? `Model: ${d.model}` : '';
    if (d.context_cap && el('capVal')){ el('capVal').textContent = String(d.context_cap); }
    if (d.api_dump && el('apiStatus')){
    // Attempt ensure if not loaded
    if (d.api_dump && !d.api_dump.loaded){
      fetch('/api/ensure_api_dump', {method:'POST'}).then(()=> fetch('/api/status')).then(r=>r.json()).then(x=>{
        if (x.api_dump && el('apiStatus')){ el('apiStatus').textContent = x.api_dump.loaded ? ('loaded ('+x.api_dump.property_count+' props)') : 'not detected'; }
      }).catch(()=>{});
    }
      el('apiStatus').textContent = d.api_dump.loaded ? ('loaded ('+d.api_dump.property_count+' props)') : 'not detected';
    }
  }).catch(()=>{});
}

function renderMessages(msgs){
  const wrap = el('chat');
  wrap.innerHTML = '';
  const md = window.marked;
  msgs.forEach(m => {
    const div = document.createElement('div');
    div.className = 'msg ' + (m.role === 'user' ? 'user' : 'assistant');
    const hdr = document.createElement('div');
    hdr.className = 'hdr';
    const role = m.role === 'user' ? 'You' : 'assistant';
    const t = m.time ? new Date(m.time).toLocaleTimeString() : '';
    hdr.innerHTML = `<span>${role}</span><span>${t}</span>`;
    let html = m.content;
    try{ html = DOMPurify.sanitize(md.parse(m.content)); }catch{}
    div.appendChild(hdr);
    const body = document.createElement('div');
    body.innerHTML = html;
    div.appendChild(body);
    wrap.appendChild(div);
  });
  wrap.scrollTop = wrap.scrollHeight;
  document.querySelectorAll('pre code').forEach((block)=>{ try{ hljs.highlightElement(block); }catch{} });
  addCopyButtons();

  
}

async function fetchChats(){
  const r = await fetch('/api/chats');
  const data = await r.json();
  state.chats = data.chats || [];
  const sel = el('chatList');
  sel.innerHTML = '';
  state.chats.forEach(c => {
    const opt = document.createElement('option');
    const dt = new Date(c.timestamp);
    opt.value = c.id; opt.textContent = dt.toLocaleString();
    sel.appendChild(opt);
  });
}

async function loadChat(id){
  const r = await fetch(`/api/chats/${encodeURIComponent(id)}`);
  const data = await r.json();
  state.chatId = data.id;
  renderMessages(data.messages || []);
}

async function newChat(){
  const r = await fetch('/api/new_chat', {method:'POST'});
  const data = await r.json();
  state.chatId = data.id;
  await fetchChats();
  await fetchDocs();
  updatePinnedUI();
  el('chatList').value = state.chatId;
  renderMessages([]);
}

function buildPayload(msg){
  return {
    message: msg,
    chat_id: state.chatId,
    temperature: parseFloat(el('temp').value),
    plan_mode: el('plan').checked,
    self_check: el('selfCheck').checked,
    show_thinking: el('showThinking').checked,
    plan_depth: parseInt(el('planDepth').value || '2', 10),
    top_k: valNum('setTopK'),
    top_p: valNum('setTopP'),
    repeat_penalty: valNum('setRep'),
    freq_penalty: valNum('setFreq'),
    retr_k: parseInt((el('setRetrK')?.value || '5'), 10)
  };
}

async function send(){
  const msg = el('msg').value.trim();
  if(!msg) return;
  appendUser(msg);
  const typingId = appendTyping();
  const payload = buildPayload(msg);
  el('send').disabled = true; const prevTxt = el('send').textContent; el('send').textContent = 'Sending...';
  try{
    if (el('stream').checked){
      await sendStream(payload, typingId);
    } else {
      await sendOnce(payload, typingId);
    }
    el('msg').value = '';
  } finally {
    el('send').disabled = false; el('send').textContent = prevTxt;
  }
}

async function sendOnce(payload, typingId){
  const r = await fetch('/api/chat', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(payload)});
  const data = await r.json();
  state.chatId = data.id;
  replaceTyping(typingId, data.messages || []);
  if (payload.show_thinking && data.thinking && data.thinking.steps){
    el('thinking').textContent = JSON.stringify(data.thinking, null, 2);
    el('thinkingPanel').open = true;
  }
  await fetchChats();
  el('chatList').value = state.chatId;
}

async function sendStream(payload, typingId){
  const resp = await fetch('/api/chat_stream', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(payload)});
  const reader = resp.body.getReader();
  const dec = new TextDecoder();
  let buf = '';
  let acc = '';
  while (true){
    const {done, value} = await reader.read();
    if (done) break;
    buf += dec.decode(value, {stream:true});
    let idx;
    while ((idx = buf.indexOf('
')) >= 0){
      const line = buf.slice(0, idx).trim();
      buf = buf.slice(idx+1);
      if (!line) continue;
      try{
        const obj = JSON.parse(line);
        if (obj.type === 'chunk'){
          acc += obj.text || '';
          updateTyping(typingId, acc);
        } else if (obj.type === 'done'){
          state.chatId = obj.id;
          await fetchChats();
          el('chatList').value = state.chatId;
          const refs = (obj.diagnostics && obj.diagnostics.sources) ? obj.diagnostics.sources : extractReferences(acc);
          renderRefs(refs || []);
          if (refs && refs.length) loadSource(refs[0]);
          if (payload.show_thinking){
            el('thinking').textContent = JSON.stringify({steps: (obj.diagnostics?.plan_steps||[])}, null, 2);
            el('thinkingPanel').open = true;
          }
        }
      }catch{}
    }
  }
  if (state.chatId){
    const r = await fetch(`/api/chats/${encodeURIComponent(state.chatId)}`);
    const data = await r.json();
    replaceTyping(typingId, data.messages || []);
  } else {
    replaceTyping(typingId, []);
  }
}

function updateTyping(typingId, text){
  const node = document.getElementById(typingId);
  if (!node) return;
  const md = window.marked;
  const body = `<div class="typing-dots"><span></span><span></span><span></span></div>` +
               `<div>${DOMPurify.sanitize(md.parse(text))}</div>`;
  node.innerHTML = body;
  document.querySelectorAll(`#${typingId} pre code`).forEach((block)=>{ try{ hljs.highlightElement(block); }catch{} });
}

function extractReferences(text){
  const refs = [];
  if (!text) return refs;
  const lines = text.split(/\r?
/);
  let inRefs = false;
  for (const ln of lines){
    if (/^\s*references\s*:?/i.test(ln)){ inRefs = true; continue; }
    if (inRefs){
      const m = ln.match(/^\s*[-*]\s+(.*)$/);
      if (m){ refs.push(m[1].trim()); }
      else if (ln.trim() === ''){ break; }
    }
  }
  return refs;
}

function renderRefs(refs){
  const c = el('refs');
  c.innerHTML='';
  [...new Set(refs)].forEach(p =>{
    const chip = document.createElement('span');
    chip.className='ref';
    chip.textContent = p.length>48? ('�'+p.slice(-48)):p;
    chip.title = p;
    chip.addEventListener('click', ()=> loadSource(p));
    c.appendChild(chip);
  });
}

async function loadSource(path){
  try{
    const q = await fetch('/api/source?path=' + encodeURIComponent(path));
    const data = await q.json();
    el('previewTitle').textContent = data.path || 'Reference Preview';
    const cont = el('previewContent');
    const md = window.marked;
    if (data.kind === 'markdown'){
      cont.innerHTML = DOMPurify.sanitize(md.parse(data.content || ''));
    } else if (data.kind === 'code'){
      cont.innerHTML = `<pre><code class="language-${data.language||'plaintext'}"></code></pre>`;
      cont.querySelector('code').textContent = data.content || '';
      try{ cont.querySelectorAll('pre code').forEach((b)=> hljs.highlightElement(b)); }catch{}
    } else {
      cont.textContent = data.content || '';
    }
  }catch(e){
    el('previewTitle').textContent = 'Sending...';
    el('previewContent').textContent = 'Sending...';
  }
}

function appendUser(text){
  const wrap = el('chat');
  const div = document.createElement('div');
  div.className = 'msg user';
  const hdr = document.createElement('div'); hdr.className='hdr'; hdr.innerHTML = `<span>You</span><span>${new Date().toLocaleTimeString()}</span>`; div.appendChild(hdr);
  const body = document.createElement('div');
  try{ body.innerHTML = DOMPurify.sanitize(marked.parse(text)); }catch{ body.textContent = text; }
  div.appendChild(body);
  wrap.appendChild(div); wrap.scrollTop = wrap.scrollHeight;
}

function appendTyping(){
  const id = 'typing-' + Math.random().toString(36).slice(2);
  const wrap = el('chat');
  const div = document.createElement('div');
  div.className = 'msg assistant typing';
  div.id = id;
  div.innerHTML = '<div class="typing-dots"><span></span><span></span><span></span></div><span>assistant is responding�</span>';
  wrap.appendChild(div); wrap.scrollTop = wrap.scrollHeight;
  return id;
}

function replaceTyping(typingId, messages){
  const node = document.getElementById(typingId);
  if (node){ node.remove(); }
  renderMessages(messages);
}

function addCopyButtons(){
  document.querySelectorAll('.msg pre').forEach(pre => {
    if (pre.querySelector('.copy-btn')) return;
    const btn = document.createElement('button');
    btn.textContent='Copy'; btn.className='copy-btn';
    Object.assign(btn.style,{position:'absolute',right:'8px',top:'8px',fontSize:'12px'});
    btn.addEventListener('click', ()=>{
      const code = pre.querySelector('code');
      if (!code) return;
      navigator.clipboard.writeText(code.innerText||code.textContent||'');
      btn.textContent='Copied!'; setTimeout(()=> btn.textContent='Copy', 1200);
    });
    pre.style.position='relative';
    pre.appendChild(btn);
  });
}

document.addEventListener('keydown', (e)=>{
  if (e.target === el('msg')){
    if (e.key === 'Enter' && !e.shiftKey){ e.preventDefault(); send(); }
  }
});


  if (document.getElementById('pinBtn')){ document.getElementById('pinBtn').addEventListener('click', ()=>{ if(state.selectedDoc){ state.pinnedPath = state.selectedDoc;}); if (typeof updatePinnedUI==='function') updatePinnedUI(); };
  if (document.getElementById('clearPinBtn')){ document.getElementById('clearPinBtn').addEventListener('click', ()=>{ state.pinnedPath=null;}); if (typeof updatePinnedUI==='function') updatePinnedUI(); };
  if (state.chats.length){ el('chatList').value = state.chats[0].id; loadChat(state.chats[0].id); }
});

function valNum(id){
  const e = document.getElementById(id);
  if(!e) return null;
  const v = (e.value||'').trim();
  if (v==='') return null;
  const n = Number(v);
  return Number.isFinite(n)? n : null;
}

async function clearChat(){
  const cid = state.chatId;
  const r = await fetch('/api/clear_chat',{method:'POST',headers:{'Content-Type':'application/json'},body: JSON.stringify({chat_id: cid})});
  const data = await r.json();
  state.chatId = data.id;
  renderMessages([]);
  await fetchChats();
  el('chatList').value = state.chatId;
}

async function exportChat(){
  if (!state.chatId) return;
  const r = await fetch('/api/chats/' + encodeURIComponent(state.chatId));
  const data = await r.json();
  const blob = new Blob([JSON.stringify(data, null, 2)], {type:'application/json'});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = (state.chatId || 'chat') + '.json';
  document.body.appendChild(a); a.click(); a.remove();
}

function resetSettings(){
  el('temp').value = '0.4'; el('tempVal').textContent = 'Sending...';
  el('plan').checked = true;
  el('strict').checked = false;
  el('selfCheck').checked = false;
  el('showThinking').checked = false;
  el('planDepth').value = '2';
  el('minSources').value = '0';
  el('stream').checked = true;
  const advIds = ['setTopK','setTopP','setRep','setFreq'];
  advIds.forEach(id=>{ const e=el(id); if(e) e.value=''; });
  if (el('setRetrK')) el('setRetrK').value = '5';
}





// Docs viewer
async function fetchDocs(){\n  const r = await fetch('/api/docs_list');\n  const data = await r.json();\n  state.docs = data.files || [];\n  renderDocsList(state.docs);\n}\nfunction renderDocsList(files){
  const box = el('docsList');
  const search = (el('docSearch')?.value || '').toLowerCase();
  box.innerHTML = '';
  files.filter(f => !search || f.toLowerCase().includes(search)).slice(0, 500).forEach(p => {
    const div = document.createElement('div');
    div.className = 'doc';
    div.textContent = p.replace(/^Docs\//,'');
    div.title = p;
    div.addEventListener('click', ()=> loadDoc(p));
    box.appendChild(div);
  });
}

async function loadDoc(path){\n  try{\n    const r = await fetch('/api/docs_read?path=' + encodeURIComponent(path));\n    const data = await r.json();\n    const cont = el('docView');\n    const md = window.marked;\n    cont.innerHTML = DOMPurify.sanitize(md.parse(data.content || ''));\n    try{ cont.querySelectorAll('pre code').forEach((b)=> hljs.highlightElement(b)); }catch{}\n  }catch(e){\n    el('docView').textContent = 'Failed to load document.';\n  }\n}\ncatch(e){
    el('docView').textContent = 'Failed to load document.';
  }
}








\nfunction updatePinnedUI(){ const b = el('pinnedBadge'); if(!b) return; b.textContent = state.pinnedPath ? ('Pinned: ' + state.pinnedPath.replace(/^Docs\\\//,'')) : ''; }\n\ndocument.addEventListener('DOMContentLoaded', async ()=>{\n  setModelTag();\n  const tv = el('tempVal'); if (tv) { tv.textContent = el('temp')?.value || tv.textContent; }\n  if (el('temp')) el('temp').addEventListener('input', e=> { const v = (e.target?.value||''); const tv2 = el('tempVal'); if (tv2) tv2.textContent = v; });\n  if (el('send')) el('send').addEventListener('click', send);\n  if (el('msg')) el('msg').addEventListener('keydown', e=>{ if(e.key==='Enter' && !e.shiftKey){ e.preventDefault(); send(); } });\n  if (el('newChatBtn')) el('newChatBtn').addEventListener('click', newChat);\n  if (el('clearChatBtn')) el('clearChatBtn').addEventListener('click', clearChat);\n  if (el('exportBtn')) el('exportBtn').addEventListener('click', exportChat);\n  if (el('chatList')) el('chatList').addEventListener('change', e=> loadChat(e.target.value));\n  if (el('pinBtn')) el('pinBtn').addEventListener('click', ()=>{ if(state.selectedDoc){ state.pinnedPath = state.selectedDoc; const s=el('pinStatus'); if(s) s.textContent='Pinned: ' + state.pinnedPath.replace(/^Docs\\\//,''); updatePinnedUI(); }});\n  if (el('clearPinBtn')) el('clearPinBtn').addEventListener('click', ()=>{ state.pinnedPath=null; const s=el('pinStatus'); if(s) s.textContent=''; updatePinnedUI(); });\n  if (el('docSearch')) el('docSearch').addEventListener('input', ()=> renderDocsList(state.docs));\n  await fetchChats();\n  await fetchDocs();\n  updatePinnedUI();\n  if (state.chats.length){ el('chatList').value = state.chats[0].id; loadChat(state.chats[0].id); }\n});\n
