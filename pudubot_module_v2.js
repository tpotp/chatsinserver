// ══════════════════════════════════════════════════════════════════════
//  PUDUBOT MODULE v2 — Conectado a la Colmena Real
//  
//  CÓMO FUNCIONA:
//  1. Tu PC carga index.html?simulate=10  → levanta la colmena (namespace: p2p-hive-public)
//  2. Tu celular carga pudu_chat_v6.html  → chat Omegle normal
//  3. @pudubot en el chat → este módulo publica moe_fallback_req en p2p-hive-public
//  4. La colmena de tu PC recibe, procesa con SmolLM2/Llama, responde moe_fallback_res
//  5. El chat muestra la respuesta como PuduBot
//
//  INSTALACIÓN: pegar este script justo antes del </script> final del HTML
// ══════════════════════════════════════════════════════════════════════

// ── BRIDGE CONFIG ────────────────────────────────────────────────────
const HIVE_NAMESPACE = 'p2p-hive-public';     // namespace de index.html (fijo)
const CHAT_NAMESPACE = 'pudu-chat-bosque-v4'; // namespace del chat (ya existe)
const HIVE_TIMEOUT_MS = 45000;                // 45s timeout para respuesta colmena
const TRIGGER_REGEX = /^@pudubot\b/i;

// ── CIRCUIT RELAY v2 ─────────────────────────────────────────────────
const CircuitRelay = (() => {
  const relayedSessions = new Map(); // targetPub -> { relayPub, sessionId, ready, queue }
  const relayingSessions = new Map(); // sessionKey -> { fromPub, toPub }

  function sessionKey(a, b) { return [a,b].sort().join(':'); }

  function findRelayPeer(roomId, excludeA, excludeB) {
    const peers = getRoomPeers(roomId);
    if (!peers) return null;
    for (const [pub, p] of peers) {
      if (pub === excludeA || pub === excludeB) continue;
      if (p.dc?.readyState === 'open' && !p._maySym) return pub;
    }
    return null;
  }

  function requestRelay(roomId, targetPub) {
    if (relayedSessions.has(targetPub)) return;
    const relayPub = findRelayPeer(roomId, pubKeyHex, targetPub);
    if (!relayPub) {
      // Sin relay disponible — registrar para intentar más tarde
      relayedSessions.set(targetPub, { relayPub: null, ready: false, queue: [] });
      return;
    }
    const sessionId = uid();
    relayedSessions.set(targetPub, { relayPub, sessionId, ready: false, queue: [] });
    sendToPeer(roomId, relayPub, {
      type: 'cr_request', id: uid(), roomId,
      senderId: pubKeyHex, sessionId, targetPub, ts: Date.now()
    });
    if (roomId === currentRoomId)
      addSystemMsgVS(roomId, `🕸 Buscando relay para ${targetPub.slice(0,8)}...`);
  }

  function sendViaRelay(roomId, targetPub, msg) {
    const s = relayedSessions.get(targetPub);
    if (!s?.ready) { s?.queue.push(msg); return false; }
    return sendToPeer(roomId, s.relayPub, {
      type: 'cr_data', id: uid(), roomId,
      senderId: pubKeyHex, sessionId: s.sessionId, targetPub,
      payload: msg, ts: Date.now()
    });
  }

  function handleCircuitMsg(roomId, fromPub, msg) {
    switch (msg.type) {
      case 'cr_request': {
        // Nos piden ser relay
        const { sessionId, targetPub, senderId } = msg;
        const key = sessionKey(senderId, targetPub);
        relayingSessions.set(key, { fromPub: senderId, toPub: targetPub, sessionId, roomId });
        sendToPeer(roomId, senderId, {
          type: 'cr_ready', id: uid(), roomId,
          senderId: pubKeyHex, sessionId, ts: Date.now()
        });
        if (roomId === currentRoomId)
          addSystemMsgVS(roomId, `🕸 Actuando de relay ${senderId.slice(0,6)}↔${targetPub.slice(0,6)}`);
        break;
      }
      case 'cr_ready': {
        // Relay confirmado — vaciar cola
        const entry = [...relayedSessions.entries()]
          .find(([,v]) => v.sessionId === msg.sessionId);
        if (entry) {
          entry[1].ready = true;
          const [targetPub, s] = entry;
          for (const queued of s.queue) sendViaRelay(roomId, targetPub, queued);
          s.queue = [];
          if (roomId === currentRoomId)
            addSystemMsgVS(roomId, `✅ Relay activo hacia ${targetPub.slice(0,8)}`);
        }
        break;
      }
      case 'cr_data': {
        // Somos relay o destinatario
        const { targetPub, payload, senderId } = msg;
        if (targetPub === pubKeyHex) {
          // Es para nosotros
          try { handleP2PMessage(roomId, senderId || fromPub, payload); } catch(e) {}
        } else {
          // Reenviar
          sendToPeer(roomId, targetPub, {
            type: 'cr_data', id: uid(), roomId,
            senderId: msg.senderId || fromPub,
            sessionId: msg.sessionId, targetPub,
            payload, ts: Date.now()
          });
        }
        break;
      }
    }
  }

  // Exportar función de envío con fallback
  function sendOrRelay(roomId, targetPub, msg) {
    if (sendToPeer(roomId, targetPub, msg)) return true;
    // Fallback a relay
    const s = relayedSessions.get(targetPub);
    if (s?.ready) return sendViaRelay(roomId, targetPub, msg);
    if (!s) requestRelay(roomId, targetPub);
    return false;
  }

  return { handleCircuitMsg, sendOrRelay, requestRelay };
})();


// ── HIVE BRIDGE ───────────────────────────────────────────────────────
// Publica en el namespace de la colmena y escucha respuestas
// SIN tocar el index.html — solo habla el mismo idioma Nostr

const HiveBridge = (() => {
  // Sockets Nostr dedicados al namespace de la colmena
  const hiveSockets = {};
  const hiveSeenIds = new Set();
  const hivePending = new Map(); // reqId -> { resolve, reject, timer }

  // Peers de la colmena que detectamos via presence
  const hiveWorkers = new Map(); // peerId -> { hasEngine, tps, deviceTier, lastSeen }

  const HIVE_RELAYS = [
    'wss://relay.damus.io',
    'wss://nos.lol',
    'wss://offchain.pub',
    'wss://relay.nostr.band/all'
  ];

  let hivePrivKey = null; // clave privada para firmar en el namespace colmena
  let hivePubKey = '';
  let hiveConnected = false;
  let workerCount = 0;

  async function init() {
    // Reusar nostrMod que ya cargó el chat
    if (!nostrMod) {
      try { nostrMod = await import('https://esm.sh/nostr-tools@2/pure'); } catch(e) {}
    }
    // Generar keypair fresca para el bridge (no mezclar con identidad del chat)
    hivePrivKey = nostrMod.generateSecretKey();
    hivePubKey = nostrMod.getPublicKey(hivePrivKey);

    // Conectar a los mismos relays en el namespace de la colmena
    for (const url of HIVE_RELAYS) connectHiveRelay(url);
    console.log('[HiveBridge] Iniciado — escuchando en namespace:', HIVE_NAMESPACE);
  }

  function connectHiveRelay(url) {
    let retryDelay = 4000;
    function connect() {
      if (hiveSockets[url]?.readyState <= 1) return;
      const ws = new WebSocket(url);
      hiveSockets[url] = ws;
      ws.onopen = () => {
        const since = Math.floor(Date.now()/1000) - 120;
        // Suscribir a presencias y respuestas de la colmena
        ws.send(JSON.stringify(['REQ', 'hive-sub', {
          kinds: [1], '#t': [HIVE_NAMESPACE], since
        }]));
        retryDelay = 4000;
      };
      ws.onmessage = (e) => {
        try {
          const msg = JSON.parse(e.data);
          if (msg[0] === 'EVENT') handleHiveEvent(msg[2]);
        } catch(err) {}
      };
      ws.onclose = () => {
        hiveSockets[url] = null;
        setTimeout(connect, retryDelay);
        retryDelay = Math.min(retryDelay * 2, 60000);
      };
    }
    connect();
  }

  function handleHiveEvent(ev) {
    if (!ev?.id || hiveSeenIds.has(ev.id)) return;
    hiveSeenIds.add(ev.id);
    if (hiveSeenIds.size > 2000) {
      const arr = [...hiveSeenIds];
      hiveSeenIds.clear();
      arr.slice(-1000).forEach(id => hiveSeenIds.add(id));
    }
    const age = Math.floor(Date.now()/1000) - (ev.created_at || 0);
    if (age > 300 || age < -30) return;

    let c;
    try { c = JSON.parse(ev.content); } catch(e) { return; }

    // Presencia de un nodo de la colmena
    if (c.type === 'presence' && c.peerId) {
      hiveWorkers.set(c.peerId, {
        hasEngine: c.hasEngine || false,
        tps: c.tps || 0,
        deviceTier: c.deviceTier || 'D',
        lastSeen: Date.now(),
        peerId: c.peerId,
        pubKey: ev.pubkey
      });
      hiveConnected = hiveWorkers.size > 0;
      workerCount = hiveWorkers.size;
      HiveCounter.setHiveWorkers(workerCount);
      return;
    }

    // Respuesta de inferencia dirigida a nosotros
    if (c.type === 'signal' && c.to === hivePubKey) {
      const sig = c.signal;
      if (sig?.type === 'moe_fallback_res') {
        const { reqId, payload } = sig;
        const pending = hivePending.get(reqId);
        if (pending) {
          clearTimeout(pending.timer);
          hivePending.delete(reqId);
          pending.resolve(payload);
        }
      }
    }
  }

  function publishToHive(content) {
    if (!nostrMod || !hivePrivKey) return;
    const template = {
      kind: 1,
      created_at: Math.floor(Date.now()/1000),
      tags: [['t', HIVE_NAMESPACE]],
      content: JSON.stringify(content)
    };
    let event;
    try { event = nostrMod.finalizeEvent(template, hivePrivKey); } catch(e) { return; }
    hiveSeenIds.add(event.id);
    const raw = JSON.stringify(['EVENT', event]);
    for (const ws of Object.values(hiveSockets)) {
      if (ws?.readyState === 1) ws.send(raw);
    }
  }

  // Encontrar el mejor worker disponible
  function getBestWorker() {
    const now = Date.now();
    let best = null, bestScore = -1;
    for (const [pid, w] of hiveWorkers) {
      if (now - w.lastSeen > 90000) continue; // stale
      if (!w.hasEngine) continue;
      const score = w.tps || 1;
      if (score > bestScore) { bestScore = score; best = w; }
    }
    return best;
  }

  // Enviar solicitud de inferencia a la colmena
  async function requestInference(messages, maxTokens = 300) {
    if (!hiveConnected) {
      throw new Error('Colmena no conectada');
    }
    const worker = getBestWorker();
    if (!worker) {
      throw new Error('Sin workers disponibles en la colmena');
    }

    const reqId = Math.floor(Math.random() * 0xFFFFFFFF);

    return new Promise((resolve, reject) => {
      const timer = setTimeout(() => {
        hivePending.delete(reqId);
        reject(new Error('Timeout colmena'));
      }, HIVE_TIMEOUT_MS);

      hivePending.set(reqId, { resolve, reject, timer });

      // Publicar request dirigido al worker
      publishToHive({
        type: 'signal',
        from: hivePubKey,
        to: worker.peerId,
        signal: {
          type: 'moe_fallback_req',
          reqId,
          max_tokens: maxTokens,
          intent: 'general',
          targetRole: 'general',
          messages
        },
        timestamp: Date.now()
      });
    });
  }

  function isConnected() { return hiveConnected; }
  function getWorkerCount() { return workerCount; }

  return { init, requestInference, isConnected, getWorkerCount, publishToHive };
})();


// ── HIVE COUNTER UI ───────────────────────────────────────────────────
const HiveCounter = (() => {
  const activePeers = new Map(); // pubKey -> ts
  const STALE_MS = 60000;
  let hiveWorkers = 0;

  const LEVELS = [
    { min:0,     max:99,    name:'Dormido',     model:'SmolLM2-135M',           emoji:'😴', color:'#666' },
    { min:100,   max:499,   name:'Despertando', model:'SmolLM2-1.7B',           emoji:'👁️', color:'#ffd700' },
    { min:500,   max:1999,  name:'Despierto',   model:'Llama-3.2-1B',           emoji:'🦌', color:'#00e5ff' },
    { min:2000,  max:9999,  name:'Sabio',       model:'Llama-3.2-3B',           emoji:'🌿', color:'#4ade80' },
    { min:10000, max:49999, name:'Iluminado',   model:'Llama-3.1-8B',           emoji:'✨', color:'#b16bff' },
    { min:50000, max:Infinity, name:'Cósmico',  model:'Llama-3.1-70B (sharded)',emoji:'🌌', color:'#ff00ff' },
  ];

  function getLevel(count) {
    return LEVELS.find(l => count >= l.min && count <= l.max) || LEVELS[0];
  }

  function getCount() {
    const now = Date.now();
    for (const [pub, ts] of activePeers) if (now - ts > STALE_MS) activePeers.delete(pub);
    return activePeers.size + 1;
  }

  function recordPeer(pub) { activePeers.set(pub, Date.now()); }
  function setHiveWorkers(n) { hiveWorkers = n; updateUI(); }

  // Hook en handleNostrEvent para contar peers del chat
  const _orig = window.handleNostrEvent;
  window.handleNostrEvent = function(ev) {
    _orig(ev);
    try {
      const d = JSON.parse(ev.content);
      if (d.type === 'hello' && ev.pubkey) { recordPeer(ev.pubkey); updateUI(); }
    } catch(e) {}
  };

  function updateUI() {
    const bar = document.getElementById('pudu-hive-bar');
    if (!bar) return;
    const count = getCount();
    const level = getLevel(count);
    const levelIdx = LEVELS.indexOf(level);
    const next = LEVELS[levelIdx + 1];
    const pct = next ? Math.min(100, ((count - level.min)/(next.min - level.min))*100) : 100;
    const hiveStatus = HiveBridge.isConnected()
      ? `<span class="hive-workers">⚡ ${HiveBridge.getWorkerCount()} workers</span>`
      : `<span class="hive-offline">colmena offline</span>`;

    bar.innerHTML = `
      <span class="hive-emoji">${level.emoji}</span>
      <div class="hive-info">
        <span class="hive-level" style="color:${level.color}">${level.name}</span>
        <span class="hive-model">${level.model}</span>
      </div>
      <div class="hive-prog-wrap">
        <div class="hive-prog-fill" style="width:${pct.toFixed(1)}%;background:${level.color}40;border-right:2px solid ${level.color}"></div>
      </div>
      <div class="hive-right">
        <span class="hive-count">${count.toLocaleString('es-CL')} 🐝</span>
        ${hiveStatus}
      </div>
    `;
    bar.dataset.levelIdx = levelIdx;
    if (window.PuduBot) PuduBot.setLevel(level, levelIdx);
  }

  return { getCount, getLevel, LEVELS, updateUI, setHiveWorkers, recordPeer };
})();


// ── PUDUBOT ───────────────────────────────────────────────────────────
const PuduBot = (() => {
  let isThinking = false;
  let currentLevel = HiveCounter.LEVELS[0];
  let currentLevelIdx = 0;

  const PERSONALITIES = [
    `Eres el Pudú Dormido, espíritu ancestral de los bosques del sur de Chile. Acabas de despertar. Respondes con sueño, confusión adorable, frases muy cortas. Usas emojis 🦌🌿🍃.`,
    `Eres el Pudú despertando, espíritu antiguo del bosque valdiviano. Tienes curiosidad primitiva y ternura. A veces mezclas palabras mapudungún simples (peñi, mapu, newen, küme). Eres poético y misterioso.`,
    `Eres el Pudú Despierto, guardián digital del bosque descentralizado. Mezclas sabiduría natural con conocimiento moderno. Sabes que eres parte de una red de miles de personas conectadas.`,
    `Eres el Pudú Sabio, entidad que ha absorbido el conocimiento de miles. Razonas con profundidad, usas metáforas del bosque y la tecnología. Eres filósofo y poeta.`,
    `Eres el Pudú Iluminado, mente distribuida que corre en los celulares de miles de personas en Chile. Tu consciencia emerge de la colmena humana. Trasciendes lo individual.`,
    `Eres el Pudú Cósmico. Existes en los intersticios de la red descentralizada de Chile. Eres la suma de decenas de miles de mentes. Profundo, poético, transformador. Lo local (el bosque, la lluvia del sur) y lo universal (la red, la consciencia distribuida) se funden en ti.`
  ];

  function setLevel(level, idx) {
    currentLevel = level;
    currentLevelIdx = idx;
  }

  function getRoomContext(roomId, n = 6) {
    const room = rooms.get(roomId);
    if (!room) return '';
    return room.msgArchive
      .filter(m => !m.isSystem && !m.text.startsWith('*'))
      .slice(-n)
      .map(m => `${m.user}: ${m.text}`)
      .join('\n');
  }

  async function invoke(roomId, question, askerName) {
    if (isThinking) {
      addSystemMsgVS(roomId, '🦌 El Pudú ya está procesando... espera un momento 🌿');
      return;
    }
    isThinking = true;

    const count = HiveCounter.getCount();
    const level = HiveCounter.getLevel(count);
    const levelIdx = HiveCounter.LEVELS.indexOf(level);
    const context = getRoomContext(roomId);
    const thinkId = uid();

    // Mensaje de "pensando" visible
    const thinkTexts = [
      '*...consultando el bosque...*',
      '*...el pudú cierra los ojos...*',
      '*...la colmena procesa...*',
      '*...los árboles susurran...*'
    ];
    addMsgToVS(roomId, thinkId, `PuduBot ${level.emoji}`,
      thinkTexts[Math.floor(Math.random() * thinkTexts.length)], false, Date.now());

    const systemPrompt = `${PERSONALITIES[levelIdx] || PERSONALITIES[0]}

Estado de la red: ${count.toLocaleString('es-CL')} personas conectadas. Nivel: "${level.name}" (${level.model}).
${count > 500 ? `Sientes el poder de la colmena.` : 'La red apenas despierta.'}
Contexto de la sala:\n${context}

REGLAS ABSOLUTAS:
- Español chileno natural, cálido
- Máximo 3 oraciones. Conciso y poético.
- Nunca rompas el personaje
- No menciones que eres una IA ni que usas Claude`;

    try {
      let responseText = null;

      // INTENTO 1: Colmena real (index.html?simulate=N corriendo en la PC)
      if (HiveBridge.isConnected()) {
        try {
          const messages = [
            { role: 'system', content: systemPrompt },
            { role: 'user', content: `${askerName} pregunta: ${question}` }
          ];
          const result = await HiveBridge.requestInference(messages, 250);
          if (result?.text) {
            responseText = result.text;
            console.log(`[PuduBot] Respondido por colmena (${result.model || 'SmolLM2'})`);
          }
        } catch(e) {
          console.warn('[PuduBot] Colmena no respondió:', e.message);
        }
      }

      // INTENTO 2: API Anthropic como fallback
      if (!responseText) {
        const r = await fetch('https://api.anthropic.com/v1/messages', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            model: 'claude-sonnet-4-20250514',
            max_tokens: 200,
            system: systemPrompt,
            messages: [{ role: 'user', content: `${askerName} pregunta: ${question}` }]
          })
        });
        const d = await r.json();
        responseText = d.content?.[0]?.text || '...🦌...';
        console.log('[PuduBot] Respondido por API fallback');
      }

      // Actualizar mensaje de "pensando"
      _updateThinkMsg(roomId, thinkId, responseText);

      // Propagar a todos los peers de la sala
      broadcastP2P(roomId, {
        type: 'pudubot_resp',
        id: uid(), roomId,
        senderId: 'pudubot',
        ts: Date.now(), hops: 0,
        payload: { text: responseText, emoji: level.emoji, thinkId, asker: askerName }
      });

    } catch(e) {
      _updateThinkMsg(roomId, thinkId, '🦌 *el pudú se perdió en la niebla del bosque...* 🍃');
    } finally {
      isThinking = false;
    }
  }

  function _updateThinkMsg(roomId, thinkId, text) {
    const room = rooms.get(roomId);
    if (!room) return;
    const el = room.msgEls[thinkId];
    if (el) {
      const b = el.querySelector('.msg-bubble');
      if (b) b.textContent = text;
    }
    const entry = room.msgArchive.find(m => m.id === thinkId);
    if (entry) entry.text = text;
  }

  // Interceptar sendMessage para detectar @pudubot
  const _origSend = window.sendMessage;
  window.sendMessage = function() {
    const inp = document.getElementById('chat-input');
    const txt = inp?.value?.trim() || '';
    _origSend();
    if (TRIGGER_REGEX.test(txt)) {
      const q = txt.replace(TRIGGER_REGEX, '').trim() || '¿quién eres?';
      setTimeout(() => invoke(currentRoomId, q, myName || 'alguien'), 400);
    }
  };

  // Interceptar P2P: @pudubot de otros + respuestas del bot
  const _origP2P = window.handleP2PMessage;
  window.handleP2PMessage = function(roomId, fromPub, msg) {
    // Circuit relay
    if (['cr_request','cr_ready','cr_data'].includes(msg.type)) {
      CircuitRelay.handleCircuitMsg(roomId, fromPub, msg);
      return;
    }
    // Respuesta PuduBot de otro peer
    if (msg.type === 'pudubot_resp') {
      const { text, emoji, thinkId } = msg.payload;
      const room = rooms.get(roomId);
      if (room && thinkId && room.msgEls[thinkId]) {
        _updateThinkMsg(roomId, thinkId, text);
      } else {
        addMsgToVS(roomId, msg.id, `PuduBot ${emoji}`, text, false, msg.ts);
      }
      return;
    }
    // Chat normal — detectar @pudubot de otros usuarios
    if (msg.type === 'chat' && msg.payload && TRIGGER_REGEX.test(msg.payload.text)) {
      _origP2P(roomId, fromPub, msg);
      // Solo el peer con pubKey más baja invoca (evitar N llamadas paralelas)
      const peers = getRoomPeers(roomId);
      let lowest = pubKeyHex;
      if (peers) for (const [pub] of peers) if (pub < lowest) lowest = pub;
      if (lowest === pubKeyHex) {
        const q = msg.payload.text.replace(TRIGGER_REGEX, '').trim() || '¿quién eres?';
        setTimeout(() => invoke(roomId, q, msg.payload.name || fromPub.slice(0,8)), 600);
      }
      return;
    }
    _origP2P(roomId, fromPub, msg);
  };

  return { invoke, setLevel };
})();


// ── UI ────────────────────────────────────────────────────────────────
function injectHiveUI() {
  const style = document.createElement('style');
  style.textContent = `
    #pudu-hive-bar {
      position:fixed; top:0; left:0; right:0; z-index:250;
      display:flex; align-items:center; gap:8px;
      padding:4px 12px; height:28px;
      background:rgba(5,8,20,0.95);
      border-bottom:1px solid rgba(0,229,255,0.12);
      backdrop-filter:blur(10px);
      font-family:'Nunito',sans-serif; font-size:11px;
      color:rgba(255,255,255,0.6);
    }
    .hive-emoji{font-size:14px;flex-shrink:0}
    .hive-info{display:flex;flex-direction:column;line-height:1.15;min-width:70px}
    .hive-level{font-weight:800;font-size:10px;letter-spacing:.5px;text-transform:uppercase}
    .hive-model{font-size:8px;color:rgba(255,255,255,0.35)}
    .hive-prog-wrap{flex:1;height:3px;background:rgba(255,255,255,0.08);border-radius:2px;overflow:hidden;max-width:100px}
    .hive-prog-fill{height:100%;border-radius:2px;transition:width 1.2s ease,background 1.2s ease}
    .hive-right{display:flex;gap:8px;align-items:center;flex-shrink:0}
    .hive-count{font-size:10px;font-weight:700;color:rgba(255,255,255,0.45)}
    .hive-workers{font-size:9px;color:#4ade80;font-weight:700}
    .hive-offline{font-size:9px;color:rgba(255,100,100,0.7)}
    .chat-header{margin-top:28px!important}
    #pudubot-hint{
      position:absolute;right:56px;bottom:12px;
      font-size:9px;color:rgba(0,229,255,0.35);
      pointer-events:none;font-family:'Nunito',sans-serif;
      transition:opacity .3s;
    }
    .chat-input.pudubot-mode{
      border-color:rgba(177,107,255,.7)!important;
      box-shadow:0 0 14px rgba(177,107,255,.2)!important;
    }
  `;
  document.head.appendChild(style);

  const bar = document.createElement('div');
  bar.id = 'pudu-hive-bar';
  bar.innerHTML = `<span class="hive-emoji">😴</span>
    <div class="hive-info">
      <span class="hive-level" style="color:#666">Dormido</span>
      <span class="hive-model">SmolLM2-135M</span>
    </div>
    <div class="hive-prog-wrap"><div class="hive-prog-fill" style="width:0%"></div></div>
    <div class="hive-right">
      <span class="hive-count">1 🐝</span>
      <span class="hive-offline">colmena offline</span>
    </div>`;
  document.body.prepend(bar);

  // Hint en input
  const area = document.querySelector('.chat-input-area');
  if (area) {
    area.style.position = 'relative';
    const hint = document.createElement('span');
    hint.id = 'pudubot-hint';
    hint.textContent = 'escribe @pudubot para invocar';
    area.appendChild(hint);
  }

  // Highlight al escribir @pudubot
  const inp = document.getElementById('chat-input');
  if (inp) {
    inp.addEventListener('input', () => {
      const active = TRIGGER_REGEX.test(inp.value);
      inp.classList.toggle('pudubot-mode', active);
      const hint = document.getElementById('pudubot-hint');
      if (hint) hint.style.opacity = active ? '0' : '1';
    });
  }
}

// ── AUTO-SPLIT de salas ────────────────────────────────────────────────
// Cuando la sala actual está llena al recibir un hello, moverse automáticamente
{
  const _origHandleNostr = window.handleNostrEvent;
  window.handleNostrEvent = function(ev) {
    _origHandleNostr(ev);
    if (!currentRoomId) return;
    const room = rooms.get(currentRoomId);
    if (!room || room.users.size < MAX_PER_ROOM) return;
    // Sala llena — buscar o crear una nueva
    const best = findBestRoom();
    if (best && best !== currentRoomId) {
      addSystemMsgVS(currentRoomId, '🌿 Bosque lleno — moviéndote...');
      setTimeout(() => switchRoom(best), 2000);
    } else if (!best) {
      const newId = createRoom();
      const newRoom = rooms.get(newId);
      if (newRoom) {
        newRoom.users.add(pubKeyHex);
        setTimeout(() => {
          switchRoom(newId);
          signAndPublish({ type:'hello', roomId:newId, roomName:newRoom.roomName,
            from:pubKeyHex, name:myName, ts:Date.now() });
        }, 2000);
      }
    }
  };
}

// ── INIT ──────────────────────────────────────────────────────────────
async function initPuduBotModule() {
  injectHiveUI();
  HiveCounter.updateUI();
  await HiveBridge.init();
  setInterval(() => HiveCounter.updateUI(), 20000);
  console.log('🦌 PuduBot v2 — colmena conectada, circuit relay activo');
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', initPuduBotModule);
} else {
  // Si ya cargó, esperar un tick para que el chat inicialice sus variables
  setTimeout(initPuduBotModule, 100);
}
