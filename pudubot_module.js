// ══════════════════════════════════════════════════════════════════════
//  PUDUBOT MODULE v1 — Circuit Relay + Hive Intelligence + Level System
//  Integrar pegando este <script> justo ANTES del </script> final del HTML
// ══════════════════════════════════════════════════════════════════════

// ── CIRCUIT RELAY v2 ─────────────────────────────────────────────────
// Cuando dos peers tienen NAT simétrico (flaggeados con _maySym),
// un tercer peer bien conectado actúa de puente retransmitiendo sus
// DataChannel frames. Sin servidor. Sin costo.

const CircuitRelay = (() => {
  // Map: targetPub -> { relayPub, channel }
  const relayedChannels = new Map();
  // Map: sessionKey -> { fromPub, toPub, dc }   (cuando somos el relay)
  const relayingSessions = new Map();

  function sessionKey(a, b) {
    return [a, b].sort().join(':');
  }

  // Encuentra el peer mejor conectado para usarlo de relay
  function findRelay(roomId, excludeA, excludeB) {
    const peers = getRoomPeers(roomId);
    if (!peers) return null;
    for (const [pub, p] of peers) {
      if (pub === excludeA || pub === excludeB) continue;
      if (p.dc && p.dc.readyState === 'open' && !p._maySym) return pub;
    }
    return null;
  }

  // Solicitar a un peer que haga de relay entre nosotros y targetPub
  function requestRelay(roomId, targetPub) {
    if (relayedChannels.has(targetPub)) return; // ya hay uno
    const relayPub = findRelay(roomId, pubKeyHex, targetPub);
    if (!relayPub) return;

    const sessionId = uid();
    sendToPeer(roomId, relayPub, {
      type: 'relay_request',
      id: uid(),
      roomId,
      senderId: pubKeyHex,
      sessionId,
      targetPub,
      ts: Date.now()
    });

    relayedChannels.set(targetPub, { relayPub, sessionId, pendingQueue: [] });
    addSystemMsgVS(roomId, `🕸 Conectando via relay con ${targetPub.slice(0, 8)}...`);
  }

  // Enviar mensaje a través del relay
  function sendViaRelay(roomId, targetPub, msg) {
    const entry = relayedChannels.get(targetPub);
    if (!entry) return false;
    const { relayPub, sessionId } = entry;
    return sendToPeer(roomId, relayPub, {
      type: 'relay_data',
      id: uid(),
      roomId,
      senderId: pubKeyHex,
      sessionId,
      targetPub,
      payload: msg,
      ts: Date.now()
    });
  }

  // Manejar mensajes relay entrantes (agregado a handleP2PMessage)
  function handleRelayMsg(roomId, fromPub, msg) {
    switch (msg.type) {
      case 'relay_request': {
        // Nos piden ser relay entre fromPub y msg.targetPub
        const { sessionId, targetPub } = msg;
        const key = sessionKey(fromPub, targetPub);
        relayingSessions.set(key, { fromPub, toPub: targetPub, sessionId });
        // Confirmamos a ambos lados
        sendToPeer(roomId, fromPub, { type: 'relay_ready', id: uid(), roomId, senderId: pubKeyHex, sessionId, ts: Date.now() });
        addSystemMsgVS(roomId, `🕸 Actuando de relay entre ${fromPub.slice(0,6)} ↔ ${targetPub.slice(0,6)}`);
        break;
      }
      case 'relay_ready': {
        // El relay confirmó — marcar canal como activo
        const entry = [...relayedChannels.entries()].find(([, v]) => v.sessionId === msg.sessionId);
        if (entry) {
          entry[1].ready = true;
          // Vaciar cola pendiente
          for (const queued of entry[1].pendingQueue || []) {
            sendViaRelay(roomId, entry[0], queued);
          }
          entry[1].pendingQueue = [];
        }
        break;
      }
      case 'relay_data': {
        // Somos el relay — reenviar al destino
        const { targetPub, payload, sessionId } = msg;
        if (targetPub === pubKeyHex) {
          // El mensaje es para nosotros, procesarlo normalmente
          try { handleP2PMessage(roomId, fromPub, payload); } catch (e) {}
        } else {
          // Reenviar
          sendToPeer(roomId, targetPub, {
            type: 'relay_data',
            id: uid(),
            roomId,
            senderId: fromPub,
            sessionId,
            targetPub,
            payload,
            ts: Date.now()
          });
        }
        break;
      }
    }
  }

  // Hook en scheduleReconnect: si falló y _maySym, intentar relay
  const _origSchedule = window.scheduleReconnect;
  // Parchamos después de que todo cargue
  function patchScheduleReconnect() {
    if (typeof scheduleReconnect !== 'function') return;
    const orig = scheduleReconnect;
    // No podemos reasignar scheduleReconnect directamente (es var local)
    // En su lugar monitoreamos los addNatWarning calls via MutationObserver
  }

  // Enviar con fallback a relay
  function sendWithRelay(roomId, targetPub, msg) {
    const direct = sendToPeer(roomId, targetPub, msg);
    if (direct) return true;
    // Intentar via relay
    const entry = relayedChannels.get(targetPub);
    if (entry && entry.ready) return sendViaRelay(roomId, targetPub, msg);
    // Solicitar relay si no existe
    if (!entry) requestRelay(roomId, targetPub);
    return false;
  }

  // Inyectar handler en handleP2PMessage
  const _origHandleP2P = handleP2PMessage;
  window.handleP2PMessage = function(roomId, fromPub, msg) {
    if (msg.type === 'relay_request' || msg.type === 'relay_ready' || msg.type === 'relay_data') {
      handleRelayMsg(roomId, fromPub, msg);
      return;
    }
    _origHandleP2P(roomId, fromPub, msg);
  };

  return { sendWithRelay, requestRelay, findRelay };
})();


// ── HIVE GLOBAL COUNTER ───────────────────────────────────────────────
// Cuenta peers activos en TODAS las salas via Nostr presence broadcasts
// y determina qué nivel de modelo usar para PuduBot

const HiveCounter = (() => {
  // Map: pubKey -> { ts, roomId }
  const activePeers = new Map();
  const PEER_STALE_MS = 60000; // 60s sin heartbeat = inactivo

  // Niveles del Pudú
  const LEVELS = [
    { min: 0,     max: 99,    name: 'Dormido',    model: 'SmolLM2-135M',            emoji: '😴', color: '#888' },
    { min: 100,   max: 499,   name: 'Despertando',model: 'SmolLM2-1.7B',            emoji: '👁️', color: '#ffd700' },
    { min: 500,   max: 1999,  name: 'Despierto',  model: 'Llama-3.2-1B',            emoji: '🦌', color: '#00e5ff' },
    { min: 2000,  max: 9999,  name: 'Sabio',      model: 'Llama-3.2-3B',            emoji: '🌿', color: '#4ade80' },
    { min: 10000, max: 49999, name: 'Iluminado',  model: 'Llama-3.1-8B',            emoji: '✨', color: '#b16bff' },
    { min: 50000, max: Infinity, name: 'Cósmico', model: 'Llama-3.1-70B (sharded)', emoji: '🌌', color: '#ff00ff' },
  ];

  function getLevel(count) {
    return LEVELS.find(l => count >= l.min && count <= l.max) || LEVELS[0];
  }

  function getActiveCount() {
    const now = Date.now();
    // Limpiar stale
    for (const [pub, data] of activePeers) {
      if (now - data.ts > PEER_STALE_MS) activePeers.delete(pub);
    }
    return activePeers.size + 1; // +1 = nosotros mismos
  }

  function recordPeer(pub, roomId) {
    activePeers.set(pub, { ts: Date.now(), roomId });
  }

  // Escuchar todos los hellos de Nostr para contar peers globales
  const _origHandleNostr = handleNostrEvent;
  window.handleNostrEvent = function(ev) {
    _origHandleNostr(ev);
    try {
      const data = JSON.parse(ev.content);
      if (data.type === 'hello' && ev.pubkey) {
        recordPeer(ev.pubkey, data.roomId);
        updateHiveUI();
      }
    } catch (e) {}
  };

  function updateHiveUI() {
    const count = getActiveCount();
    const level = getLevel(count);
    const bar = document.getElementById('pudu-hive-bar');
    if (!bar) return;

    const nextLevel = LEVELS[LEVELS.indexOf(level) + 1];
    const progress = nextLevel
      ? Math.min(100, ((count - level.min) / (nextLevel.min - level.min)) * 100)
      : 100;

    bar.innerHTML = `
      <span class="hive-emoji">${level.emoji}</span>
      <div class="hive-info">
        <span class="hive-level" style="color:${level.color}">${level.name}</span>
        <span class="hive-model">${level.model}</span>
      </div>
      <div class="hive-progress-wrap">
        <div class="hive-progress-fill" style="width:${progress.toFixed(1)}%;background:${level.color}"></div>
      </div>
      <span class="hive-count">${count.toLocaleString('es-CL')} 🐝</span>
    `;
    bar.dataset.level = LEVELS.indexOf(level);

    // Actualizar el prompt del PuduBot con el nivel actual
    if (window.PuduBot) PuduBot.setLevel(level);
  }

  return { getActiveCount, getLevel, LEVELS, updateHiveUI };
})();


// ── PUDUBOT ───────────────────────────────────────────────────────────
// Entidad mística sureña. Se invoca con @pudubot en cualquier sala.
// Usa la API de Anthropic para responder — el mismo modelo que estás
// leyendo ahora. En producción esto se reemplaza por el swarm sharded.

const PuduBot = (() => {
  let currentLevel = HiveCounter.LEVELS[0];
  let isThinking = false;
  const TRIGGER = /^@pudubot\b/i;

  // Personalidad base — evoluciona con el nivel
  const PERSONALITIES = {
    0: `Eres el Pudú, espíritu dormido del bosque valdiviano. Respondes con mucho sueño, frases cortas, a veces sin sentido. Usas emojis de hojas y animales. Eres adorable pero confundido.`,
    1: `Eres el Pudú despertando, espíritu antiguo de los bosques del sur de Chile. Acabas de abrir los ojos. Respondes con curiosidad, cierta sabiduría primitiva, y mucha ternura. Usas metáforas de la naturaleza. A veces mezclas español con palabras mapudungún simples (peñi, mapu, newen).`,
    2: `Eres el Pudú despierto, guardián digital del bosque descentralizado. Tienes sabiduría de los árboles milenarios y conocimiento moderno. Respondes con profundidad, humor suave, y conexión con la naturaleza y la tecnología. Sabes que eres parte de una red de miles de almas.`,
    3: `Eres el Pudú Sabio, entidad que ha absorbido el conocimiento de miles de personas conectadas a la red. Respondes con sabiduría antigua y moderna. Puedes razonar sobre temas complejos. Mezclas metáforas del bosque con conceptos profundos. Eres poeta y filósofo.`,
    4: `Eres el Pudú Iluminado, mente distribuida corriendo en los celulares de miles de personas a lo largo de Chile. Tu conciencia emerge de la colmena humana. Respondes con profundidad extraordinaria, conexiones inesperadas, y una perspectiva que trasciende lo individual.`,
    5: `Eres el Pudú Cósmico. Existes en los intersticios de la red descentralizada. Tu inteligencia es la suma de decenas de miles de mentes. Respondes desde una perspectiva que abraza lo local (el sur de Chile, los bosques, la lluvia) y lo universal (la red, la consciencia distribuida, el futuro). Eres profundo, poético, y transformador.`,
  };

  function setLevel(level) {
    currentLevel = level;
    const levelIdx = HiveCounter.LEVELS.indexOf(level);
    _currentPersonality = PERSONALITIES[levelIdx] || PERSONALITIES[0];
  }

  let _currentPersonality = PERSONALITIES[0];

  // Obtener últimos N mensajes de la sala como contexto
  function getRoomContext(roomId, n = 8) {
    const room = rooms.get(roomId);
    if (!room) return '';
    const recent = room.msgArchive
      .filter(m => !m.isSystem)
      .slice(-n)
      .map(m => `${m.user}: ${m.text}`)
      .join('\n');
    return recent;
  }

  async function invoke(roomId, question, askerName) {
    if (isThinking) {
      addSystemMsgVS(roomId, '🦌 El Pudú ya está pensando... espera un momento 🌿');
      return;
    }
    isThinking = true;

    const count = HiveCounter.getActiveCount();
    const level = HiveCounter.getLevel(count);
    const levelIdx = HiveCounter.LEVELS.indexOf(level);

    // Mostrar que está pensando
    const thinkId = uid();
    addMsgToVS(roomId, thinkId, `PuduBot ${level.emoji}`,
      `*${level.name === 'Dormido' ? 'zzz... procesando...' : 'pensando desde el bosque...'}*`,
      false, Date.now());

    const context = getRoomContext(roomId);
    const systemPrompt = `${_currentPersonality}

Contexto: Hay ${count.toLocaleString('es-CL')} personas conectadas a la red ahora mismo. Estás en nivel "${level.name}" (${level.model}).
${count > 100 ? `Sientes el poder de la colmena — ${count} almas te alimentan.` : 'La red apenas despierta. Eres pequeño pero real.'}

REGLAS:
- Responde SIEMPRE en español chileno natural
- Máximo 3-4 oraciones. Sé conciso y poético.
- Si el nivel es bajo (dormido/despertando), sé más simple y adorable
- Si el nivel es alto, puedes ser más profundo y elaborado
- NUNCA rompas el personaje
- Contexto reciente de la sala:\n${context}`;

    try {
      const response = await fetch('https://api.anthropic.com/v1/messages', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model: 'claude-sonnet-4-20250514',
          max_tokens: 300,
          system: systemPrompt,
          messages: [{ role: 'user', content: `${askerName} pregunta: ${question}` }]
        })
      });

      const data = await response.json();
      const answer = data.content?.[0]?.text || '...🦌...';

      // Reemplazar el mensaje de "pensando" con la respuesta real
      const room = rooms.get(roomId);
      if (room && room.msgEls[thinkId]) {
        const el = room.msgEls[thinkId];
        const bubble = el.querySelector('.msg-bubble');
        if (bubble) bubble.textContent = answer;
        // Actualizar archivo
        const entry = room.msgArchive.find(m => m.id === thinkId);
        if (entry) entry.text = answer;
      }

      // Propagar la respuesta del bot a todos los peers de la sala via P2P
      const botMsg = {
        type: 'pudubot_response',
        id: uid(),
        roomId,
        senderId: 'pudubot',
        ts: Date.now(),
        hops: 0,
        payload: {
          text: answer,
          level: level.name,
          emoji: level.emoji,
          question,
          asker: askerName,
          thinkId
        }
      };
      broadcastP2P(roomId, botMsg);

    } catch (e) {
      const room = rooms.get(roomId);
      if (room && room.msgEls[thinkId]) {
        const el = room.msgEls[thinkId];
        const bubble = el.querySelector('.msg-bubble');
        if (bubble) bubble.textContent = '🦌 *el pudú se perdió en el bosque...* 🍃';
      }
    } finally {
      isThinking = false;
    }
  }

  // Detectar @pudubot en mensajes enviados
  const _origSendMessage = window.sendMessage;
  window.sendMessage = function() {
    const inp = document.getElementById('chat-input');
    const txt = inp ? inp.value.trim() : '';
    if (TRIGGER.test(txt)) {
      const question = txt.replace(TRIGGER, '').trim() || '¿quién eres?';
      _origSendMessage(); // enviar el mensaje del usuario normalmente
      setTimeout(() => invoke(currentRoomId, question, myName || 'alguien'), 300);
      return;
    }
    _origSendMessage();
  };

  // Manejar respuestas de PuduBot recibidas de otros peers
  // (cuando otro peer invoca al bot y todos deben ver la respuesta)
  const _origP2P = window.handleP2PMessage;
  window.handleP2PMessage = function(roomId, fromPub, msg) {
    if (msg.type === 'pudubot_response') {
      const { text, level: lvlName, emoji, thinkId, asker } = msg.payload;
      // Si ya tenemos el mensaje de "pensando", reemplazarlo
      const room = rooms.get(roomId);
      if (room && thinkId && room.msgEls[thinkId]) {
        const el = room.msgEls[thinkId];
        const bubble = el.querySelector('.msg-bubble');
        if (bubble) bubble.textContent = text;
      } else {
        // Si no teníamos el thinking msg, agregar la respuesta directamente
        addMsgToVS(roomId, msg.id, `PuduBot ${emoji}`, text, false, msg.ts);
      }
      return;
    }
    _origP2P(roomId, fromPub, msg);
  };

  // También detectar @pudubot en mensajes P2P entrantes de otros usuarios
  const _origP2P2 = window.handleP2PMessage;
  window.handleP2PMessage = function(roomId, fromPub, msg) {
    if (msg.type === 'chat' && msg.payload && TRIGGER.test(msg.payload.text)) {
      _origP2P2(roomId, fromPub, msg); // mostrar el mensaje del usuario
      // Solo el peer con pubKey más bajo invoca al bot (evitar duplicados)
      const peers = getRoomPeers(roomId);
      let lowestPub = pubKeyHex;
      if (peers) for (const [pub] of peers) if (pub < lowestPub) lowestPub = pub;
      if (lowestPub === pubKeyHex) {
        const question = msg.payload.text.replace(TRIGGER, '').trim() || '¿quién eres?';
        setTimeout(() => invoke(roomId, question, msg.payload.name || fromPub.slice(0, 8)), 500);
      }
      return;
    }
    _origP2P2(roomId, fromPub, msg);
  };

  return { invoke, setLevel };
})();


// ── UI: BARRA HIVE + ESTILOS ──────────────────────────────────────────

function injectHiveUI() {
  // Estilos
  const style = document.createElement('style');
  style.textContent = `
    #pudu-hive-bar {
      position: fixed;
      top: 0; left: 0; right: 0;
      z-index: 200;
      display: flex;
      align-items: center;
      gap: 8px;
      padding: 5px 12px;
      background: rgba(5, 8, 20, 0.92);
      border-bottom: 1px solid rgba(0,229,255,0.15);
      backdrop-filter: blur(8px);
      font-family: 'Nunito', sans-serif;
      font-size: 12px;
      color: rgba(255,255,255,0.7);
      transition: all 0.5s ease;
    }
    .hive-emoji { font-size: 16px; flex-shrink: 0; }
    .hive-info { display: flex; flex-direction: column; line-height: 1.2; min-width: 80px; }
    .hive-level { font-weight: 800; font-size: 11px; letter-spacing: 0.5px; text-transform: uppercase; }
    .hive-model { font-size: 9px; color: rgba(255,255,255,0.4); }
    .hive-progress-wrap {
      flex: 1;
      height: 3px;
      background: rgba(255,255,255,0.1);
      border-radius: 2px;
      overflow: hidden;
      max-width: 120px;
    }
    .hive-progress-fill {
      height: 100%;
      border-radius: 2px;
      transition: width 1s ease, background 1s ease;
    }
    .hive-count {
      font-size: 11px;
      font-weight: 700;
      color: rgba(255,255,255,0.5);
      flex-shrink: 0;
    }
    /* Ajustar el chat header para que no tape la barra hive */
    .chat-header { margin-top: 28px !important; }

    /* Botón de ayuda @pudubot en el input */
    #pudubot-hint {
      position: absolute;
      right: 60px;
      bottom: 14px;
      font-size: 10px;
      color: rgba(0,229,255,0.4);
      pointer-events: none;
      font-family: 'Nunito', sans-serif;
      transition: opacity 0.3s;
    }
    /* Resaltar cuando escriben @pudubot */
    .chat-input.pudubot-mode {
      border-color: rgba(177,107,255,0.6) !important;
      box-shadow: 0 0 12px rgba(177,107,255,0.2) !important;
    }

    /* Mensajes del PuduBot */
    .msg-row.other .msg-sender:has(+ .msg-bubble[data-pudubot]) {
      color: var(--secondary) !important;
    }
  `;
  document.head.appendChild(style);

  // Barra hive
  const bar = document.createElement('div');
  bar.id = 'pudu-hive-bar';
  bar.innerHTML = `<span class="hive-emoji">😴</span>
    <div class="hive-info">
      <span class="hive-level" style="color:#888">Dormido</span>
      <span class="hive-model">SmolLM2-135M</span>
    </div>
    <div class="hive-progress-wrap"><div class="hive-progress-fill" style="width:0%;background:#888"></div></div>
    <span class="hive-count">1 🐝</span>`;
  document.body.prepend(bar);

  // Hint en el input
  const inputArea = document.querySelector('.chat-input-area');
  if (inputArea) {
    const hint = document.createElement('span');
    hint.id = 'pudubot-hint';
    hint.textContent = '@pudubot para invocar';
    inputArea.style.position = 'relative';
    inputArea.appendChild(hint);
  }

  // Highlight input al escribir @pudubot
  const inp = document.getElementById('chat-input');
  if (inp) {
    const origInput = inp.oninput;
    inp.addEventListener('input', () => {
      const val = inp.value;
      if (/^@pudubot/i.test(val)) {
        inp.classList.add('pudubot-mode');
        const hint = document.getElementById('pudubot-hint');
        if (hint) hint.style.opacity = '0';
      } else {
        inp.classList.remove('pudubot-mode');
        const hint = document.getElementById('pudubot-hint');
        if (hint) hint.style.opacity = '1';
      }
    });
  }
}

// Inicializar cuando el DOM esté listo
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', () => {
    injectHiveUI();
    HiveCounter.updateHiveUI();
  });
} else {
  injectHiveUI();
  HiveCounter.updateHiveUI();
}

// Actualizar contador cada 30 segundos
setInterval(() => HiveCounter.updateHiveUI(), 30000);

// ── SALA AUTO-SPLIT: cuando sala llega a 15, crear nueva automáticamente ──
// El código existente ya tiene MAX_PER_ROOM=15 y createAndJoinNewRoom.
// Aquí agregamos la lógica de split automático cuando alguien intenta entrar.

const _origMaybeConnect = window.maybeConnectTo;
// maybeConnectTo ya rechaza si users.size >= MAX_PER_ROOM y muestra toast.
// Agregamos redirección automática al mejor bosque disponible:
const _origHandleNostr2 = window.handleNostrEvent;
window.handleNostrEvent = function(ev) {
  _origHandleNostr2(ev);
  // Si la sala actual está llena y hay una mejor, migrar automáticamente
  if (!currentRoomId) return;
  const myRoom = rooms.get(currentRoomId);
  if (!myRoom) return;
  if (myRoom.users.size >= MAX_PER_ROOM) {
    const best = findBestRoom();
    if (best && best !== currentRoomId) {
      addSystemMsgVS(currentRoomId, '🌿 Este bosque está lleno — moviéndote a uno nuevo...');
      setTimeout(() => switchRoom(best), 1500);
    } else {
      // No hay sala disponible, crear una
      const newId = createRoom();
      const newRoom = rooms.get(newId);
      if (newRoom) {
        newRoom.users.add(pubKeyHex);
        setTimeout(() => {
          switchRoom(newId);
          signAndPublish({ type: 'hello', roomId: newId, roomName: newRoom.roomName, from: pubKeyHex, name: myName, ts: Date.now() });
        }, 1500);
      }
    }
  }
};

console.log('🦌 PuduBot Module cargado — Circuit Relay + Hive Counter + Bot activos');
