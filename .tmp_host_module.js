
    const ROOM_CAPACITY = 15;
    const ROOM_LOBBY_NAMESPACE = 'p2p-hive-lobby-v2';
    const ROOM_LOBBY_TTL_MS = 45000;
    const ROOM_BASE_PREFIX = 'sala-';
    const URL_FLAGS = new URLSearchParams(location.search);
const LOCAL_RELAYS = ['localhost', '127.0.0.1'].includes(location.hostname)
  ? [`ws://${location.hostname}:7777`]
  : [];
const RELAYS = [
  ...LOCAL_RELAYS,
  'wss://relay.damus.io',
  'wss://relay.primal.net',
  'wss://nos.lol',
  'wss://offchain.pub',
  'wss://nostr.mom',
];
    const lobbyPeers = new Map();
    const relayMap = new Map();
    const roomControllers = new Map();
    const publicIndexUrl = new URL('./index.html', location.href);
    const publicHostUrl = new URL('./host.html', location.href);
    const roomGrid = document.getElementById('room-grid');
    const logEl = document.getElementById('host-log');
    let nostrMod = null;
    async function getNostrMod() {
      if (!nostrMod) nostrMod = await import('https://esm.sh/nostr-tools@2/pure');
      return nostrMod;
    }

    function logLine(text, cls = '') {
      const line = document.createElement('div');
      line.className = `log-line ${cls}`.trim();
      line.textContent = `[${new Date().toLocaleTimeString('es-CL')}] ${text}`;
      logEl.prepend(line);
      while (logEl.children.length > 60) logEl.removeChild(logEl.lastChild);
    }

    function sanitizeRoomName(raw) {
      return (raw || '').toString().trim().toLowerCase().replace(/[^a-z0-9_-]/g, '').slice(0, 30);
    }
    function normalizeRoomName(raw) {
      const clean = sanitizeRoomName(raw);
      if (!clean || clean === 'public') return 'sala-1';
      if (/^sala-\d+$/.test(clean)) return clean;
      return clean;
    }
    function getRoomOrdinal(name) {
      const match = normalizeRoomName(name).match(/^sala-(\d+)$/);
      return match ? Math.max(1, parseInt(match[1], 10)) : Number.MAX_SAFE_INTEGER;
    }
    function formatRoomLabel(name) {
      const ordinal = getRoomOrdinal(name);
      return Number.isFinite(ordinal) && ordinal < Number.MAX_SAFE_INTEGER ? `Sala ${ordinal}` : normalizeRoomName(name);
    }
    function buildNextRoomName(existingNames = []) {
      let maxOrdinal = 1;
      for (const name of existingNames) maxOrdinal = Math.max(maxOrdinal, getRoomOrdinal(name));
      return `${ROOM_BASE_PREFIX}${maxOrdinal + 1}`;
    }
    function buildUserRoomUrl(roomName) {
      const url = new URL(publicIndexUrl);
      url.searchParams.set('room', normalizeRoomName(roomName));
      return url.toString();
    }
    function buildHostFrameUrl(roomName, options = {}) {
      const url = new URL(publicIndexUrl);
      url.searchParams.set('room', normalizeRoomName(roomName));
      url.searchParams.set('hostEmbed', '1');
      url.searchParams.set('hostKey', normalizeRoomName(roomName));
      if (options.prewarm) url.searchParams.set('prewarm', '1');
      return url.toString();
    }
    function getActiveLobbyEntries() {
      const now = Date.now();
      return [...lobbyPeers.values()].filter(entry => now - (entry.lastSeen || 0) <= ROOM_LOBBY_TTL_MS);
    }
    function getRoomDirectory() {
      const grouped = new Map();
      for (const entry of getActiveLobbyEntries()) {
        const roomName = normalizeRoomName(entry.roomName);
        if (!grouped.has(roomName)) grouped.set(roomName, { roomName, count: 0, newestJoinAt: 0, peers: [] });
        const room = grouped.get(roomName);
        room.count++;
        room.peers.push(entry);
        room.newestJoinAt = Math.max(room.newestJoinAt, entry.joinedAt || 0);
      }
      const rooms = [...grouped.values()].sort((a, b) => getRoomOrdinal(a.roomName) - getRoomOrdinal(b.roomName));
      if (!rooms.length) rooms.push({ roomName: 'sala-1', count: 0, newestJoinAt: 0, peers: [] });
      return rooms;
    }
    function updateSummary() {
      const rooms = getRoomDirectory();
      const statuses = [...roomControllers.values()];
      const ready = statuses.filter(entry => {
        const status = entry.status || {};
        const coverageTotal = status.coverageTotal || 26;
        return !!status.ready && !!status.petalsReady && (status.coverageCount || 0) >= coverageTotal;
      }).length;
      const humans = rooms.reduce((sum, room) => sum + room.count, 0);
      document.getElementById('stat-rooms').textContent = String(Math.max(1, roomControllers.size));
      document.getElementById('stat-humans').textContent = String(humans);
      document.getElementById('stat-ready').textContent = String(ready);
      const connectedRelays = [...relayMap.values()].filter(entry => entry.status === 'connected').length;
      document.getElementById('stat-relays').textContent = `${connectedRelays}/${relayMap.size}`;
      document.getElementById('public-link-box').textContent = publicIndexUrl.toString();
    }

    function connectRelay(url) {
      if (relayMap.has(url) && ['connecting', 'connected'].includes(relayMap.get(url).status)) return;
      relayMap.set(url, { ws: null, status: 'connecting' });
      const entry = relayMap.get(url);
      try {
        const ws = new WebSocket(url);
        entry.ws = ws;
        ws.onopen = () => {
          entry.status = 'connected';
          const since = Math.floor(Date.now() / 1000) - 900;
          ws.send(JSON.stringify(['REQ', `host-lobby-${Math.random().toString(36).slice(2, 8)}`, { kinds: [1], '#t': [ROOM_LOBBY_NAMESPACE], since, limit: 500 }]));
          logLine(`Relay conectado: ${url}`, 'ok');
          updateSummary();
        };
        ws.onmessage = event => {
          let msg;
          try { msg = JSON.parse(event.data); } catch { return; }
          if (!Array.isArray(msg) || msg[0] !== 'EVENT') return;
          const content = msg[2]?.content;
          if (!content) return;
          let payload;
          try { payload = JSON.parse(content); } catch { return; }
          if (payload?.type === 'lobby_presence' && payload.peerId) handleLobbyPresence(payload);
        };
        ws.onclose = ws.onerror = () => {
          entry.status = 'disconnected';
          updateSummary();
          setTimeout(() => connectRelay(url), 5000);
        };
      } catch (error) {
        entry.status = 'disconnected';
        setTimeout(() => connectRelay(url), 5000);
      }
    }

    function handleLobbyPresence(payload) {
      lobbyPeers.set(payload.peerId, {
        peerId: payload.peerId,
        roomName: normalizeRoomName(payload.roomName),
        nickname: payload.nickname || null,
        lastSeen: payload.timestamp || Date.now(),
        joinedAt: payload.joinedAt || payload.timestamp || Date.now(),
      });
      ensureRoomControllers();
      renderRooms();
    }

    function ensureRoomController(roomName, options = {}) {
      const normalized = normalizeRoomName(roomName);
      let controller = roomControllers.get(normalized);
      if (controller) {
        if (options.prewarm) warmRoom(normalized);
        return controller;
      }
      const wrapper = document.createElement('section');
      wrapper.className = 'room-card';
      wrapper.dataset.room = normalized;
      wrapper.innerHTML = `
        <div class="room-head">
          <div class="room-title-row">
            <div class="room-title">${formatRoomLabel(normalized)}</div>
            <div class="room-badge idle" data-room-badge>inicializando</div>
          </div>
          <div class="room-meta">
            <div class="mini"><span class="mini-label">Humanos</span><span class="mini-value" data-room-humans>0 / ${ROOM_CAPACITY}</span></div>
            <div class="mini"><span class="mini-label">Estado</span><span class="mini-value" data-room-stage>Arrancando</span></div>
            <div class="mini"><span class="mini-label">Motor</span><span class="mini-value" data-room-model>SmolLM2-1.7B-Q8 · esperando shards</span></div>
          </div>
          <div class="room-actions">
            <button class="btn secondary" data-open-room>↗ Abrir sala</button>
            <button class="btn secondary" data-copy-room>📋 Copiar URL</button>
            <button class="btn secondary" data-warm-room>🌸 Sincronizar</button>
          </div>
          <div class="room-progress" data-room-progress>Esperando presencia humana en este bosque…</div>
        </div>
        <iframe class="room-frame" loading="eager" allow="clipboard-read; clipboard-write"></iframe>
      `;
      const iframe = wrapper.querySelector('iframe');
      iframe.src = buildHostFrameUrl(normalized, { prewarm: !!options.prewarm });
      wrapper.querySelector('[data-open-room]').addEventListener('click', () => window.open(buildUserRoomUrl(normalized), '_blank', 'noopener'));
      wrapper.querySelector('[data-copy-room]').addEventListener('click', async () => {
        try {
          await navigator.clipboard.writeText(buildUserRoomUrl(normalized));
          logLine(`URL copiada: ${normalized}`, 'ok');
        } catch (error) {
          logLine(`No pude copiar ${normalized}: ${error.message}`, 'err');
        }
      });
      wrapper.querySelector('[data-warm-room]').addEventListener('click', () => warmRoom(normalized, true));
      controller = {
        roomName: normalized,
        wrapper,
        iframe,
        status: null,
        loaded: false,
        pendingWarm: !!options.prewarm,
        warmed: !!options.prewarm,
      };
      roomControllers.set(normalized, controller);
      roomGrid.appendChild(wrapper);
      logLine(`Colmena creada para ${normalized}`, 'ok');
      iframe.addEventListener('load', () => {
        controller.loaded = true;
        if (controller.pendingWarm || options.prewarm) {
          setTimeout(() => warmRoom(normalized, true), 1200);
        }
      });
      updateSummary();
      return controller;
    }

    function warmRoom(roomName, forced = false) {
      const controller = roomControllers.get(normalizeRoomName(roomName));
      if (!controller) return;
      if (controller.warmed && !forced) return;
      controller.pendingWarm = true;
      if (!controller.loaded || !controller.iframe?.contentWindow) {
        renderRooms();
        return;
      }
      controller.warmed = true;
      controller.pendingWarm = false;
      controller.iframe.contentWindow.postMessage({ type: 'pudu-host-warm', roomName: controller.roomName }, location.origin);
      logLine(`Warm solicitado para ${controller.roomName}`, 'warn');
      renderRooms();
    }

    function updateControllerFromStatus(status) {
      const roomName = normalizeRoomName(status.roomName);
      const controller = ensureRoomController(roomName);
      controller.status = status;
      const humanCount = getRoomDirectory().find(room => room.roomName === roomName)?.count || 0;
      const coverageTotal = status.coverageTotal || 26;
      const reallyReady = !!status.ready && !!status.petalsReady && (status.coverageCount || 0) >= coverageTotal;
      controller.wrapper.querySelector('[data-room-humans]').textContent = `${humanCount} / ${ROOM_CAPACITY}`;
      controller.wrapper.querySelector('[data-room-stage]').textContent = status.stageLabel || 'Pudu Dormido';
      controller.wrapper.querySelector('[data-room-model]').textContent = reallyReady
        ? `${status.modelLabel || 'SmolLM2-1.7B-Q8 · Web-Petals'} · listo`
        : `${status.modelLabel || 'SmolLM2-1.7B-Q8 · Web-Petals'} · ${status.petalsReady ? 'sincronizando cobertura' : 'esperando shards'}`;
      controller.wrapper.querySelector('[data-room-progress]').textContent = status.progressText || (reallyReady ? 'La colmena de esta sala ya está despierta.' : 'Esperando que entren suficientes celulares y se repartan los shards.');
      const badge = controller.wrapper.querySelector('[data-room-badge]');
      badge.className = `room-badge ${reallyReady ? 'ready' : controller.warmed ? 'warming' : 'idle'}`;
      badge.textContent = reallyReady
        ? `lista · ${Math.round(status.collectiveTPS || 0)} tok/s`
        : controller.warmed
          ? 'calentando cerebro'
          : 'en espera';
      updateSummary();
    }

    function renderRooms() {
      const rooms = getRoomDirectory();
      if (!roomControllers.size) {
        roomGrid.innerHTML = '<div class="empty-state">Esperando el primer bosque… Sala 1 se levantará apenas arranque el host.</div>';
        updateSummary();
        return;
      }
      const empty = roomGrid.querySelector('.empty-state');
      if (empty) empty.remove();
      for (const room of rooms) {
        const controller = roomControllers.get(room.roomName);
        if (!controller) continue;
        const humanEl = controller.wrapper.querySelector('[data-room-humans]');
        if (humanEl) humanEl.textContent = `${room.count} / ${ROOM_CAPACITY}`;
      }
      updateSummary();
    }

    function ensureRoomControllers() {
      const rooms = getRoomDirectory();
      const names = rooms.map(room => room.roomName);
      ensureRoomController('sala-1', { prewarm: true });
      for (const room of rooms) {
        ensureRoomController(room.roomName, { prewarm: room.roomName === 'sala-1' });
        if (room.count > 0) warmRoom(room.roomName);
      }
      const highestActive = names.reduce((max, name) => Math.max(max, getRoomOrdinal(name)), 1);
      const hottest = rooms.find(room => getRoomOrdinal(room.roomName) === highestActive);
      if (hottest && hottest.count >= ROOM_CAPACITY - 2) {
        ensureRoomController(buildNextRoomName(names));
      }
      renderRooms();
    }

    function pruneLobby() {
      const now = Date.now();
      for (const [peerId, entry] of lobbyPeers) {
        if (now - (entry.lastSeen || 0) > ROOM_LOBBY_TTL_MS * 1.5) lobbyPeers.delete(peerId);
      }
    }

    window.addEventListener('message', event => {
      if (event.origin !== location.origin) return;
      const data = event.data || {};
      if (data.type === 'pudu-host-status' && data.roomName) {
        updateControllerFromStatus(data);
      }
    });

    document.getElementById('ensure-first-room').addEventListener('click', () => {
      ensureRoomController('sala-1', { prewarm: true });
      warmRoom('sala-1', true);
    });
    document.getElementById('copy-public-link').addEventListener('click', async () => {
      try {
        await navigator.clipboard.writeText(publicIndexUrl.toString());
        logLine('index.html copiado al portapapeles', 'ok');
      } catch (error) {
        logLine(`No pude copiar index.html: ${error.message}`, 'err');
      }
    });
    document.getElementById('open-public-link').addEventListener('click', () => {
      window.open(publicIndexUrl.toString(), '_blank', 'noopener');
    });

    for (const relay of RELAYS) connectRelay(relay);
    ensureRoomController('sala-1', { prewarm: true });
    updateSummary();
    renderRooms();
    setInterval(() => {
      pruneLobby();
      ensureRoomControllers();
    }, 2500);
    setInterval(updateSummary, 3000);
    logLine(`Host listo en ${publicHostUrl.toString()}`, 'ok');
    logLine(`Index público: ${publicIndexUrl.toString()}`, 'ok');
  
