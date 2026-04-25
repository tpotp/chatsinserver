
    const roomInput = document.getElementById('sim-room');
    const peersInput = document.getElementById('sim-peers');
    const frame = document.getElementById('sim-frame');
    const hostUrlEl = document.getElementById('sim-host-url');
    const phoneUrlEl = document.getElementById('sim-phone-url');
    const assistUrlEl = document.getElementById('sim-assist-url');

    function normalizeRoomName(raw) {
      const clean = (raw || '').toString().trim().toLowerCase().replace(/[^a-z0-9_-]/g, '').slice(0, 30);
      return clean || 'sala-1';
    }

    function makeRoomName() {
      return 'sala-1';
    }

    function nextNumericRoom(current) {
      const match = normalizeRoomName(current).match(/^sala-(\d+)$/);
      const next = match ? Math.max(1, parseInt(match[1], 10)) + 1 : 1;
      return `sala-${next}`;
    }

    function clampPeers(value) {
      const parsed = parseInt(value, 10);
      if (!Number.isFinite(parsed)) return 50;
      return Math.min(1000, Math.max(2, parsed));
    }

    function buildUrls() {
      const room = normalizeRoomName(roomInput.value || makeRoomName());
      const peers = clampPeers(peersInput.value);
      roomInput.value = room;
      peersInput.value = String(peers);

      const base = new URL('./index.html', location.href);
      base.searchParams.set('room', room);

      const host = new URL(base);
      host.searchParams.set('simulate', String(peers));

      const phone = new URL(base);
      const assist = new URL(base);
      assist.searchParams.set('assistOnly', '1');

      return { room, peers, host: host.toString(), phone: phone.toString(), assist: assist.toString() };
    }

    function renderUrls(pushState = true) {
      const urls = buildUrls();
      hostUrlEl.textContent = urls.host;
      phoneUrlEl.textContent = urls.phone;
      assistUrlEl.textContent = urls.assist;
      frame.src = urls.host;

      const pageUrl = new URL(location.href);
      pageUrl.searchParams.set('room', urls.room);
      pageUrl.searchParams.set('peers', String(urls.peers));
      if (pushState) history.replaceState({}, '', pageUrl.toString());
      return urls;
    }

    function initializeFromQuery() {
      const params = new URLSearchParams(location.search);
      roomInput.value = normalizeRoomName(params.get('room') || makeRoomName());
      peersInput.value = String(clampPeers(params.get('peers') || '50'));
    }

    async function copyPhoneUrl() {
      const { phone } = buildUrls();
      await navigator.clipboard.writeText(phone);
    }

    document.getElementById('launch-sim').addEventListener('click', () => renderUrls());
    document.getElementById('new-room').addEventListener('click', () => {
      roomInput.value = nextNumericRoom(roomInput.value || 'sala-1');
      renderUrls();
    });
    document.getElementById('copy-phone').addEventListener('click', async () => {
      try {
        await copyPhoneUrl();
        document.getElementById('copy-phone').textContent = '✅ Link copiado';
        setTimeout(() => { document.getElementById('copy-phone').textContent = '📋 Copiar link celular'; }, 1600);
      } catch (_) {}
    });

    roomInput.addEventListener('change', () => renderUrls());
    peersInput.addEventListener('change', () => renderUrls());

    initializeFromQuery();
    renderUrls(false);
  
