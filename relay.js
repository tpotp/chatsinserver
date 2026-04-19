#!/usr/bin/env node
/**
 * relay.js — Minimal Nostr-Compatible WebSocket Relay
 * 
 * Implements a subset of NIP-01 for local P2P signaling and testing.
 * No signature verification (for local dev), in-memory event storage.
 * 
 * Usage:
 *   node relay.js              # starts on port 7777
 *   node relay.js 9999         # starts on port 9999
 * 
 * Dependencies: ws (npm install ws)
 */

const http = require('http');
let WebSocket, WebSocketServer;

try {
  const ws = require('ws');
  WebSocket = ws;
  WebSocketServer = ws.Server || ws.WebSocketServer;
} catch (e) {
  console.error('Error: "ws" package not found. Run: npm install ws');
  process.exit(1);
}

const PORT = parseInt(process.argv[2] || '7777', 10);
const MAX_EVENTS = 5000;       // max events in memory
const EVENT_TTL_MS = 600000;   // 10 minutes TTL for ephemeral events

// ─── In-Memory Store ──────────────────────────────────────
const events = new Map();       // id → event
const subscriptions = new Map(); // ws → Map<subId, filters[]>

// ─── Event Matching ───────────────────────────────────────
function matchesFilter(event, filter) {
  if (filter.ids && !filter.ids.includes(event.id)) return false;
  if (filter.authors && !filter.authors.includes(event.pubkey)) return false;
  if (filter.kinds && !filter.kinds.includes(event.kind)) return false;
  if (filter.since && event.created_at < filter.since) return false;
  if (filter.until && event.created_at > filter.until) return false;

  // NIP-01 tag filters (#e, #p, etc.)
  for (const [key, values] of Object.entries(filter)) {
    if (key.startsWith('#') && key.length === 2) {
      const tagName = key[1];
      const eventTags = (event.tags || [])
        .filter(t => t[0] === tagName)
        .map(t => t[1]);
      if (!values.some(v => eventTags.includes(v))) return false;
    }
  }

  return true;
}

function matchesAnyFilter(event, filters) {
  return filters.some(f => matchesFilter(event, f));
}

// ─── Cleanup ──────────────────────────────────────────────
function pruneEvents() {
  const now = Math.floor(Date.now() / 1000);
  const cutoff = now - (EVENT_TTL_MS / 1000);

  for (const [id, event] of events) {
    if (event.created_at < cutoff) {
      events.delete(id);
    }
  }

  // Hard cap
  if (events.size > MAX_EVENTS) {
    const sorted = [...events.entries()]
      .sort((a, b) => a[1].created_at - b[1].created_at);
    const toRemove = sorted.slice(0, events.size - MAX_EVENTS);
    for (const [id] of toRemove) {
      events.delete(id);
    }
  }
}

// ─── HTTP + WebSocket Server ──────────────────────────────
const server = http.createServer((req, res) => {
  // NIP-11 relay information document
  if (req.headers.accept === 'application/nostr+json' || req.url === '/') {
    res.writeHead(200, {
      'Content-Type': 'application/json',
      'Access-Control-Allow-Origin': '*'
    });
    res.end(JSON.stringify({
      name: 'p2p-llm-local-relay',
      description: 'Minimal Nostr relay for P2P LLM signaling',
      supported_nips: [1],
      software: 'p2p-llm-relay',
      version: '1.0.0'
    }));
    return;
  }
  res.writeHead(404);
  res.end();
});

const wss = new WebSocketServer({ server });

wss.on('connection', (ws, req) => {
  const clientAddr = req.socket.remoteAddress;
  console.log(`[+] Client connected from ${clientAddr}`);

  subscriptions.set(ws, new Map());

  ws.on('message', (raw) => {
    let msg;
    try {
      msg = JSON.parse(raw.toString());
    } catch (e) {
      ws.send(JSON.stringify(['NOTICE', 'error: invalid JSON']));
      return;
    }

    if (!Array.isArray(msg) || msg.length < 2) {
      ws.send(JSON.stringify(['NOTICE', 'error: invalid message format']));
      return;
    }

    const type = msg[0];

    switch (type) {
      case 'EVENT': {
        const event = msg[1];
        if (!event || !event.id || event.kind === undefined) {
          ws.send(JSON.stringify(['OK', event?.id || '', false, 'error: invalid event']));
          return;
        }

        // Store event (no sig verification for local relay)
        events.set(event.id, event);
        ws.send(JSON.stringify(['OK', event.id, true, '']));

        // Broadcast to matching subscriptions on OTHER clients
        for (const [client, subs] of subscriptions) {
          if (client === ws || client.readyState !== WebSocket.OPEN) continue;
          for (const [subId, filters] of subs) {
            if (matchesAnyFilter(event, filters)) {
              client.send(JSON.stringify(['EVENT', subId, event]));
            }
          }
        }

        pruneEvents();
        break;
      }

      case 'REQ': {
        const subId = msg[1];
        const filters = msg.slice(2);

        if (!subId || filters.length === 0) {
          ws.send(JSON.stringify(['NOTICE', 'error: invalid REQ']));
          return;
        }

        // Store subscription
        const clientSubs = subscriptions.get(ws);
        if (clientSubs) {
          clientSubs.set(subId, filters);
        }

        // Send stored events matching filters
        let count = 0;
        const limit = filters[0]?.limit || 100;
        const sortedEvents = [...events.values()]
          .sort((a, b) => b.created_at - a.created_at);

        for (const event of sortedEvents) {
          if (count >= limit) break;
          if (matchesAnyFilter(event, filters)) {
            ws.send(JSON.stringify(['EVENT', subId, event]));
            count++;
          }
        }

        ws.send(JSON.stringify(['EOSE', subId]));
        break;
      }

      case 'CLOSE': {
        const subId = msg[1];
        const clientSubs = subscriptions.get(ws);
        if (clientSubs) {
          clientSubs.delete(subId);
        }
        break;
      }

      default:
        ws.send(JSON.stringify(['NOTICE', `error: unknown message type: ${type}`]));
    }
  });

  ws.on('close', () => {
    subscriptions.delete(ws);
    console.log(`[-] Client disconnected from ${clientAddr}`);
  });

  ws.on('error', (err) => {
    console.error(`[!] WebSocket error from ${clientAddr}:`, err.message);
    subscriptions.delete(ws);
  });
});

server.listen(PORT, () => {
  console.log(`
╔══════════════════════════════════════════════╗
║  P2P-LLM Nostr Relay (Local)                ║
║  WebSocket: ws://localhost:${PORT}              ║
║  NIP-11:    http://localhost:${PORT}             ║
║  Events TTL: ${EVENT_TTL_MS / 1000}s | Max Events: ${MAX_EVENTS}       ║
╚══════════════════════════════════════════════╝
  `);
});

// Graceful shutdown
process.on('SIGINT', () => {
  console.log('\n[*] Shutting down relay...');
  wss.clients.forEach(ws => ws.close());
  server.close(() => process.exit(0));
});

process.on('SIGTERM', () => {
  wss.clients.forEach(ws => ws.close());
  server.close(() => process.exit(0));
});
