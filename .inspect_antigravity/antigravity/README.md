# P2P-LLM — Decentralized Browser AI with Auto-Scaling Model Tiers

A fully browser-based, peer-to-peer LLM inference system that runs locally on any device and automatically improves when peers join the network.

Every browser tab is **both a client and a compute node**. The system discovers peers via Nostr relays, establishes WebRTC DataChannels, and dynamically selects the best model tier based on real-time network capacity.

---

## Quick Start

### 1. Install the Local Relay

```bash
cd versionAntigravity
npm install ws
```

### 2. Start the Local Nostr Relay

```bash
node relay.js
# → WebSocket relay running on ws://localhost:7777
```

### 3. Open the App

Open `index.html` in a WebGPU-capable browser (Chrome 113+, Edge 113+):

```
# Option A: Double-click index.html
# Option B: Use a local server
npx -y serve .
```

### 4. Test with Multiple Peers

Open **2–3 browser tabs** of `index.html`. Each tab will:

1. Load the small model locally  
2. Discover other tabs via the local Nostr relay  
3. Establish WebRTC DataChannels between them  
4. Compute a network score and potentially upgrade the model tier  

> **Mobile testing**: Open on 2–3 phones on the same network. Ensure the relay is accessible (use your LAN IP instead of `localhost`).

---

## Architecture Overview

```
┌──────────────────────────────────────────────────────────────┐
│                    Browser Tab (each peer)                    │
│                                                              │
│  ┌─────────┐   ┌──────────┐   ┌────────────────────────┐   │
│  │ Chat UI │◄──│ Engine   │◄──│ Auto-Scaling Controller │   │
│  │         │   │ (WebLLM) │   │ (Score → Tier → Model)  │   │
│  └─────────┘   └──────────┘   └───────────┬────────────┘   │
│                      ▲                     │                 │
│                      │              ┌──────┴──────┐         │
│                      │              │ Network     │         │
│                      │              │ Score       │         │
│                      │              │ Calculator  │         │
│                      │              └──────┬──────┘         │
│               ┌──────┴──────┐              │                 │
│               │ Speculative │       ┌──────┴──────┐         │
│               │ Decoding    │◄──────│ Peer        │         │
│               │ Engine      │       │ Manager     │         │
│               └─────────────┘       └──────┬──────┘         │
│                                            │                 │
│                    ┌───────────────────────┤                 │
│                    │                       │                 │
│              ┌─────┴─────┐         ┌──────┴──────┐         │
│              │ WebRTC    │         │ Nostr       │         │
│              │ DataChan  │         │ Signaling   │         │
│              │ (Binary)  │         │             │         │
│              └─────┬─────┘         └──────┬──────┘         │
│                    │                       │                 │
└────────────────────┼───────────────────────┼─────────────────┘
                     │                       │
              ┌──────┴──────┐         ┌──────┴──────┐
              │ Other Peers │         │ Nostr Relay │
              │ (WebRTC)    │         │ (ws://...)  │
              └─────────────┘         └─────────────┘
```

---

## Auto-Scaling: How It Works

### Network Capacity Score

The system continuously computes a **score** (0–6) based on:

| Factor | Points |
|--------|--------|
| Each stable peer (RTT < 180ms) | +1 |
| Tier A (desktop GPU) peer present | +1 |
| Average RTT > threshold | -1 |

### Model Tiers

| Tier | Model | Min Score | Execution Mode |
|------|-------|-----------|----------------|
| **small** | Llama-3.2-1B (q4f16) | 0 | `local` — all inference on-device |
| **medium** | Llama-3.1-3B (q4f16) | 2 | `assist` — local + peer token assist |
| **large** | Llama-3.1-8B (q4f16) | 4 | `assist+verify` — assist + speculative verification |

### Tier Transitions

```
Score 0-1 → small  (always works, no peers needed)
Score 2-3 → medium (2+ stable peers with good latency)
Score 4-6 → large  (4+ stable peers or desktop GPUs in swarm)
```

**Hysteresis**: Tier changes require `CONFIG.hysteresisMs` (3 seconds) of sustained score to avoid flapping.

- **Upgrading**: The new model preloads in the background. The switch happens only after loading completes.
- **Downgrading**: Waits until the current generation finishes, then switches.

### Execution Modes

#### `local` (Tier: small)
All inference runs entirely on the local device. No network dependency.

#### `assist` (Tier: medium)
- Local node generates tokens normally
- Every N tokens (speculativeTokens=4), sends context to up to 2 remote peers
- Peers return candidate tokens
- Local node can use peer suggestions to improve or accelerate generation

#### `assist+verify` (Tier: large)
- Same as `assist`, plus:
- Local node drafts speculative tokens ahead
- Peers verify batches asynchronously
- Tokens are accepted or rolled back based on agreement

---

## Speculative Multi-Peer Decoding

The pipeline **never blocks on network**:

```
[Local]        [Peer A]       [Peer B]
  │                │              │
  ├─ Draft 4 tok ─►│              │
  │                ├─ Generate ──►│
  ├─ Draft 4 more  │              │
  │                │              │
  │◄── Response ───┤              │
  │   compare/merge│              │
  │                │◄── Response ─┤
  │◄───────────────┼── merge ─────┤
  ▼
 Output
```

- Cross-node hops per token: **≤ 1 remote hop**
- If peers are slow or disconnected, local generation continues uninterrupted

---

## Binary Protocol

All WebRTC messages use **ArrayBuffer** format:

```
┌──────────┬──────────────┬──────────┬──────────────┐
│ Type (1B)│ RequestID(4B)│ Seq (2B) │ Payload (var)│
└──────────┴──────────────┴──────────┴──────────────┘
```

| Type | Code | Direction |
|------|------|-----------|
| `PEER_HELLO` | 0x01 | Both |
| `TOKEN_REQUEST` | 0x02 | Requester → Worker |
| `TOKEN_RESPONSE` | 0x03 | Worker → Requester |
| `LOGITS` | 0x04 | Worker → Requester |
| `PING` | 0x10 | Both |
| `PONG` | 0x11 | Both |
| `VERIFY_REQUEST` | 0x20 | Requester → Worker |
| `VERIFY_RESPONSE` | 0x21 | Worker → Requester |

---

## Device Tier Classification

Devices are classified at runtime:

| Tier | Criteria | Role |
|------|----------|------|
| **A** | Desktop with WebGPU, ≥8GB RAM | Full compute peer |
| **B** | Laptop / tablet with moderate resources | Compute peer |
| **C** | Mobile device | Assist only (if `mobileAssistOnly=true`) |

Mobile devices (Tier C) are **never placed in the critical path** when `mobileAssistOnly` is enabled.

---

## Nostr Relay Configuration

### Using the Local Relay

The included `relay.js` is a minimal NIP-01 relay for testing:

```bash
node relay.js         # port 7777
node relay.js 9999    # custom port
```

Features:
- In-memory event storage (no persistence)
- Event TTL: 10 minutes
- Max 5000 events
- NIP-11 info document at HTTP root
- No signature verification (dev mode)

### Using Public Relays

The app connects to both local and public relays by default:

```javascript
nostrRelays: [
  "ws://localhost:7777",        // local relay
  "wss://relay.damus.io",       // public
  "wss://nos.lol",              // public
  "wss://relay.nostr.band"      // public
]
```

For **internet-wide** peer discovery, only public relays are needed. Remove the local relay entry if not testing locally.

### Using Your LAN IP (Mobile Testing)

Edit the relay URL in `index.html`:

```javascript
nostrRelays: [
  "ws://192.168.1.100:7777",   // your LAN IP
  "wss://relay.damus.io"
]
```

---

## Failure Handling

| Scenario | Response |
|----------|----------|
| Peer disconnects | Silently removed from peer list |
| RTT too high (>180ms) | Peer excluded from assist pool |
| All peers lost | Immediate fallback to local mode |
| Model preload fails | Stay on current tier/model |
| Nostr relay offline | Retry every 5 seconds; app works without peers |
| WebGPU unavailable | Falls back to WASM backend via WebLLM |

The system **never depends on peers for correctness**. Local inference is always the ground truth.

---

## Adding New Tiers / Models

Edit the `CONFIG.tiers` array in `index.html`:

```javascript
tiers: [
  {
    name: "tiny",
    model: "SmolLM2-135M-Instruct-q4f16_1-MLC",  // very small model
    minScore: 0,
    mode: "local"
  },
  {
    name: "small",
    model: "Llama-3.2-1B-Instruct-q4f16_1-MLC",
    minScore: 1,
    mode: "local"
  },
  {
    name: "medium",
    model: "Llama-3.1-3B-Instruct-q4f16_1-MLC",
    minScore: 3,
    mode: "assist"
  },
  {
    name: "large",
    model: "Llama-3.1-8B-Instruct-q4f16_1-MLC",
    minScore: 5,
    mode: "assist+verify"
  }
]
```

**Rules for adding tiers:**
1. `minScore` must increase with each tier
2. Models must be available in the [WebLLM model catalog](https://webllm.mlc.ai/)
3. Consider device memory — 8B models need ~4GB RAM
4. Update `defaultModel` to match the first tier's model

---

## Configuration Reference

```javascript
const CONFIG = {
  // Initial model (must match first tier)
  defaultModel: "Llama-3.2-1B-Instruct-q4f16_1-MLC",

  // Peer limits
  maxPeers: 6,              // max concurrent peer connections
  maxRemoteWorkers: 2,      // max peers used for assist per request

  // Latency policy
  maxAssistLatencyMs: 180,   // RTT threshold for assist eligibility
  hysteresisMs: 3000,        // min time between tier changes

  // Speculative decoding
  speculativeTokens: 4,      // tokens drafted before peer check

  // Mobile policy
  mobileAssistOnly: true,    // Tier C never in critical path

  // Nostr relays
  nostrRelays: [...],        // WebSocket URLs for peer discovery

  // Timing
  presenceIntervalMs: 15000, // how often to broadcast presence
  presenceTTLMs: 45000,      // when to consider a peer stale
  pingIntervalMs: 5000,      // RTT measurement interval
  scoreRecalcMs: 2000,       // score recalculation interval
};
```

---

## Performance Targets

| Target | Status |
|--------|--------|
| Runs on 2–3 mobile browsers | ✅ Tier C with 1B model |
| Produces responses offline | ✅ Local-first always |
| Improves latency with peers | ✅ Assist mode + tier upgrade |
| No UI freeze during model switch | ✅ Background loading |
| ≤1 remote hop per token | ✅ Direct peer assist only |

---

## Technology Stack

| Component | Technology |
|-----------|-----------|
| LLM Runtime | `@mlc-ai/web-llm` (ES module via CDN) |
| Compute | WebGPU (WASM fallback) |
| P2P | WebRTC DataChannels (binary ArrayBuffer) |
| Discovery | Nostr relays (NIP-01) |
| Signaling | Nostr events (kind 30078) |
| Storage | IndexedDB (model cache via WebLLM) |

---

## Limitations

1. **WebGPU availability**: Not all browsers/devices support WebGPU. The system falls back to WASM but performance is significantly lower.

2. **Mobile memory**: Phones typically have 2–4GB available to the browser. The 1B model (~600MB quantized) works, but larger models may fail on mobile.

3. **WebRTC NAT traversal**: Peers behind strict NATs may not connect. The system uses Google STUN servers but no TURN relay — some connections may fail.

4. **Nostr signing**: The local relay doesn't verify cryptographic signatures. For production, use proper Nostr key management and a relay that validates signatures.

5. **Model download time**: First load of any model requires downloading from CDN (hundreds of MB). Subsequent loads use IndexedDB cache.

6. **Cross-origin isolation**: Some browsers require specific headers for SharedArrayBuffer (used by WASM fallback). Serving from `file://` may have restrictions.

7. **Speculative decoding accuracy**: Peer-assisted token generation uses heuristic merging, not mathematically guaranteed correctness. Local generation is always the authority.

8. **Battery impact**: Continuous WebGPU inference drains mobile batteries quickly. Consider battery-aware policies for production use.

---

## File Structure

```
versionAntigravity/
├── index.html     # Complete application (UI + logic, single file)
├── relay.js       # Minimal Nostr relay for local testing
├── README.md      # This file
└── package.json   # Dependencies (ws for relay)
```

---

## License

MIT
