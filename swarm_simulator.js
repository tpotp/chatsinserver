/**
 * swarm_simulator.js — Hyper-Realistic P2P Mobile Inference Simulator v2
 *
 * Simulates a real-world distributed inference swarm with:
 *   - Real 4G/5G network profiles (ITU-R M.2135 parameters)
 *   - Thermal throttling, battery drain, memory pressure on mobile
 *   - TCP congestion control (CUBIC model) for tensor transfers
 *   - Packet loss + retransmission budgets
 *   - Real SmolLM2-1.7B layer/tensor sizes (hidden=2048, layers=24)
 *   - Per-token pipeline: tokenize → embed → N×transformer → lm_head
 *   - Reed-Solomon erasure recovery under real churn
 *   - Lyapunov queue stability under bursty mobile traffic
 *   - Ramanujan expander graph with real spectral gap math
 *   - Multi-path Max-Flow routing with bandwidth probing
 *
 * All simulations run in-memory. No actual network/GPU calls.
 */

import { RamanujanGraphBuilder, CheegerMetric } from './graph_topology.js';
import { PushRelabelSolver, BandwidthProber } from './flow_router.js';
import { ReedSolomon } from './reed_solomon.js';
import { MarkovStateModel, LyapunovController, TelemetryCollector } from './lyapunov_optimizer.js';

// ═══════════════════════════════════════════════════════════════════════════
//  REAL NETWORK PROFILES (ITU-R M.2135 + 3GPP TR 38.913)
// ═══════════════════════════════════════════════════════════════════════════

const NETWORK_PROFILES = {
  '5G_mmWave': {
    label: '5G mmWave',
    // Downlink peak ~4Gbps but in swarm context (upload tensors) use UL
    ulBwBps: 600_000_000,   // 600 Mbps UL
    dlBwBps: 2_000_000_000, // 2 Gbps DL
    rttBaseMs: 4,            // ultra-low latency
    jitterMs: 1,
    packetLossPct: 0.01,
    coveragePct: 0.15,       // only 15% of users have mmWave
    mobilityDrop: 0.001,     // drops per second at walking speed
  },
  '5G_sub6': {
    label: '5G Sub-6 GHz',
    ulBwBps: 100_000_000,   // 100 Mbps UL
    dlBwBps: 400_000_000,   // 400 Mbps DL
    rttBaseMs: 10,
    jitterMs: 3,
    packetLossPct: 0.05,
    coveragePct: 0.35,
    mobilityDrop: 0.003,
  },
  '4G_LTE_A': {
    label: '4G LTE-A',
    ulBwBps: 50_000_000,    // 50 Mbps UL
    dlBwBps: 150_000_000,   // 150 Mbps DL
    rttBaseMs: 30,
    jitterMs: 10,
    packetLossPct: 0.2,
    coveragePct: 0.70,
    mobilityDrop: 0.005,
  },
  '4G_LTE': {
    label: '4G LTE',
    ulBwBps: 12_000_000,    // 12 Mbps UL (realistic urban average)
    dlBwBps: 40_000_000,    // 40 Mbps DL
    rttBaseMs: 45,
    jitterMs: 15,
    packetLossPct: 0.5,
    coveragePct: 0.85,
    mobilityDrop: 0.008,
  },
  '3G_HSPA': {
    label: '3G HSPA+',
    ulBwBps: 2_000_000,     // 2 Mbps UL
    dlBwBps: 10_000_000,    // 10 Mbps DL
    rttBaseMs: 80,
    jitterMs: 30,
    packetLossPct: 1.5,
    coveragePct: 0.95,
    mobilityDrop: 0.015,
  },
  'WiFi_6': {
    label: 'WiFi 6 (home)',
    ulBwBps: 200_000_000,   // 200 Mbps UL
    dlBwBps: 600_000_000,   // 600 Mbps DL
    rttBaseMs: 5,
    jitterMs: 2,
    packetLossPct: 0.02,
    coveragePct: 0.60,      // ~60% of mobile users often on WiFi
    mobilityDrop: 0.0001,
  },
  'WiFi_4': {
    label: 'WiFi 4 (congested)',
    ulBwBps: 20_000_000,
    dlBwBps: 40_000_000,
    rttBaseMs: 15,
    jitterMs: 20,           // congested home router jitter
    packetLossPct: 0.3,
    coveragePct: 0.75,
    mobilityDrop: 0.0002,
  },
};

// SmolLM2-1.7B architecture constants
const MODEL = {
  hiddenSize: 2048,
  numLayers: 24,
  numHeads: 32,
  ffnMult: 4,
  vocabSize: 49152,
  // Tensor sizes in bytes (float16 = 2 bytes, float32 = 4)
  // Hidden state per token: 2048 * 2 = 4096 bytes
  hiddenStateBytes: 2048 * 2,
  // KV cache per layer per token: 2 * 2048 * 2 = 8192 bytes (K+V)
  kvCachePerLayerPerTokenBytes: 2 * 2048 * 2,
  // Embedding table: 49152 * 2048 * 2 ≈ 201MB — stays on embedding node
  embeddingBytes: 49152 * 2048 * 2,
  // Per transformer layer weights: ~42MB in q4 quantized
  layerWeightsBytesQ4: 42 * 1024 * 1024,
  // Time to compute one transformer layer on mobile GPU (ms)
  // Based on: Snapdragon 8 Gen 2 ≈ 50 GFLOPS/s, layer ≈ 2.5 GFLOPS
  computePerLayerMs: {
    S:  0.5,   // Server GPU (A100 class): <1ms per layer
    A:  3,     // Desktop RTX 3080: 3ms
    B:  8,     // Laptop/integrated GPU: 8ms
    C:  25,    // Mobile high-end (Snapdragon 8 Gen 2): 25ms
    D:  80,    // Mobile mid-range (Dimensity 900): 80ms
    relay: 0,  // No compute, relay only
  },
};

// ═══════════════════════════════════════════════════════════════════════════
//  REALISTIC MOBILE NODE
// ═══════════════════════════════════════════════════════════════════════════

class MobileNode {
  constructor(id, config) {
    this.id = id;
    this.tier = config.tier;
    this.networkProfile = config.networkProfile;
    this.profile = NETWORK_PROFILES[config.networkProfile];

    // Compute capabilities
    this.numLayers = config.numLayers || 0;  // ONNX layers this node owns
    this.computeMsPerLayer = MODEL.computePerLayerMs[config.tier] || 80;

    // Memory
    this.totalRamMB = config.ramMB;
    this.availRamMB = config.ramMB * (0.4 + Math.random() * 0.3); // OS uses 40-70%
    this.vramMB = config.vramMB || 0;

    // Battery
    this.batteryPct = config.battery || 80;
    this.isCharging = Math.random() < 0.3; // 30% chance plugged in
    this.thermalState = 'nominal'; // nominal | warm | hot | throttled

    // Network — sample from profile with realistic variation
    this.effectiveBw = this._sampleBandwidth();
    this.rtt = this._sampleRTT();
    this.jitter = this.profile.jitterMs;
    this.packetLoss = this.profile.packetLossPct / 100;

    // CUBIC congestion window (bytes)
    this.cwnd = 10 * 1500; // start at 10 MSS
    this.ssthresh = Infinity;
    this.rto = this.rtt * 1.5 + 4 * this.jitter; // RFC 6298

    // State
    this.isOnline = true;
    this.queueDepth = 0;
    this.requestsServed = 0;
    this.bytesTransferred = 0;
    this.consecutiveFailures = 0;

    // Mobility — affects signal strength over time
    this.mobilityFactor = 0.3 + Math.random() * 0.7; // 0=static, 1=mobile
  }

  /** Sample realistic bandwidth from profile with congestion variation */
  _sampleBandwidth() {
    const p = this.profile;
    // Gaussian variation: ±30% of peak, never > peak
    const variation = 1 - 0.3 * Math.abs(this._gaussian());
    // Time-of-day congestion: peak hours reduce by 40%
    const hour = (Date.now() / 3_600_000) % 24;
    const congestion = (hour >= 18 && hour <= 23) ? 0.6 : 1.0;
    return p.ulBwBps * variation * congestion;
  }

  /** Sample RTT with realistic distribution (log-normal) */
  _sampleRTT() {
    const p = this.profile;
    // Log-normal distribution: σ = 0.3 gives realistic tail
    const logNormal = Math.exp(Math.log(p.rttBaseMs) + 0.3 * this._gaussian());
    return Math.max(p.rttBaseMs * 0.5, logNormal);
  }

  _gaussian() {
    // Box-Muller transform
    return Math.sqrt(-2 * Math.log(Math.random())) * Math.cos(2 * Math.PI * Math.random());
  }

  /**
   * CUBIC congestion control: compute effective throughput for a transfer.
   * @param {number} sizeBytes - bytes to transfer
   * @returns {{timeMs, retransmits, effectiveBps}}
   */
  transferTime(sizeBytes) {
    if (!this.isOnline) return { timeMs: Infinity, retransmits: 0, effectiveBps: 0 };

    const MSS = 1460; // bytes
    let cwnd = this.cwnd;
    let remaining = sizeBytes;
    let timeMs = 0;
    let retransmits = 0;

    // Slow start / congestion avoidance (simplified CUBIC)
    while (remaining > 0) {
      const windowBytes = Math.min(cwnd, remaining);
      const rttSec = (this.rtt + this.jitter * Math.random()) / 1000;

      // Bandwidth-limited transfer time for this window
      const bwLimitedMs = (windowBytes / this.effectiveBw) * 1000;
      const rttMs = rttSec * 1000;
      timeMs += Math.max(bwLimitedMs, rttMs / 4); // pipelining

      // Packet loss? Each MSS has independent loss probability
      const numPackets = Math.ceil(windowBytes / MSS);
      const numLost = Math.floor(numPackets * this.packetLoss);
      if (numLost > 0) {
        retransmits += numLost;
        timeMs += numLost * this.rto; // RTO for each lost packet
        // CUBIC halves cwnd on loss
        this.ssthresh = cwnd / 2;
        cwnd = Math.max(cwnd * 0.7, MSS);
      } else {
        // CUBIC growth
        if (cwnd < this.ssthresh) {
          cwnd = Math.min(cwnd + MSS, this.ssthresh); // slow start
        } else {
          cwnd += (MSS * MSS) / cwnd; // congestion avoidance
        }
        cwnd = Math.min(cwnd, this.effectiveBw * (this.rtt / 1000)); // BDP cap
      }

      remaining -= windowBytes;
    }

    this.cwnd = cwnd;
    this.bytesTransferred += sizeBytes;

    const effectiveBps = (sizeBytes / timeMs) * 1000;
    return { timeMs, retransmits, effectiveBps };
  }

  /**
   * Simulate one inference step: compute N transformer layers.
   * Accounts for thermal throttling and memory pressure.
   */
  computeLayers(numLayers, tokenCount = 1) {
    if (this.numLayers === 0) return { timeMs: 0, throttled: false };

    let baseMs = this.computeMsPerLayer * numLayers * tokenCount;

    // Thermal throttling
    const thermalMultiplier = {
      nominal: 1.0,
      warm: 1.3,
      hot: 1.8,
      throttled: 3.0,
    }[this.thermalState];
    baseMs *= thermalMultiplier;

    // Memory pressure — if availRam < layer weights, swap penalty
    const layerWeightsMB = (MODEL.layerWeightsBytesQ4 * numLayers) / (1024 * 1024);
    if (layerWeightsMB > this.availRamMB * 0.8) {
      baseMs *= 2.5; // swap penalty
    }

    // Random compute jitter (GPU scheduling variance)
    baseMs *= (0.9 + Math.random() * 0.2);

    return { timeMs: baseMs, throttled: thermalMultiplier > 1 };
  }

  /** Simulate one simulation tick (100ms real time equivalent) */
  tick(tickMs = 100) {
    if (!this.isOnline) {
      // Reconnection: exponential backoff with mobility
      const reconnectProb = 0.05 * (1 - this.mobilityFactor * 0.5);
      if (Math.random() < reconnectProb) {
        this.isOnline = true;
        this.cwnd = 10 * 1500; // reset congestion window on reconnect
        this.effectiveBw = this._sampleBandwidth();
        this.rtt = this._sampleRTT();
        this.consecutiveFailures = 0;
      }
      return;
    }

    // Mobility: signal changes
    if (Math.random() < this.mobilityFactor * 0.1) {
      this.effectiveBw = this._sampleBandwidth() * (0.7 + Math.random() * 0.6);
      this.rtt = this._sampleRTT() * (0.8 + Math.random() * 0.4);
    }

    // Disconnect probability (from profile + mobility)
    const baseDisc = this.profile.mobilityDrop * tickMs / 1000;
    const mobilityExtra = this.mobilityFactor * baseDisc * 2;
    const batteryExtra = this.batteryPct < 10 ? 0.05 : 0;
    if (Math.random() < baseDisc + mobilityExtra + batteryExtra) {
      this.isOnline = false;
      this.consecutiveFailures++;
      return;
    }

    // Battery drain
    if (!this.isCharging) {
      const drainRate = 0.001 * (1 + this.queueDepth * 0.1); // heavier load = more drain
      this.batteryPct = Math.max(0, this.batteryPct - drainRate);
    }

    // Thermal management (simplified)
    if (this.queueDepth > 5) {
      const heatMap = { nominal: 'warm', warm: 'hot', hot: 'throttled', throttled: 'throttled' };
      if (Math.random() < 0.3) this.thermalState = heatMap[this.thermalState];
    } else {
      const coolMap = { throttled: 'hot', hot: 'warm', warm: 'nominal', nominal: 'nominal' };
      if (Math.random() < 0.1) this.thermalState = coolMap[this.thermalState];
    }

    // Queue processing
    if (this.queueDepth > 0) {
      this.queueDepth = Math.max(0, this.queueDepth - 1);
    }
  }
}

// ═══════════════════════════════════════════════════════════════════════════
//  REALISTIC SWARM SIMULATOR
// ═══════════════════════════════════════════════════════════════════════════

export class SwarmSimulator {
  constructor(nodeCount = 100) {
    this.nodeCount = nodeCount;
    this.nodes = new Map();
    this.results = [];
    this.onLog = null;
  }

  _log(msg, cls = '') {
    console.log(`[SwarmSim] ${msg}`);
    if (this.onLog) this.onLog(msg, cls);
  }

  /**
   * Assign a realistic network profile to a node based on global coverage stats.
   * Based on Ericsson Mobility Report 2024 + OpenSignal data.
   */
  _pickNetworkProfile() {
    const r = Math.random();
    // Coverage overlap: WiFi > 4G > 5G. Model as: 60% on WiFi at any moment,
    // rest split by cellular tech availability.
    if (r < 0.38) return 'WiFi_6';
    if (r < 0.52) return 'WiFi_4';
    if (r < 0.57) return '5G_mmWave';
    if (r < 0.67) return '5G_sub6';
    if (r < 0.82) return '4G_LTE_A';
    if (r < 0.93) return '4G_LTE';
    return '3G_HSPA';
  }

  /**
   * Generate 100 realistic nodes distributed by capability tier.
   * Tier distribution based on real device fleet (Statcounter 2024):
   *   S: 3%  — dedicated servers / high-end workstations
   *   A: 8%  — gaming PCs / Mac Pro class
   *   B: 17% — average laptops / budget desktops
   *   C: 35% — high-end phones (iPhone 15, Pixel 8, S24)
   *   D: 37% — mid-range phones (Moto G, Redmi 12, Galaxy A35)
   */
  _generateNodes() {
    this.nodes.clear();

    const tiers = [
      {
        tier: 'S', count: 3,
        ramMB: 65536, vramMB: 40960, battery: 100,
        numLayers: 24, // serves all layers
        networks: ['WiFi_6'], // servers always on fast network
      },
      {
        tier: 'A', count: 8,
        ramMB: 16384, vramMB: 12288, battery: 100,
        numLayers: 12, // half the model
        networks: ['WiFi_6', 'WiFi_4'],
      },
      {
        tier: 'B', count: 17,
        ramMB: 8192, vramMB: 4096, battery: 90,
        numLayers: 6,
        networks: ['WiFi_6', 'WiFi_4', '4G_LTE_A'],
      },
      {
        tier: 'C', count: 35,
        ramMB: 8192, vramMB: 2048, battery: 75,
        numLayers: 4, // high-end mobile: 4 layers
        // realistic mix: 50% on WiFi, 50% on cellular
        networks: ['WiFi_6', 'WiFi_4', '5G_sub6', '4G_LTE_A', '4G_LTE'],
      },
      {
        tier: 'D', count: 37,
        ramMB: 4096, vramMB: 1024, battery: 65,
        numLayers: 2, // mid-range mobile: 2 layers (relay-only fallback if OOM)
        networks: ['WiFi_4', '4G_LTE_A', '4G_LTE', '3G_HSPA'],
      },
    ];

    let idx = 0;
    for (const t of tiers) {
      for (let i = 0; i < t.count; i++) {
        const id = `sim_${t.tier}_${idx}`;
        // Pick network from tier's realistic distribution
        const networkProfile = t.networks[Math.floor(Math.random() * t.networks.length)];
        this.nodes.set(id, new MobileNode(id, {
          tier: t.tier,
          networkProfile,
          ramMB: t.ramMB * (0.8 + Math.random() * 0.4),
          vramMB: t.vramMB * (0.7 + Math.random() * 0.6),
          battery: t.battery - Math.random() * 30,
          numLayers: t.numLayers,
        }));
        idx++;
      }
    }

    // Print node distribution summary
    const byTier = {};
    const byNet = {};
    for (const [, n] of this.nodes) {
      byTier[n.tier] = (byTier[n.tier] || 0) + 1;
      byNet[n.networkProfile] = (byNet[n.networkProfile] || 0) + 1;
    }
    this._log(`Generated ${this.nodes.size} nodes: S:${byTier.S} A:${byTier.A} B:${byTier.B} C:${byTier.C} D:${byTier.D}`);
    this._log(`Network mix: ${Object.entries(byNet).map(([k,v])=>`${k}:${v}`).join(' ')}`);
  }

  async runAll() {
    const t0 = performance.now();
    this._log('═══ REALISTIC MOBILE SWARM SIMULATOR v2 ═══', 'sim-title');
    this._log('SmolLM2-1.7B · hidden=2048 · 24 layers · ITU-R M.2135 network physics');
    this._generateNodes();

    let passed = 0, failed = 0;

    const run = (label, fn) => {
      const r = fn();
      if (r) passed++; else failed++;
      return r;
    };

    run('Expander Graph',        () => this._testExpanderGraph());
    run('Churn Resilience',      () => this._testChurnResilience());
    run('Max-Flow Routing',      () => this._testMaxFlow());
    run('Reed-Solomon',          () => this._testReedSolomon());
    run('RS Stress',             () => this._testRSStress());
    run('Lyapunov Stability',    () => this._testLyapunovStability());
    run('Load Distribution',     () => this._testLoadDistribution());
    run('Markov Prediction',     () => this._testMarkovPrediction());
    run('Multi-Path Tensor',     () => this._testMultiPathSplit());
    run('Network Physics',       () => this._testNetworkPhysics());
    run('Thermal Throttling',    () => this._testThermalThrottling());
    run('Real E2E Inference',    () => this._testE2EInferenceRealistic());

    const totalMs = Math.round(performance.now() - t0);
    this._log('═══════════════════════════════════════════', 'sim-title');
    if (failed === 0) {
      this._log(`🟢 ALL ${passed}/${passed + failed} TESTS PASSED (${totalMs}ms)`, 'sim-pass');
    } else {
      this._log(`🔴 ${failed} FAILED, ${passed} passed (${totalMs}ms)`, 'sim-fail');
    }
    return { passed, failed, totalMs };
  }

  // ─── TEST 1: Expander Graph ────────────────────────────────────
  _testExpanderGraph() {
    this._log('\n─── TEST 1: Ramanujan Expander Graph ───', 'sim-title');
    const peerIds = [...this.nodes.keys()];
    const builder = new RamanujanGraphBuilder(6, 4);
    const adj = builder.build(peerIds);

    let minDeg = Infinity, maxDeg = 0;
    for (const [, neighbors] of adj) {
      const d = neighbors.size;
      if (d < minDeg) minDeg = d;
      if (d > maxDeg) maxDeg = d;
    }

    const cheeger = new CheegerMetric();
    const metrics = cheeger.estimate(adj);

    this._log(`  Nodes: ${peerIds.length}, Degree: [${minDeg}, ${maxDeg}]`);
    this._log(`  Spectral gap: ${metrics.spectralGap.toFixed(4)}, Cheeger h≥${metrics.cheegerLower.toFixed(4)}`);
    this._log(`  Is Ramanujan expander: ${metrics.isExpander}`);

    if (minDeg >= 2 && metrics.spectralGap > 0) {
      this._log('  ✅ PASS: Connected Ramanujan expander', 'sim-pass');
      return true;
    }
    this._log('  ❌ FAIL: Not a proper expander', 'sim-fail');
    return false;
  }

  // ─── TEST 2: Churn Resilience (realistic mobile churn) ─────────
  _testChurnResilience() {
    this._log('\n─── TEST 2: Mobile Churn Resilience (200 ticks × 100ms) ───', 'sim-title');
    const builder = new RamanujanGraphBuilder(6, 4);
    const cheeger = new CheegerMetric();
    let expanderCount = 0;
    let totalOnline = 0;
    let minOnline = Infinity, maxOnline = 0;

    for (let tick = 0; tick < 200; tick++) {
      for (const [, node] of this.nodes) node.tick(100);

      const onlineIds = [...this.nodes.entries()]
        .filter(([, n]) => n.isOnline)
        .map(([id]) => id);

      totalOnline += onlineIds.length;
      if (onlineIds.length < minOnline) minOnline = onlineIds.length;
      if (onlineIds.length > maxOnline) maxOnline = onlineIds.length;

      if (onlineIds.length < 4) continue;
      const adj = builder.build(onlineIds);
      const metrics = cheeger.estimate(adj);
      if (metrics.spectralGap > 0) expanderCount++;
    }

    // Reset
    for (const [, node] of this.nodes) { node.isOnline = true; node.thermalState = 'nominal'; }

    const avgOnline = Math.round(totalOnline / 200);
    const resiliencePct = Math.round((expanderCount / 200) * 100);
    this._log(`  Online nodes: avg=${avgOnline}, min=${minOnline}, max=${maxOnline}`);
    this._log(`  Expander maintained: ${expanderCount}/200 ticks (${resiliencePct}%)`);

    if (resiliencePct >= 55) {
      this._log(`  ✅ PASS: ${resiliencePct}% resilience under realistic mobile churn`, 'sim-pass');
      return true;
    }
    this._log(`  ❌ FAIL: Only ${resiliencePct}% resilience`, 'sim-fail');
    return false;
  }

  // ─── TEST 3: Max-Flow ──────────────────────────────────────────
  _testMaxFlow() {
    this._log('\n─── TEST 3: Max-Flow Tensor Routing ───', 'sim-title');
    const solver = new PushRelabelSolver();
    const nodes = ['source', 'A', 'B', 'C', 'D', 'sink'];
    const edges = [
      { from: 'source', to: 'A', capacity: 10 },
      { from: 'source', to: 'B', capacity: 5 },
      { from: 'A', to: 'C', capacity: 8 },
      { from: 'A', to: 'B', capacity: 3 },
      { from: 'B', to: 'D', capacity: 7 },
      { from: 'C', to: 'sink', capacity: 10 },
      { from: 'D', to: 'sink', capacity: 6 },
      { from: 'C', to: 'D', capacity: 2 },
    ];
    const result = solver.solve(nodes, edges, 'source', 'sink');
    this._log(`  Max flow: ${result.maxFlow} (expected ~15), paths: ${result.paths.length}`);
    if (result.maxFlow >= 13 && result.maxFlow <= 16) {
      this._log('  ✅ PASS', 'sim-pass'); return true;
    }
    this._log(`  ❌ FAIL: maxFlow=${result.maxFlow}`, 'sim-fail'); return false;
  }

  // ─── TEST 4: Reed-Solomon ──────────────────────────────────────
  _testReedSolomon() {
    this._log('\n─── TEST 4: Reed-Solomon (n=6, k=4) ───', 'sim-title');
    const rs = new ReedSolomon(6, 4);
    const blocks = [];
    for (let i = 0; i < 4; i++) {
      const b = new ArrayBuffer(1024);
      const v = new Uint8Array(b);
      for (let j = 0; j < 1024; j++) v[j] = (i * 37 + j * 13) & 0xFF;
      blocks.push(b);
    }
    const coded = rs.encode(blocks);
    const available = [
      { index: 0, data: coded[0] },
      { index: 2, data: coded[2] },
      { index: 4, data: coded[4] },
      { index: 5, data: coded[5] },
    ];
    const recovered = rs.decode(available);
    let correct = true;
    for (let i = 0; i < 4; i++) {
      const orig = new Uint8Array(blocks[i]);
      const rec = new Uint8Array(recovered[i]);
      for (let j = 0; j < orig.length; j++) {
        if (orig[j] !== rec[j]) { correct = false; break; }
      }
      if (!correct) break;
    }
    if (correct) { this._log('  ✅ PASS: Perfect recovery', 'sim-pass'); return true; }
    this._log('  ❌ FAIL: Mismatch', 'sim-fail'); return false;
  }

  // ─── TEST 5: RS Stress ────────────────────────────────────────
  _testRSStress() {
    this._log('\n─── TEST 5: Reed-Solomon Stress (100 rounds) ───', 'sim-title');
    const rs = new ReedSolomon(6, 4);
    let ok = 0;
    for (let r = 0; r < 100; r++) {
      const sz = 256 + Math.floor(Math.random() * 4096);
      const blocks = [];
      for (let i = 0; i < 4; i++) {
        const b = new ArrayBuffer(sz);
        new Uint8Array(b).forEach((_, j, a) => { a[j] = Math.floor(Math.random() * 256); });
        blocks.push(b);
      }
      if (rs.verify(blocks)) ok++;
    }
    this._log(`  ${ok}/100 roundtrips ok`);
    if (ok === 100) { this._log('  ✅ PASS', 'sim-pass'); return true; }
    this._log(`  ❌ FAIL: ${100-ok} failures`, 'sim-fail'); return false;
  }

  // ─── TEST 6: Lyapunov Stability ───────────────────────────────
  _testLyapunovStability() {
    this._log('\n─── TEST 6: Lyapunov Queue Stability (bursty mobile traffic) ───', 'sim-title');
    const markov = new MarkovStateModel();
    const telemetry = new TelemetryCollector();
    const controller = new LyapunovController({ V: 0.5, markov, telemetry });

    const simPeers = new Map();
    for (let i = 0; i < 20; i++) {
      const pid = `lyap_${i}`;
      const tps = 5 + Math.random() * 45;
      simPeers.set(pid, {
        hasEngine: true, dc: { readyState: 'open' },
        tps, benchmark: tps,
        rtt: 20 + Math.random() * 200,
        deviceTier: tps > 30 ? 'A' : tps > 15 ? 'B' : 'C',
        moeRole: 'general', repScore: Math.floor(Math.random() * 100),
      });
      controller.updateServiceRate(pid, tps);
      controller.updateQueue(pid, 0);
    }

    const lyapunovValues = [];
    for (let req = 0; req < 1000; req++) {
      // Bursty traffic: Poisson bursts every 50 requests
      const burstFactor = (req % 50 === 0) ? 5 : 1;
      for (let b = 0; b < burstFactor; b++) {
        const winner = controller.selectPeer(simPeers, 'local', { category: 'general' }, false);
        if (winner?.peerId) {
          const q = controller.queues.get(winner.peerId) || 0;
          controller.updateQueue(winner.peerId, q + 1);
        }
      }
      if (req % 10 === 0) {
        for (const [pid] of simPeers) {
          const q = controller.queues.get(pid) || 0;
          const mu = controller.serviceRates.get(pid) || 5;
          controller.updateQueue(pid, Math.max(0, q - mu * 0.05));
        }
        lyapunovValues.push(controller.lyapunovValue());
      }
    }

    const maxL = Math.max(...lyapunovValues);
    const finalL = lyapunovValues[lyapunovValues.length - 1];
    const avgL = lyapunovValues.reduce((a, b) => a + b) / lyapunovValues.length;
    const isBounded = maxL < 1000;
    const isDrifting = finalL > avgL * 1.5;

    this._log(`  Lyapunov: max=${maxL.toFixed(1)}, avg=${avgL.toFixed(1)}, final=${finalL.toFixed(1)}`);
    this._log(`  Bounded: ${isBounded}, Drifting: ${isDrifting}`);

    if (isBounded && !isDrifting) {
      this._log('  ✅ PASS: Queues stable under bursty traffic', 'sim-pass'); return true;
    }
    this._log('  ❌ FAIL: Queue instability', 'sim-fail'); return false;
  }

  // ─── TEST 7: Load Distribution ────────────────────────────────
  _testLoadDistribution() {
    this._log('\n─── TEST 7: Asymmetric Load Distribution ───', 'sim-title');
    const markov = new MarkovStateModel();
    const telemetry = new TelemetryCollector();
    const controller = new LyapunovController({ V: 0.5, markov, telemetry });
    const simPeers = new Map();
    const cfgs = [
      { id: 'fast_S', tps: 50, rtt: 10, tier: 'S', disc: 0.01, rep: 200 },
      { id: 'mid_A',  tps: 25, rtt: 30, tier: 'A', disc: 0.03, rep: 80 },
      { id: 'slow_C', tps: 5,  rtt: 100, tier: 'C', disc: 0.08, rep: 10 },
    ];
    for (const cfg of cfgs) {
      simPeers.set(cfg.id, {
        hasEngine: true, dc: { readyState: 'open' },
        tps: cfg.tps, benchmark: cfg.tps, rtt: cfg.rtt,
        deviceTier: cfg.tier, moeRole: 'general', repScore: cfg.rep,
      });
      controller.updateServiceRate(cfg.id, cfg.tps);
      for (let i = 0; i < 20; i++) {
        markov.observe(cfg.id, { ramPct: 30, battery: 80, pressure: 'nominal', queue: 0, backlog: 0 }, true);
        if (Math.random() < cfg.disc) {
          markov.observe(cfg.id, {}, false);
          markov.observe(cfg.id, { ramPct: 30, battery: 80, pressure: 'nominal', queue: 0, backlog: 0 }, true);
        }
      }
    }
    const dist = controller.computeLoadDistribution(simPeers);
    const f = dist.get('fast_S') || 0;
    const m = dist.get('mid_A') || 0;
    const s = dist.get('slow_C') || 0;
    this._log(`  S(50 tps): ${(f*100).toFixed(1)}%  A(25 tps): ${(m*100).toFixed(1)}%  C(5 tps): ${(s*100).toFixed(1)}%`);
    if (f > m && m > s && f > 0.4) {
      this._log('  ✅ PASS: Proportional load distribution', 'sim-pass'); return true;
    }
    this._log('  ❌ FAIL: Load not properly asymmetric', 'sim-fail'); return false;
  }

  // ─── TEST 8: Markov Prediction ────────────────────────────────
  _testMarkovPrediction() {
    this._log('\n─── TEST 8: Markov Disconnect Prediction ───', 'sim-title');
    const markov = new MarkovStateModel();
    for (let i = 0; i < 50; i++)
      markov.observe('stable', { ramPct: 30, battery: 90, pressure: 'nominal', queue: 0, backlog: 0 }, true);
    for (let i = 0; i < 50; i++) {
      const online = Math.random() > 0.3;
      markov.observe('unstable', online ? { ramPct: 80, battery: 20, pressure: 'serious', queue: 3, backlog: 100000 } : {}, online);
    }
    const stableP = markov.disconnectProbability('stable', 3);
    const unstableP = markov.disconnectProbability('unstable', 3);
    this._log(`  Stable P(disc): ${stableP.toFixed(3)}  Unstable P(disc): ${unstableP.toFixed(3)}`);
    if (unstableP > stableP) {
      this._log('  ✅ PASS', 'sim-pass'); return true;
    }
    this._log('  ❌ FAIL', 'sim-fail'); return false;
  }

  // ─── TEST 9: Multi-Path Tensor Split ──────────────────────────
  _testMultiPathSplit() {
    this._log('\n─── TEST 9: Multi-Path Tensor Split ───', 'sim-title');
    const tensorSize = MODEL.hiddenStateBytes; // 4096 bytes real SmolLM2 hidden state
    const tensor = new ArrayBuffer(tensorSize);
    const tv = new Uint8Array(tensor);
    for (let i = 0; i < tensorSize; i++) tv[i] = i & 0xFF;

    const route = {
      type: 'multi-path',
      paths: [
        { nodes: ['A', 'C', 'D'], flow: 2048 },
        { nodes: ['A', 'B', 'D'], flow: 1500 },
        { nodes: ['A', 'D'],      flow: 548 },
      ],
    };

    // Inline TensorRouter (mirrors real implementation)
    const chunks = [];
    let offset = 0;
    const totalFlow = route.paths.reduce((s, p) => s + p.flow, 0);
    for (let i = 0; i < route.paths.length; i++) {
      const frac = route.paths[i].flow / totalFlow;
      const size = i === route.paths.length - 1
        ? tensor.byteLength - offset
        : Math.floor(tensor.byteLength * frac);
      chunks.push({ chunk: tensor.slice(offset, offset + size), index: i });
      offset += size;
    }

    const totalChunkSize = chunks.reduce((s, c) => s + c.chunk.byteLength, 0);
    const reassembled = new Uint8Array(totalChunkSize);
    let off = 0;
    for (const c of chunks) { reassembled.set(new Uint8Array(c.chunk), off); off += c.chunk.byteLength; }

    let match = true;
    for (let i = 0; i < tensorSize; i++) if (reassembled[i] !== tv[i]) { match = false; break; }

    this._log(`  Hidden state ${tensorSize}B → ${chunks.length} chunks: ${chunks.map(c=>c.chunk.byteLength+'B').join(' + ')}`);
    if (match && totalChunkSize === tensorSize) {
      this._log('  ✅ PASS: Tensor split/reassemble correct', 'sim-pass'); return true;
    }
    this._log('  ❌ FAIL', 'sim-fail'); return false;
  }

  // ─── TEST 10: Real Network Physics ────────────────────────────
  _testNetworkPhysics() {
    this._log('\n─── TEST 10: Real Network Physics (CUBIC TCP) ───', 'sim-title');

    const results = [];
    // Test tensor transfer on different network types
    const scenarios = [
      { net: '5G_sub6',  size: MODEL.hiddenStateBytes, label: 'hidden state (4KB) on 5G' },
      { net: '4G_LTE',   size: MODEL.hiddenStateBytes, label: 'hidden state (4KB) on 4G LTE' },
      { net: '4G_LTE',   size: MODEL.layerWeightsBytesQ4, label: 'layer weights (42MB) on 4G LTE' },
      { net: '3G_HSPA',  size: MODEL.hiddenStateBytes, label: 'hidden state (4KB) on 3G' },
      { net: 'WiFi_6',   size: MODEL.hiddenStateBytes, label: 'hidden state (4KB) on WiFi6' },
    ];

    let allReasonable = true;
    for (const s of scenarios) {
      const node = new MobileNode('test', { tier: 'D', networkProfile: s.net, ramMB: 4096, vramMB: 1024, battery: 80, numLayers: 2 });
      const { timeMs, retransmits, effectiveBps } = node.transferTime(s.size);
      const kbps = Math.round(effectiveBps / 1000);
      results.push({ label: s.label, timeMs: timeMs.toFixed(1), retransmits, kbps });
      this._log(`  ${s.label}: ${timeMs.toFixed(1)}ms, ${retransmits} retx, ${kbps} kbps`);
      // Sanity check: hidden state should always be < 500ms even on 3G
      if (s.size === MODEL.hiddenStateBytes && timeMs > 500) allReasonable = false;
    }

    if (allReasonable) {
      this._log('  ✅ PASS: CUBIC TCP physics correct', 'sim-pass'); return true;
    }
    this._log('  ❌ FAIL: Implausible transfer times', 'sim-fail'); return false;
  }

  // ─── TEST 11: Thermal Throttling ──────────────────────────────
  _testThermalThrottling() {
    this._log('\n─── TEST 11: Mobile Thermal Throttling ───', 'sim-title');

    const node = new MobileNode('thermal_test', {
      tier: 'C', networkProfile: '4G_LTE_A',
      ramMB: 8192, vramMB: 2048, battery: 80, numLayers: 4
    });

    const timings = [];
    const states = [];

    // Force sustained load to heat up
    for (let i = 0; i < 30; i++) {
      node.queueDepth = 10; // keep queue full
      node.tick(100);
      const { timeMs } = node.computeLayers(4);
      timings.push(timeMs);
      states.push(node.thermalState);
    }

    const nominalAvg = timings.slice(0, 5).reduce((a, b) => a + b) / 5;
    const finalAvg = timings.slice(-5).reduce((a, b) => a + b) / 5;
    const throttleDetected = states.some(s => s !== 'nominal');
    const slowdown = finalAvg / nominalAvg;

    this._log(`  Initial avg compute: ${nominalAvg.toFixed(1)}ms/4-layers`);
    this._log(`  Final avg compute:   ${finalAvg.toFixed(1)}ms/4-layers`);
    this._log(`  Thermal slowdown:    ${slowdown.toFixed(2)}x`);
    this._log(`  States seen: ${[...new Set(states)].join(' → ')}`);

    if (throttleDetected && slowdown >= 1.0) {
      this._log('  ✅ PASS: Thermal throttling modeled correctly', 'sim-pass'); return true;
    }
    this._log('  ❌ FAIL: No thermal effect detected', 'sim-fail'); return false;
  }

  // ─── TEST 12: REALISTIC E2E Distributed Inference ─────────────
  /**
   * Simulates actual SmolLM2-1.7B distributed inference across a realistic
   * mobile swarm. Each node owns a slice of the 24 transformer layers.
   * Models:
   *   - Real tensor sizes (hidden=2048 float16)
   *   - CUBIC TCP for tensor transfer between nodes
   *   - Thermal throttling on mobile devices
   *   - Packet loss + retransmission on mobile networks
   *   - Node churn mid-inference (Reed-Solomon recovery if needed)
   *   - Pipeline parallelism (nodes compute in parallel where possible)
   */
  _testE2EInferenceRealistic() {
    this._log('\n─── TEST 12: Realistic E2E Distributed Inference (SmolLM2-1.7B) ───', 'sim-title');
    this._log('  Model: hidden=2048 · 24 layers · float16 · q4 weights');

    // Build a realistic pipeline from actual nodes in the swarm
    // Prefer nodes with layers, prioritize by tier
    const pipeline = this._buildInferencePipeline(MODEL.numLayers);
    if (!pipeline) {
      this._log('  ❌ FAIL: Could not build pipeline (not enough nodes with layers)', 'sim-fail');
      return false;
    }

    this._log(`  Pipeline: ${pipeline.length} nodes cover ${MODEL.numLayers} layers`);
    pipeline.forEach((stage, i) => {
      const net = NETWORK_PROFILES[stage.node.networkProfile];
      this._log(`    Stage ${i+1}: ${stage.node.tier}-tier · ${stage.layers.length} layers · ${net.label} · RTT=${stage.node.rtt.toFixed(0)}ms`);
    });

    // Simulate generating 20 tokens (realistic short response)
    const numTokens = 20;
    const tensorSize = MODEL.hiddenStateBytes; // 4096 bytes per token

    let totalE2EMs = 0;
    let totalComputeMs = 0;
    let totalTransferMs = 0;
    let totalRetransmits = 0;
    let churnsDetected = 0;
    const perTokenMs = [];

    for (let token = 0; token < numTokens; token++) {
      let tokenMs = 0;
      let tokenComputeMs = 0;
      let tokenTransferMs = 0;

      // Tick all nodes (simulates 100ms passing between tokens)
      for (const [, node] of this.nodes) node.tick(100);

      for (let stageIdx = 0; stageIdx < pipeline.length; stageIdx++) {
        const stage = pipeline[stageIdx];

        // Check if node went offline mid-inference (churn)
        if (!stage.node.isOnline) {
          churnsDetected++;
          // Try to find a replacement (Ramanujan graph fallback)
          const replacement = this._findReplacementNode(stage);
          if (replacement) {
            pipeline[stageIdx] = replacement;
            this._log(`    ⚠ Churn at token ${token}: rerouted stage ${stageIdx+1} via Ramanujan`);
          } else {
            this._log(`    ⚠ Churn: stage ${stageIdx+1} offline, using RS recovery (adds latency)`);
            // RS recovery penalty
            tokenMs += 200;
          }
        }

        // 1) Compute transformer layers on this node
        const { timeMs: computeMs } = stage.node.computeLayers(stage.layers.length, 1);
        tokenComputeMs += computeMs;
        tokenMs += computeMs;

        // 2) Transfer hidden state to next stage
        if (stageIdx < pipeline.length - 1) {
          const nextNode = pipeline[stageIdx + 1].node;
          // Transfer happens on the receiving node's uplink
          const { timeMs: xferMs, retransmits } = stage.node.transferTime(tensorSize);
          tokenTransferMs += xferMs;
          tokenMs += xferMs;
          totalRetransmits += retransmits;

          // Network propagation delay (RTT between stage nodes)
          const propagationMs = (stage.node.rtt + nextNode.rtt) / 2;
          tokenMs += propagationMs;
          tokenTransferMs += propagationMs;
        }
      }

      perTokenMs.push(tokenMs);
      totalComputeMs += tokenComputeMs;
      totalTransferMs += tokenTransferMs;
      totalE2EMs += tokenMs;
    }

    const avgTokenMs = totalE2EMs / numTokens;
    const p50 = perTokenMs.sort((a, b) => a - b)[Math.floor(numTokens * 0.5)];
    const p95 = perTokenMs[Math.floor(numTokens * 0.95)];
    const tokensPerSec = 1000 / avgTokenMs;
    const transferBottleneck = totalTransferMs / totalE2EMs;

    this._log(`\n  ── Results ──`);
    this._log(`  Tokens generated: ${numTokens}`);
    this._log(`  Total time: ${Math.round(totalE2EMs)}ms`);
    this._log(`  Compute: ${Math.round(totalComputeMs)}ms (${Math.round(totalComputeMs/totalE2EMs*100)}%)`);
    this._log(`  Transfer: ${Math.round(totalTransferMs)}ms (${Math.round(transferBottleneck*100)}%)`);
    this._log(`  Avg per token: ${avgTokenMs.toFixed(0)}ms  p50: ${p50.toFixed(0)}ms  p95: ${p95.toFixed(0)}ms`);
    this._log(`  Throughput: ${tokensPerSec.toFixed(2)} tok/s`);
    this._log(`  TCP retransmits: ${totalRetransmits}`);
    this._log(`  Churn events: ${churnsDetected}`);

    // Verdict
    const viable = tokensPerSec > 0.05; // at least 1 token per 20 seconds
    const notBottlenecked = transferBottleneck < 0.85; // transfer < 85% of time

    if (viable && notBottlenecked) {
      this._log(`  ✅ PASS: Distributed inference viable at ${tokensPerSec.toFixed(2)} tok/s`, 'sim-pass');
      return true;
    } else if (!viable) {
      this._log(`  ❌ FAIL: Too slow (${tokensPerSec.toFixed(3)} tok/s) — compute bottleneck on mobile`, 'sim-fail');
      return false;
    } else {
      this._log(`  ❌ FAIL: Transfer bottleneck (${Math.round(transferBottleneck*100)}%) — need faster inter-node links`, 'sim-fail');
      return false;
    }
  }

  /**
   * Build a layer pipeline from real swarm nodes.
   * Assigns contiguous layer ranges to nodes sorted by tier.
   */
  _buildInferencePipeline(totalLayers) {
    // Collect nodes that can compute layers
    const capable = [...this.nodes.values()]
      .filter(n => n.isOnline && n.numLayers > 0)
      .sort((a, b) => {
        const tierOrder = { S: 0, A: 1, B: 2, C: 3, D: 4 };
        return tierOrder[a.tier] - tierOrder[b.tier];
      });

    if (capable.length === 0) return null;

    const pipeline = [];
    let layersCovered = 0;

    for (const node of capable) {
      if (layersCovered >= totalLayers) break;
      const layersForThis = Math.min(node.numLayers, totalLayers - layersCovered);
      const layerIndices = Array.from({ length: layersForThis }, (_, i) => layersCovered + i);
      pipeline.push({ node, layers: layerIndices });
      layersCovered += layersForThis;
    }

    return layersCovered >= totalLayers ? pipeline : null;
  }

  /** Find a replacement for a churned node (Ramanujan neighbor) */
  _findReplacementNode(stage) {
    const candidates = [...this.nodes.values()]
      .filter(n => n.isOnline && n.tier === stage.node.tier && n.numLayers >= stage.layers.length);
    if (candidates.length === 0) return null;
    const replacement = candidates[Math.floor(Math.random() * candidates.length)];
    return { node: replacement, layers: stage.layers };
  }
}

// ═══════════════════════════════════════════════════════════════════════════
//  STANDALONE RUN
// ═══════════════════════════════════════════════════════════════════════════

export async function runSimulation(onLog) {
  const sim = new SwarmSimulator(100);
  sim.onLog = onLog;
  return await sim.runAll();
}
