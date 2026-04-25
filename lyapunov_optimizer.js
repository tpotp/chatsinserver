/**
 * lyapunov_optimizer.js — Markov Chain State Model + Lyapunov Drift Optimization
 * 
 * Models the P2P swarm as a stochastic system and optimizes inference
 * routing to prevent queue instabilities on heterogeneous devices.
 * 
 * Mathematical foundations:
 *   - Discrete-time Markov Chain with states {IDLE, COMPUTING, TRANSMITTING, OVERLOADED, DISCONNECTED}
 *   - Lyapunov drift-plus-penalty framework for queue stability
 *   - Asymmetric load assignment inversely proportional to disconnect probability
 * 
 * The Lyapunov function L(Q) = Σᵢ Qᵢ² measures total queue pressure.
 * At each decision epoch, we minimize:
 *   Δ(Q) + V × cost(decision)
 * where Δ is the expected drift and V is the throughput-stability tradeoff parameter.
 * 
 * Integration:
 *   - Replaces/augments existing `routeRequest()` for distributed inference
 *   - Telemetry piggybacks on existing `publishPresence()` events
 *   - Queue depth tracked via existing `fedPending` Map
 */

// ═══════════════════════════════════════════════════════════════════════════
//  PEER STATES (Markov Chain)
// ═══════════════════════════════════════════════════════════════════════════

export const PeerState = {
  IDLE:          0,
  COMPUTING:     1,
  TRANSMITTING:  2,
  OVERLOADED:    3,
  DISCONNECTED:  4,
};

const STATE_NAMES = ['IDLE', 'COMPUTING', 'TRANSMITTING', 'OVERLOADED', 'DISCONNECTED'];


// ═══════════════════════════════════════════════════════════════════════════
//  TELEMETRY COLLECTOR
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Collects local device metrics and receives remote metrics via Nostr/WebRTC.
 * Metrics are used to estimate Markov transition probabilities and
 * inform the Lyapunov controller.
 */
export class TelemetryCollector {
  constructor() {
    this.local = {
      ramUsagePct: 0,
      batteryPct: 100,
      batteryCharging: true,
      cpuPressure: 'nominal', // nominal | fair | serious | critical
      queueDepth: 0,
      avgInferenceMs: 0,
      tensorBacklog: 0,       // bytes waiting to send
      inferenceCount: 0,
      uptimeMs: 0,
      lastUpdate: Date.now(),
    };

    this.remote = new Map(); // peerId -> telemetry object
    this._inferenceTimings = []; // last N inference durations
    this._startTime = Date.now();
    this._batteryAPI = null;
    this._pressureObserver = null;
  }

  /**
   * Start collecting local telemetry.
   */
  async start() {
    // Battery API
    try {
      if (navigator.getBattery) {
        this._batteryAPI = await navigator.getBattery();
        this._updateBattery();
        this._batteryAPI.addEventListener('levelchange', () => this._updateBattery());
        this._batteryAPI.addEventListener('chargingchange', () => this._updateBattery());
      }
    } catch(e) { /* Battery API may not be available */ }

    // Compute Pressure API (Chrome 115+)
    try {
      if ('PressureObserver' in window) {
        this._pressureObserver = new PressureObserver((records) => {
          const latest = records[records.length - 1];
          if (latest) this.local.cpuPressure = latest.state;
        }, { sampleInterval: 2000 });
        await this._pressureObserver.observe('cpu');
      }
    } catch(e) { /* Compute Pressure may not be available */ }

    // Periodic local metrics update
    setInterval(() => this._updateLocal(), 3000);
  }

  _updateBattery() {
    if (this._batteryAPI) {
      this.local.batteryPct = Math.round(this._batteryAPI.level * 100);
      this.local.batteryCharging = this._batteryAPI.charging;
    }
  }

  _updateLocal() {
    this.local.uptimeMs = Date.now() - this._startTime;
    this.local.lastUpdate = Date.now();

    // Estimate RAM usage from performance.memory (Chrome only)
    if (performance.memory) {
      const used = performance.memory.usedJSHeapSize;
      const total = performance.memory.jsHeapSizeLimit;
      this.local.ramUsagePct = Math.round((used / total) * 100);
    }

    // Compute average inference time from recent timings
    if (this._inferenceTimings.length > 0) {
      this.local.avgInferenceMs = Math.round(
        this._inferenceTimings.reduce((a, b) => a + b, 0) / this._inferenceTimings.length
      );
    }
  }

  /**
   * Record an inference completion timing.
   */
  recordInference(durationMs) {
    this._inferenceTimings.push(durationMs);
    if (this._inferenceTimings.length > 50) {
      this._inferenceTimings = this._inferenceTimings.slice(-25);
    }
    this.local.inferenceCount++;
  }

  /**
   * Update queue depth (call when fedPending changes).
   */
  setQueueDepth(depth) {
    this.local.queueDepth = depth;
  }

  /**
   * Update tensor backlog (bytes waiting in send queue).
   */
  setTensorBacklog(bytes) {
    this.local.tensorBacklog = bytes;
  }

  /**
   * Receive telemetry from a remote peer (via presence or direct message).
   */
  updateRemote(peerId, telemetry) {
    this.remote.set(peerId, {
      ...telemetry,
      receivedAt: Date.now(),
    });
  }

  /**
   * Get telemetry data to include in presence announcements.
   */
  getPresenceData() {
    return {
      telemetry: {
        ramPct: this.local.ramUsagePct,
        battery: this.local.batteryPct,
        charging: this.local.batteryCharging,
        pressure: this.local.cpuPressure,
        queue: this.local.queueDepth,
        avgMs: this.local.avgInferenceMs,
        backlog: this.local.tensorBacklog,
      }
    };
  }

  /**
   * Clean up stale remote telemetry (older than 60 seconds).
   */
  cleanup() {
    const cutoff = Date.now() - 60000;
    for (const [pid, t] of this.remote) {
      if (t.receivedAt < cutoff) this.remote.delete(pid);
    }
  }
}


// ═══════════════════════════════════════════════════════════════════════════
//  MARKOV STATE MODEL
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Models each peer's state as a discrete-time Markov chain.
 * 
 * Transition probabilities are estimated from telemetry history.
 * States: IDLE → COMPUTING → TRANSMITTING → IDLE (normal cycle)
 *         Any → OVERLOADED (when queue/RAM thresholds exceeded)
 *         Any → DISCONNECTED (churn)
 * 
 * The transition matrix P is updated every observation window
 * using maximum likelihood estimation from state counts.
 */
export class MarkovStateModel {
  constructor() {
    this.states = new Map(); // peerId -> current state
    this.history = new Map(); // peerId -> [{ state, timestamp }]
    
    // Transition count matrix: transitionCounts[from][to]
    // Used for MLE estimation of transition probabilities
    this.transitionCounts = Array.from({ length: 5 }, () => new Float64Array(5));
    
    // Estimated transition probabilities
    this.P = Array.from({ length: 5 }, () => new Float64Array(5));
    this._initDefaultTransitions();
  }

  /**
   * Initialize with reasonable default transition probabilities.
   * These will be updated as we observe actual transitions.
   */
  _initDefaultTransitions() {
    // Row = from state, Col = to state
    // IDLE:         70% stay idle, 25% start computing, 5% disconnect
    this.P[0] = Float64Array.from([0.70, 0.25, 0.00, 0.00, 0.05]);
    // COMPUTING:    10% back to idle, 20% stay, 60% transmit, 5% overload, 5% disconnect
    this.P[1] = Float64Array.from([0.10, 0.20, 0.60, 0.05, 0.05]);
    // TRANSMITTING: 50% back to idle, 10% compute more, 30% stay, 5% overload, 5% disconnect
    this.P[2] = Float64Array.from([0.50, 0.10, 0.30, 0.05, 0.05]);
    // OVERLOADED:   20% recover to idle, 0% compute, 0% transmit, 70% stay overloaded, 10% disconnect
    this.P[3] = Float64Array.from([0.20, 0.00, 0.00, 0.70, 0.10]);
    // DISCONNECTED: 30% reconnect to idle, 0%, 0%, 0%, 70% stay disconnected
    this.P[4] = Float64Array.from([0.30, 0.00, 0.00, 0.00, 0.70]);
  }

  /**
   * Observe a peer's current state (call periodically).
   * Updates transition counts for MLE.
   */
  observe(peerId, telemetry, isConnected) {
    const newState = this._classifyState(telemetry, isConnected);
    const prevState = this.states.get(peerId);
    
    if (prevState !== undefined && prevState !== newState) {
      this.transitionCounts[prevState][newState]++;
      this._updateProbabilities();
    }

    this.states.set(peerId, newState);
    
    // Track history for time-series analysis
    if (!this.history.has(peerId)) this.history.set(peerId, []);
    const h = this.history.get(peerId);
    h.push({ state: newState, timestamp: Date.now() });
    if (h.length > 100) this.history.set(peerId, h.slice(-50));

    return newState;
  }

  /**
   * Classify peer state from telemetry data.
   */
  _classifyState(telemetry, isConnected) {
    if (!isConnected) return PeerState.DISCONNECTED;
    if (!telemetry) return PeerState.IDLE;
    
    const { ramPct, battery, pressure, queue, backlog } = telemetry;
    
    // Overloaded: high RAM, low battery, critical pressure, or deep queue
    if (ramPct > 85 || (battery < 10 && !telemetry.charging) || 
        pressure === 'critical' || queue > 5) {
      return PeerState.OVERLOADED;
    }
    
    // Transmitting: significant tensor backlog
    if (backlog > 50000) return PeerState.TRANSMITTING;
    
    // Computing: has pending inferences
    if (queue > 0) return PeerState.COMPUTING;
    
    return PeerState.IDLE;
  }

  /**
   * Update transition probability matrix from observed counts (MLE).
   */
  _updateProbabilities() {
    for (let i = 0; i < 5; i++) {
      const rowSum = this.transitionCounts[i].reduce((a, b) => a + b, 0);
      if (rowSum > 10) { // Only update if we have enough observations
        for (let j = 0; j < 5; j++) {
          this.P[i][j] = this.transitionCounts[i][j] / rowSum;
        }
      }
    }
  }

  /**
   * Estimate probability that a peer disconnects within the next N steps.
   */
  disconnectProbability(peerId, steps = 3) {
    const currentState = this.states.get(peerId);
    if (currentState === undefined) return 0.5; // Unknown peer
    if (currentState === PeerState.DISCONNECTED) return 1.0;

    // Compute P^steps and read the DISCONNECTED column
    let prob = new Float64Array(5);
    prob[currentState] = 1.0;

    for (let step = 0; step < steps; step++) {
      const next = new Float64Array(5);
      for (let i = 0; i < 5; i++) {
        if (prob[i] === 0) continue;
        for (let j = 0; j < 5; j++) {
          next[j] += prob[i] * this.P[i][j];
        }
      }
      prob = next;
    }

    return prob[PeerState.DISCONNECTED];
  }

  /**
   * Estimate steady-state probability of being in each state.
   */
  steadyState() {
    // Power iteration on P^T to find left eigenvector
    let pi = new Float64Array(5).fill(0.2); // uniform start
    for (let iter = 0; iter < 100; iter++) {
      const next = new Float64Array(5);
      for (let j = 0; j < 5; j++) {
        for (let i = 0; i < 5; i++) {
          next[j] += pi[i] * this.P[i][j];
        }
      }
      // Normalize
      const sum = next.reduce((a, b) => a + b, 0);
      pi = next.map(x => x / sum);
    }
    return {
      idle: pi[0], computing: pi[1], transmitting: pi[2],
      overloaded: pi[3], disconnected: pi[4],
    };
  }
}


// ═══════════════════════════════════════════════════════════════════════════
//  LYAPUNOV DRIFT-PLUS-PENALTY CONTROLLER
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Implements Lyapunov optimization for stable inference routing.
 * 
 * Lyapunov function: L(Q) = Σᵢ Qᵢ²
 *   where Qᵢ = queue depth at peer i
 * 
 * At each decision epoch, for pending request r, select peer:
 *   p* = argmin_p { Qₚ / μₚ + V × latency(self, p) }
 *   where μₚ = service rate of peer p (tokens/sec)
 *         V = throughput-stability tradeoff parameter
 * 
 * Higher V → more weight on latency (better throughput)
 * Lower V → more weight on queue stability (better reliability)
 */
export class LyapunovController {
  /**
   * @param {Object} opts
   * @param {number} opts.V - Tradeoff parameter (default 0.5)
   * @param {number} opts.maxQueueDepth - Queue depth alarm threshold
   * @param {MarkovStateModel} opts.markov - Markov state model
   * @param {TelemetryCollector} opts.telemetry - Telemetry collector
   */
  constructor(opts = {}) {
    this.V = opts.V || 0.5;
    this.maxQueueDepth = opts.maxQueueDepth || 8;
    this.markov = opts.markov || new MarkovStateModel();
    this.telemetry = opts.telemetry || new TelemetryCollector();
    
    // Queue state per peer
    this.queues = new Map(); // peerId -> queue depth
    this.serviceRates = new Map(); // peerId -> tokens/sec
    
    // Decision log for analysis
    this.decisionLog = [];
  }

  /**
   * Update the observed queue depth for a peer.
   */
  updateQueue(peerId, depth) {
    this.queues.set(peerId, depth);
  }

  /**
   * Update the observed service rate for a peer.
   */
  updateServiceRate(peerId, tokensPerSec) {
    this.serviceRates.set(peerId, tokensPerSec);
  }

  /**
   * Compute the current Lyapunov function value.
   * L(Q) = Σᵢ Qᵢ²
   */
  lyapunovValue() {
    let L = 0;
    for (const [, q] of this.queues) {
      L += q * q;
    }
    return L;
  }

  /**
   * Select the optimal peer for an inference request.
   * 
   * This is the core Lyapunov drift-plus-penalty decision:
   *   p* = argmin_p { Q_p / μ_p + V × latency(self, p) }
   * 
   * Additionally weighted by:
   *   - Disconnect probability (from Markov model)
   *   - Device tier (prefer higher tiers for complex tasks)
   *   - Battery level (avoid dying devices)
   * 
   * @param {Map} peers - Current peer Map
   * @param {string} localPeerId - Self peer ID
   * @param {Object} intent - Intent classification {category, targetRole}
   * @param {boolean} hasLocalEngine - Whether local engine is available
   * @returns {Object|null} { peerId, score, reason }
   */
  selectPeer(peers, localPeerId, intent, hasLocalEngine) {
    const candidates = [];
    
    // Local option
    if (hasLocalEngine) {
      const localQ = this.queues.get(localPeerId) || 0;
      const localMu = this.serviceRates.get(localPeerId) || 5; // default 5 tok/s
      const localLatency = 0; // zero network latency for local
      const localDisconnect = 0; // can't disconnect from self
      
      const score = this._computeScore(localQ, localMu, localLatency, localDisconnect, 
        this.telemetry.local.batteryPct, this.telemetry.local.ramUsagePct);
      
      candidates.push({
        peerId: null, // null = local
        type: 'local',
        score,
        queue: localQ,
        serviceRate: localMu,
        reason: `L-score=${score.toFixed(2)}, Q=${localQ}, μ=${localMu}`,
      });
    }

    // Remote peers
    for (const [pid, p] of peers) {
      if (!p.hasEngine || p.dc?.readyState !== 'open') continue;
      
      const q = this.queues.get(pid) || 0;
      const mu = this.serviceRates.get(pid) || p.tps || p.benchmark || 3;
      const latency = p.rtt < Infinity ? p.rtt : 500;
      const pDisconnect = this.markov.disconnectProbability(pid, 3);
      
      // Get remote telemetry
      const remoteTelemetry = this.telemetry.remote.get(pid) || {};
      const battery = remoteTelemetry.battery || 100;
      const ram = remoteTelemetry.ramPct || 50;
      
      const score = this._computeScore(q, mu, latency, pDisconnect, battery, ram);
      
      // Skip overloaded peers
      if (q > this.maxQueueDepth) continue;
      
      candidates.push({
        peerId: pid,
        type: 'remote',
        score,
        queue: q,
        serviceRate: mu,
        tier: p.deviceTier,
        role: p.moeRole,
        reason: `L-score=${score.toFixed(2)}, Q=${q}, μ=${mu}, RTT=${latency}ms, P(disc)=${pDisconnect.toFixed(2)}`,
      });
    }

    if (candidates.length === 0) return null;

    // Select minimum score (lower is better)
    candidates.sort((a, b) => a.score - b.score);
    const winner = candidates[0];
    
    // Log decision
    this.decisionLog.push({
      timestamp: Date.now(),
      winner: winner.peerId || 'local',
      candidates: candidates.length,
      lyapunov: this.lyapunovValue(),
      intent: intent?.category,
    });
    if (this.decisionLog.length > 200) {
      this.decisionLog = this.decisionLog.slice(-100);
    }

    return winner;
  }

  /**
   * Compute the Lyapunov drift-plus-penalty score for a candidate peer.
   * 
   * score = (Q / μ) + V × latency/1000 + penalty_disconnect + penalty_battery + penalty_ram
   * 
   * Lower score = better candidate.
   */
  _computeScore(queue, serviceRate, latencyMs, disconnectProb, batteryPct, ramPct) {
    const mu = Math.max(serviceRate, 0.1); // prevent division by zero
    
    // Queue pressure term: how long until this peer's queue clears
    const queueTerm = queue / mu;
    
    // Latency term: network cost
    const latencyTerm = this.V * (latencyMs / 1000);
    
    // Disconnect penalty: expected cost of losing this peer mid-inference
    const disconnectPenalty = disconnectProb * 10;
    
    // Battery penalty: avoid peers about to die
    const batteryPenalty = batteryPct < 15 ? 5 : batteryPct < 30 ? 2 : 0;
    
    // RAM pressure penalty
    const ramPenalty = ramPct > 85 ? 3 : ramPct > 70 ? 1 : 0;
    
    return queueTerm + latencyTerm + disconnectPenalty + batteryPenalty + ramPenalty;
  }

  /**
   * Compute asymmetric load weights for all peers.
   * Higher weight = peer should receive more load.
   * 
   * weight(p) = serviceRate(p) × (1 - P_disconnect(p)) × reliability(p)
   * load(p) = weight(p) / Σ weight(all)
   * 
   * @param {Map} peers - Current peer Map
   * @returns {Map<string, number>} peerId -> load fraction [0, 1]
   */
  computeLoadDistribution(peers) {
    const weights = new Map();
    let totalWeight = 0;

    for (const [pid, p] of peers) {
      if (!p.hasEngine || p.dc?.readyState !== 'open') continue;
      
      const mu = this.serviceRates.get(pid) || p.tps || p.benchmark || 1;
      const pDisc = this.markov.disconnectProbability(pid, 5);
      const repScore = p.repScore || 0;
      const reliability = 1 + Math.log1p(repScore) / 5; // log-scaled reputation bonus
      
      const weight = mu * (1 - pDisc) * reliability;
      weights.set(pid, weight);
      totalWeight += weight;
    }

    // Normalize to fractions
    const distribution = new Map();
    if (totalWeight > 0) {
      for (const [pid, w] of weights) {
        distribution.set(pid, w / totalWeight);
      }
    }

    return distribution;
  }

  /**
   * Get a diagnostic summary of the controller state.
   */
  diagnostics() {
    const queueValues = [...this.queues.values()];
    const totalQueue = queueValues.reduce((a, b) => a + b, 0);
    const maxQueue = Math.max(0, ...queueValues);
    
    return {
      lyapunovValue: this.lyapunovValue(),
      totalQueueDepth: totalQueue,
      maxQueueDepth: maxQueue,
      peersTracked: this.queues.size,
      V: this.V,
      recentDecisions: this.decisionLog.slice(-5).map(d => ({
        target: d.winner,
        L: d.lyapunov,
        candidates: d.candidates,
      })),
      steadyState: this.markov.steadyState(),
    };
  }

  /**
   * Auto-tune the V parameter based on observed stability.
   * If queues are consistently growing, decrease V (prioritize stability).
   * If queues are consistently empty, increase V (prioritize throughput).
   */
  autoTuneV() {
    const recent = this.decisionLog.slice(-20);
    if (recent.length < 10) return;
    
    const avgL = recent.reduce((s, d) => s + d.lyapunov, 0) / recent.length;
    const trend = recent.slice(-5).reduce((s, d) => s + d.lyapunov, 0) / 5 - 
                  recent.slice(0, 5).reduce((s, d) => s + d.lyapunov, 0) / 5;
    
    if (trend > 0 && avgL > 10) {
      // Queues growing — prioritize stability
      this.V = Math.max(0.1, this.V * 0.9);
    } else if (avgL < 2) {
      // Queues consistently empty — can prioritize throughput
      this.V = Math.min(2.0, this.V * 1.1);
    }
  }
}


// ═══════════════════════════════════════════════════════════════════════════
//  INTEGRATED CONTROLLER — Combines all Phase 4 components
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Full Lyapunov-based routing controller that integrates:
 * - TelemetryCollector
 * - MarkovStateModel
 * - LyapunovController
 * 
 * Designed to be instantiated once and hooked into the existing
 * inference routing pipeline.
 */
export class SwarmOptimizer {
  constructor(opts = {}) {
    this.telemetry = new TelemetryCollector();
    this.markov = new MarkovStateModel();
    this.controller = new LyapunovController({
      V: opts.V || 0.5,
      maxQueueDepth: opts.maxQueueDepth || 8,
      markov: this.markov,
      telemetry: this.telemetry,
    });
    this.observeInterval = null;
  }

  /**
   * Start the optimizer. Call once during boot.
   */
  async start(peers, peerId) {
    this.peers = peers;
    this.peerId = peerId;
    
    await this.telemetry.start();
    
    // Periodic state observation
    this.observeInterval = setInterval(() => {
      this._observeAll();
      this.controller.autoTuneV();
      this.telemetry.cleanup();
    }, 5000);

    console.log('[Lyapunov] SwarmOptimizer started');
  }

  stop() {
    if (this.observeInterval) {
      clearInterval(this.observeInterval);
      this.observeInterval = null;
    }
  }

  /**
   * Observe all peers and update Markov model.
   */
  _observeAll() {
    if (!this.peers) return;
    
    for (const [pid, p] of this.peers) {
      const isConnected = p.dc?.readyState === 'open';
      const telemetry = this.telemetry.remote.get(pid) || {};
      
      this.markov.observe(pid, telemetry, isConnected);
      
      // Update service rates from peer data
      if (p.tps > 0) this.controller.updateServiceRate(pid, p.tps);
      if (p.benchmark > 0 && !this.controller.serviceRates.has(pid)) {
        this.controller.updateServiceRate(pid, p.benchmark);
      }
    }
  }

  /**
   * Route an inference request using Lyapunov optimization.
   * Drop-in replacement for the existing `routeRequest()`.
   */
  route(intent, hasEngine) {
    return this.controller.selectPeer(this.peers, this.peerId, intent, hasEngine);
  }

  /**
   * Get load distribution for layer assignment in distributed inference.
   */
  getLoadDistribution() {
    return this.controller.computeLoadDistribution(this.peers);
  }

  /**
   * Record a completed inference for telemetry.
   */
  recordInference(durationMs) {
    this.telemetry.recordInference(durationMs);
  }

  /**
   * Update remote peer telemetry (call from presence handler).
   */
  updatePeerTelemetry(peerId, telemetryData) {
    this.telemetry.updateRemote(peerId, telemetryData);
  }

  /**
   * Get telemetry data to include in presence.
   */
  getPresenceData() {
    return this.telemetry.getPresenceData();
  }

  /**
   * Get full diagnostics for debug UI.
   */
  diagnostics() {
    return this.controller.diagnostics();
  }
}
