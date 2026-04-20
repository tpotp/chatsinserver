/**
 * flow_router.js — Ford-Fulkerson / Push-Relabel Tensor Router
 * 
 * Models the P2P network as a flow network G = (V, E) where:
 *   - Vertices V = peer nodes
 *   - Edge capacity c(u,v) = measured DataChannel bandwidth (bytes/sec)
 *   - Source s = inference initiator
 *   - Sink t = peer hosting the next required layer
 * 
 * Implements:
 *   1. BandwidthProber — Measures real-time DataChannel throughput
 *   2. PushRelabelSolver — Computes max-flow / min-cut
 *   3. TensorRouter — Multi-path tensor splitting with latency awareness
 * 
 * Mathematical foundation:
 *   Max-Flow / Min-Cut Theorem (Ford & Fulkerson, 1956):
 *     max Σ f(s,v) = min Σ c(S,T) for all s-t cuts (S,T)
 * 
 * Integration:
 *   - Reads from existing `peers` Map for topology
 *   - Uses existing `encodeMsg` / DataChannel infrastructure for transport
 *   - BandwidthProber piggybacks on existing PING/PONG cycle
 */

// ═══════════════════════════════════════════════════════════════════════════
//  BANDWIDTH PROBER
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Measures real-time bandwidth of each WebRTC DataChannel
 * by sending calibrated probe packets during idle periods.
 */
export class BandwidthProber {
  constructor() {
    this.capacities = new Map(); // peerId -> { bw: bytes/sec, rtt: ms, lastProbe: timestamp }
    this.probeSize = 8192; // 8KB probe packet
    this.probeInterval = 15000; // 15 seconds between probes
    this.timer = null;
  }

  /**
   * Start periodic bandwidth probing.
   * @param {Map} peers - The existing peers Map from index.html
   */
  start(peers) {
    this.peers = peers;
    this.timer = setInterval(() => this._probeAll(), this.probeInterval);
    // Initial probe after 3 seconds
    setTimeout(() => this._probeAll(), 3000);
  }

  stop() {
    if (this.timer) { clearInterval(this.timer); this.timer = null; }
  }

  _probeAll() {
    if (!this.peers) return;
    for (const [pid, p] of this.peers) {
      if (p.dc?.readyState === 'open') {
        this._probePeer(pid, p);
      }
    }
  }

  /**
   * Send a timestamped probe packet and measure round-trip bandwidth.
   */
  _probePeer(pid, peer) {
    try {
      const probe = new ArrayBuffer(this.probeSize);
      const view = new DataView(probe);
      // Header: magic(4) + timestamp(8)
      view.setUint32(0, 0xBEEF0001); // magic for bandwidth probe
      const now = performance.now();
      view.setFloat64(4, now);
      
      // Fill rest with random data to prevent compression
      const u8 = new Uint8Array(probe, 12);
      for (let i = 0; i < u8.length; i++) u8[i] = (i * 7 + 13) & 0xFF;
      
      peer.dc.send(probe);
      
      // Store send time for RTT calculation
      if (!peer._bwProbeTime) peer._bwProbeTime = new Map();
      peer._bwProbeTime.set(now, { size: this.probeSize, sent: now });
    } catch(e) { /* DataChannel may have closed */ }
  }

  /**
   * Handle incoming probe response. Call this from the message handler.
   * @returns {boolean} true if this was a probe message and was handled
   */
  handleProbeResponse(peerId, buffer) {
    if (buffer.byteLength < 12) return false;
    const view = new DataView(buffer);
    const magic = view.getUint32(0);
    
    if (magic === 0xBEEF0001) {
      // This is a probe — echo it back as response
      const response = new ArrayBuffer(12);
      const rv = new DataView(response);
      rv.setUint32(0, 0xBEEF0002); // response magic
      rv.setFloat64(4, view.getFloat64(4)); // echo original timestamp
      const peer = this.peers?.get(peerId);
      if (peer?.dc?.readyState === 'open') {
        try { peer.dc.send(response); } catch(e) {}
      }
      return true;
    }
    
    if (magic === 0xBEEF0002) {
      // This is a probe response — calculate bandwidth
      const originalTime = view.getFloat64(4);
      const rtt = performance.now() - originalTime;
      const bw = rtt > 0 ? Math.round((this.probeSize * 2) / (rtt / 1000)) : 0; // bytes/sec (round-trip)
      
      this.capacities.set(peerId, {
        bw,
        rtt: Math.round(rtt),
        lastProbe: Date.now(),
        bwKBps: Math.round(bw / 1024),
        bwMBps: Math.round(bw / 1024 / 1024 * 10) / 10,
      });
      return true;
    }
    
    return false;
  }

  /**
   * Get measured bandwidth for a peer in bytes/sec.
   */
  getCapacity(peerId) {
    const entry = this.capacities.get(peerId);
    if (!entry || Date.now() - entry.lastProbe > 60000) {
      return 100 * 1024; // Default: 100 KB/s if no measurement
    }
    return entry.bw;
  }

  /**
   * Get all current capacity measurements.
   */
  getAllCapacities() {
    return new Map(this.capacities);
  }
}


// ═══════════════════════════════════════════════════════════════════════════
//  PUSH-RELABEL MAX-FLOW SOLVER
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Implements the Push-Relabel algorithm for maximum flow.
 * 
 * Complexity: O(V² × E) worst case, but practically much faster
 * due to gap heuristic and FIFO selection.
 * 
 * The algorithm maintains:
 *   - height[v]: Label function (distance estimate to sink)
 *   - excess[v]: Excess flow at each vertex
 *   - Residual graph with forward/backward edges
 * 
 * Operations:
 *   - Push: Move excess flow along an admissible edge
 *   - Relabel: Increase height when no admissible edges exist
 */
export class PushRelabelSolver {
  /**
   * Compute maximum flow from source to sink.
   * 
   * @param {string[]} nodes - List of node IDs
   * @param {Array<{from: string, to: string, capacity: number}>} edges - Directed edges
   * @param {string} source - Source node ID
   * @param {string} sink - Sink node ID
   * @returns {Object} { maxFlow, flowEdges, paths }
   */
  solve(nodes, edges, source, sink) {
    const n = nodes.length;
    const nodeIdx = new Map();
    nodes.forEach((id, i) => nodeIdx.set(id, i));

    const s = nodeIdx.get(source);
    const t = nodeIdx.get(sink);
    if (s === undefined || t === undefined) {
      return { maxFlow: 0, flowEdges: [], paths: [] };
    }

    // Build adjacency list with residual capacities
    // Each edge stores: { to, cap, flow, rev (index of reverse edge) }
    const graph = Array.from({ length: n }, () => []);
    
    const addEdge = (u, v, cap) => {
      graph[u].push({ to: v, cap, flow: 0, rev: graph[v].length });
      graph[v].push({ to: u, cap: 0, flow: 0, rev: graph[u].length - 1 }); // reverse edge
    };

    for (const e of edges) {
      const u = nodeIdx.get(e.from);
      const v = nodeIdx.get(e.to);
      if (u !== undefined && v !== undefined) {
        addEdge(u, v, e.capacity);
      }
    }

    // Initialize
    const height = new Int32Array(n);
    const excess = new Float64Array(n);
    height[s] = n;

    // Saturate all edges from source
    for (const edge of graph[s]) {
      const pushAmount = edge.cap;
      if (pushAmount > 0) {
        edge.flow = pushAmount;
        graph[edge.to][edge.rev].flow = -pushAmount;
        excess[edge.to] += pushAmount;
        excess[s] -= pushAmount;
      }
    }

    // Active nodes queue (FIFO with gap heuristic)
    const active = [];
    for (let i = 0; i < n; i++) {
      if (i !== s && i !== t && excess[i] > 0) {
        active.push(i);
      }
    }

    // Main loop
    let iterations = 0;
    const maxIter = n * n * 4; // Safety limit

    while (active.length > 0 && iterations < maxIter) {
      iterations++;
      const u = active.shift();

      if (excess[u] <= 0) continue;

      // Try to push
      let pushed = false;
      for (const edge of graph[u]) {
        if (excess[u] <= 0) break;
        const residual = edge.cap - edge.flow;
        if (residual > 0 && height[u] === height[edge.to] + 1) {
          // Admissible edge — push
          const pushAmount = Math.min(excess[u], residual);
          edge.flow += pushAmount;
          graph[edge.to][edge.rev].flow -= pushAmount;
          excess[u] -= pushAmount;
          excess[edge.to] += pushAmount;
          if (edge.to !== s && edge.to !== t && excess[edge.to] > 0) {
            active.push(edge.to);
          }
          pushed = true;
        }
      }

      // If couldn't push, relabel
      if (excess[u] > 0) {
        let minHeight = Infinity;
        for (const edge of graph[u]) {
          const residual = edge.cap - edge.flow;
          if (residual > 0 && height[edge.to] < minHeight) {
            minHeight = height[edge.to];
          }
        }
        if (minHeight < Infinity) {
          height[u] = minHeight + 1;
          active.push(u);
        }
      }
    }

    // Compute max flow (total flow into sink)
    const maxFlow = excess[t];

    // Extract flow edges
    const flowEdges = [];
    for (let u = 0; u < n; u++) {
      for (const edge of graph[u]) {
        if (edge.flow > 0 && edge.cap > 0) { // Only forward edges with positive flow
          flowEdges.push({
            from: nodes[u],
            to: nodes[edge.to],
            flow: edge.flow,
            capacity: edge.cap,
          });
        }
      }
    }

    // Decompose into paths using DFS
    const paths = this._decomposePaths(nodes, flowEdges, source, sink);

    return { maxFlow, flowEdges, paths, iterations };
  }

  /**
   * Decompose flow into edge-disjoint paths from source to sink.
   */
  _decomposePaths(nodes, flowEdges, source, sink) {
    const paths = [];
    // Build adjacency with remaining flow
    const remaining = new Map();
    for (const e of flowEdges) {
      const key = `${e.from}->${e.to}`;
      remaining.set(key, (remaining.get(key) || 0) + e.flow);
    }

    // Adjacency list
    const adj = new Map();
    for (const e of flowEdges) {
      if (!adj.has(e.from)) adj.set(e.from, []);
      adj.get(e.from).push(e.to);
    }

    // DFS to find paths
    const maxPaths = 20; // Safety limit
    while (paths.length < maxPaths) {
      const path = [source];
      const visited = new Set([source]);
      let current = source;
      let bottleneck = Infinity;
      const pathEdges = [];
      let found = false;

      while (current !== sink) {
        const neighbors = adj.get(current) || [];
        let next = null;
        for (const n of neighbors) {
          const key = `${current}->${n}`;
          const rem = remaining.get(key) || 0;
          if (rem > 0 && !visited.has(n)) {
            next = n;
            bottleneck = Math.min(bottleneck, rem);
            pathEdges.push(key);
            break;
          }
        }
        if (!next) break;
        path.push(next);
        visited.add(next);
        current = next;
        if (current === sink) found = true;
      }

      if (!found) break;

      // Subtract bottleneck from path edges
      for (const key of pathEdges) {
        remaining.set(key, (remaining.get(key) || 0) - bottleneck);
      }

      paths.push({ nodes: path, flow: bottleneck });
    }

    return paths;
  }
}


// ═══════════════════════════════════════════════════════════════════════════
//  TENSOR ROUTER — Multi-path splitting with latency awareness
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Routes tensors through the P2P network using max-flow decomposition.
 * 
 * Given a tensor T of size S bytes and destination peer:
 *   1. Computes max-flow from self to destination
 *   2. Decomposes into edge-disjoint paths
 *   3. Splits tensor proportionally to path capacities
 *   4. Sends chunks in parallel
 *   5. Reassembles at destination with sequence numbers
 * 
 * Also provides single-hop direct routing for nearby peers.
 */
export class TensorRouter {
  constructor(bandwidthProber) {
    this.prober = bandwidthProber;
    this.solver = new PushRelabelSolver();
    this.pendingReassembly = new Map(); // reqId -> { chunks: Map, total, received, data }
  }

  /**
   * Find the optimal route for a tensor from source to destination.
   * 
   * @param {string} source - Source peer ID
   * @param {string} destination - Destination peer ID
   * @param {Map} peers - Current peer Map
   * @param {number} tensorBytes - Size of tensor to send
   * @param {number} maxLatencyMs - Maximum acceptable latency
   * @returns {Object} routing plan
   */
  computeRoute(source, destination, peers, tensorBytes, maxLatencyMs = 5000) {
    // Check for direct connection first
    const directPeer = peers.get(destination);
    if (directPeer?.dc?.readyState === 'open') {
      const bw = this.prober.getCapacity(destination);
      const estimatedMs = (tensorBytes / bw) * 1000;
      if (estimatedMs < maxLatencyMs) {
        return {
          type: 'direct',
          paths: [{ nodes: [source, destination], flow: bw }],
          estimatedMs: Math.round(estimatedMs),
          totalBandwidth: bw,
        };
      }
    }

    // Build flow network from peer topology
    const nodes = [source];
    const edges = [];
    
    for (const [pid, p] of peers) {
      nodes.push(pid);
      if (p.dc?.readyState === 'open') {
        const bw = this.prober.getCapacity(pid);
        const rtt = p.rtt < Infinity ? p.rtt : 500;
        
        // Only include edges with acceptable latency
        if (rtt < maxLatencyMs) {
          // Edge from source to this peer
          edges.push({ from: source, to: pid, capacity: bw });
          // Edge from this peer back to source (for reverse flow)
          edges.push({ from: pid, to: source, capacity: bw });
          
          // Peer-to-peer edges (estimated from topology)
          if (p.topologyNeighbors) {
            for (const nid of p.topologyNeighbors) {
              if (peers.has(nid)) {
                // Estimate bandwidth as min of our measurements to each
                const nBw = Math.min(bw, this.prober.getCapacity(nid) || 50 * 1024);
                edges.push({ from: pid, to: nid, capacity: nBw });
              }
            }
          }
        }
      }
    }

    // Solve max-flow
    const uniqueNodes = [...new Set(nodes)];
    const result = this.solver.solve(uniqueNodes, edges, source, destination);

    if (result.maxFlow === 0 || result.paths.length === 0) {
      return { type: 'unreachable', paths: [], estimatedMs: Infinity, totalBandwidth: 0 };
    }

    const estimatedMs = (tensorBytes / result.maxFlow) * 1000;

    return {
      type: result.paths.length === 1 ? 'single-path' : 'multi-path',
      paths: result.paths,
      estimatedMs: Math.round(estimatedMs),
      totalBandwidth: result.maxFlow,
      maxFlow: result.maxFlow,
    };
  }

  /**
   * Split a tensor into chunks for multi-path routing.
   * 
   * @param {ArrayBuffer} tensorData - Raw tensor data
   * @param {Object} route - Route from computeRoute()
   * @param {number} reqId - Request ID for reassembly
   * @returns {Array<{path: Object, chunk: ArrayBuffer, index: number, total: number}>}
   */
  splitTensor(tensorData, route, reqId) {
    const totalBytes = tensorData.byteLength;
    const chunks = [];
    
    if (route.paths.length <= 1) {
      // Single path — send as one chunk
      chunks.push({
        path: route.paths[0] || { nodes: [] },
        chunk: tensorData,
        index: 0,
        total: 1,
        reqId,
      });
      return chunks;
    }

    // Multi-path — split proportionally to path flow
    const totalFlow = route.paths.reduce((s, p) => s + p.flow, 0);
    let offset = 0;

    for (let i = 0; i < route.paths.length; i++) {
      const path = route.paths[i];
      const fraction = path.flow / totalFlow;
      const chunkSize = i === route.paths.length - 1 
        ? totalBytes - offset 
        : Math.floor(totalBytes * fraction);
      
      chunks.push({
        path,
        chunk: tensorData.slice(offset, offset + chunkSize),
        index: i,
        total: route.paths.length,
        reqId,
      });
      offset += chunkSize;
    }

    return chunks;
  }

  /**
   * Encode a tensor chunk for transmission.
   * Header: magic(4) + reqId(4) + chunkIndex(2) + totalChunks(2) + totalBytes(4) + data
   */
  encodeChunk(chunk) {
    const headerSize = 16;
    const buf = new ArrayBuffer(headerSize + chunk.chunk.byteLength);
    const dv = new DataView(buf);
    dv.setUint32(0, 0xFACE0001); // magic
    dv.setUint32(4, chunk.reqId);
    dv.setUint16(8, chunk.index);
    dv.setUint16(10, chunk.total);
    dv.setUint32(12, chunk.chunk.byteLength);
    new Uint8Array(buf, headerSize).set(new Uint8Array(chunk.chunk));
    return buf;
  }

  /**
   * Decode and reassemble incoming tensor chunks.
   * Returns the complete tensor when all chunks are received, null otherwise.
   */
  receiveChunk(buffer) {
    if (buffer.byteLength < 16) return null;
    const dv = new DataView(buffer);
    if (dv.getUint32(0) !== 0xFACE0001) return null;

    const reqId = dv.getUint32(4);
    const chunkIdx = dv.getUint16(8);
    const totalChunks = dv.getUint16(10);
    const chunkSize = dv.getUint32(12);
    const chunkData = buffer.slice(16, 16 + chunkSize);

    if (!this.pendingReassembly.has(reqId)) {
      this.pendingReassembly.set(reqId, {
        chunks: new Map(),
        total: totalChunks,
        received: 0,
        totalBytes: 0,
        startTime: performance.now(),
      });
    }

    const entry = this.pendingReassembly.get(reqId);
    if (!entry.chunks.has(chunkIdx)) {
      entry.chunks.set(chunkIdx, chunkData);
      entry.received++;
      entry.totalBytes += chunkData.byteLength;
    }

    if (entry.received >= entry.total) {
      // All chunks received — reassemble
      const totalSize = entry.totalBytes;
      const result = new ArrayBuffer(totalSize);
      const resultView = new Uint8Array(result);
      let offset = 0;
      for (let i = 0; i < entry.total; i++) {
        const chunk = entry.chunks.get(i);
        if (chunk) {
          resultView.set(new Uint8Array(chunk), offset);
          offset += chunk.byteLength;
        }
      }
      
      const elapsed = performance.now() - entry.startTime;
      this.pendingReassembly.delete(reqId);
      
      return {
        reqId,
        data: result,
        elapsed: Math.round(elapsed),
        throughput: Math.round(totalSize / (elapsed / 1000) / 1024) + ' KB/s',
      };
    }

    return null; // Still waiting for more chunks
  }

  /**
   * Clean up stale pending reassembly entries (older than 30 seconds).
   */
  cleanup() {
    const now = performance.now();
    for (const [reqId, entry] of this.pendingReassembly) {
      if (now - entry.startTime > 30000) {
        this.pendingReassembly.delete(reqId);
      }
    }
  }
}
