/**
 * weight_loader.js — Distributed Weight Management for P2P LLM Inference
 * 
 * Handles:
 *   1. HuggingFace safetensor shard downloading
 *   2. Weight distribution across peers via Reed-Solomon encoding
 *   3. On-demand weight assembly from distributed shards
 *   4. IndexedDB persistence for cross-session caching
 * 
 * Weight format: safetensors (https://huggingface.co/docs/safetensors)
 * Shards are split per tensor, RS-encoded, and distributed.
 * 
 * Integration:
 *   - Uses ShardDistributor from reed_solomon.js
 *   - Sends shards via existing WebRTC DataChannels
 *   - Stores locally in IndexedDB for persistence
 */

// ═══════════════════════════════════════════════════════════════════════════
//  INDEXEDDB CACHE
// ═══════════════════════════════════════════════════════════════════════════

const DB_NAME = 'p2p-llm-weights';
const DB_VERSION = 1;
const STORE_NAME = 'shards';

class WeightCache {
  constructor() {
    this.db = null;
  }

  async open() {
    return new Promise((resolve, reject) => {
      const req = indexedDB.open(DB_NAME, DB_VERSION);
      req.onupgradeneeded = (e) => {
        const db = e.target.result;
        if (!db.objectStoreNames.contains(STORE_NAME)) {
          db.createObjectStore(STORE_NAME);
        }
      };
      req.onsuccess = (e) => {
        this.db = e.target.result;
        resolve(this);
      };
      req.onerror = () => reject(new Error('IndexedDB open failed'));
    });
  }

  async get(key) {
    if (!this.db) return null;
    return new Promise((resolve) => {
      const tx = this.db.transaction(STORE_NAME, 'readonly');
      const req = tx.objectStore(STORE_NAME).get(key);
      req.onsuccess = () => resolve(req.result || null);
      req.onerror = () => resolve(null);
    });
  }

  async set(key, value) {
    if (!this.db) return;
    return new Promise((resolve, reject) => {
      const tx = this.db.transaction(STORE_NAME, 'readwrite');
      tx.objectStore(STORE_NAME).put(value, key);
      tx.oncomplete = () => resolve();
      tx.onerror = () => reject(new Error('IndexedDB write failed'));
    });
  }

  async delete(key) {
    if (!this.db) return;
    return new Promise((resolve) => {
      const tx = this.db.transaction(STORE_NAME, 'readwrite');
      tx.objectStore(STORE_NAME).delete(key);
      tx.oncomplete = () => resolve();
      tx.onerror = () => resolve();
    });
  }

  async keys() {
    if (!this.db) return [];
    return new Promise((resolve) => {
      const tx = this.db.transaction(STORE_NAME, 'readonly');
      const req = tx.objectStore(STORE_NAME).getAllKeys();
      req.onsuccess = () => resolve(req.result || []);
      req.onerror = () => resolve([]);
    });
  }

  async totalSize() {
    const allKeys = await this.keys();
    let total = 0;
    for (const key of allKeys) {
      const val = await this.get(key);
      if (val instanceof ArrayBuffer) total += val.byteLength;
      else if (val?.byteLength) total += val.byteLength;
    }
    return total;
  }
}


// ═══════════════════════════════════════════════════════════════════════════
//  SAFETENSOR PARSER (Minimal)
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Parse a safetensors file header to get tensor metadata.
 * Format: 8-byte little-endian header size, then JSON header, then raw data.
 */
function parseSafetensorHeader(buffer) {
  const view = new DataView(buffer);
  const headerSize = Number(view.getBigUint64(0, true));
  const headerBytes = new Uint8Array(buffer, 8, headerSize);
  const headerJSON = new TextDecoder().decode(headerBytes);
  const header = JSON.parse(headerJSON);
  
  const tensors = {};
  const dataOffset = 8 + headerSize;
  
  for (const [name, info] of Object.entries(header)) {
    if (name === '__metadata__') continue;
    tensors[name] = {
      dtype: info.dtype,
      shape: info.shape,
      dataOffsets: info.data_offsets, // [start, end] relative to data section
      absoluteOffset: dataOffset + info.data_offsets[0],
      byteLength: info.data_offsets[1] - info.data_offsets[0],
    };
  }

  return { tensors, dataOffset, headerSize };
}

/**
 * Extract a single tensor from a safetensors buffer.
 */
function extractTensor(buffer, tensorInfo) {
  return buffer.slice(tensorInfo.absoluteOffset, 
    tensorInfo.absoluteOffset + tensorInfo.byteLength);
}


// ═══════════════════════════════════════════════════════════════════════════
//  HUGGINGFACE SHARD FETCHER
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Downloads model weight shards from HuggingFace.
 * Supports both single-file and sharded models.
 */
export class HuggingFaceShardFetcher {
  /**
   * @param {string} repoId - HuggingFace repo (e.g., "meta-llama/Llama-3-8B")
   * @param {string} revision - Branch/commit (default "main")
   */
  constructor(repoId, revision = 'main') {
    this.repoId = repoId;
    this.revision = revision;
    this.baseUrl = `https://huggingface.co/${repoId}/resolve/${revision}`;
    this.cache = new WeightCache();
    this.onProgress = null;
  }

  async init() {
    await this.cache.open();
  }

  /**
   * Fetch the model index to determine shard files.
   */
  async fetchIndex() {
    try {
      const res = await fetch(`${this.baseUrl}/model.safetensors.index.json`);
      if (res.ok) {
        const index = await res.json();
        return {
          type: 'sharded',
          weightMap: index.weight_map,
          files: [...new Set(Object.values(index.weight_map))],
        };
      }
    } catch(e) {}

    // Single-file model
    return {
      type: 'single',
      files: ['model.safetensors'],
      weightMap: null,
    };
  }

  /**
   * Download a shard file with progress tracking.
   * Returns the raw ArrayBuffer.
   */
  async fetchShard(filename, onProgress) {
    // Check cache first
    const cacheKey = `${this.repoId}/${filename}`;
    const cached = await this.cache.get(cacheKey);
    if (cached) {
      if (onProgress) onProgress(1.0, 'cached');
      return cached;
    }

    const url = `${this.baseUrl}/${filename}`;
    const res = await fetch(url);
    if (!res.ok) throw new Error(`Failed to fetch ${url}: ${res.status}`);

    const contentLength = parseInt(res.headers.get('Content-Length') || '0');
    const reader = res.body.getReader();
    const chunks = [];
    let received = 0;

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      chunks.push(value);
      received += value.length;
      if (onProgress && contentLength > 0) {
        onProgress(received / contentLength, `${(received / 1024 / 1024).toFixed(1)}MB`);
      }
    }

    // Concatenate chunks
    const buffer = new ArrayBuffer(received);
    const view = new Uint8Array(buffer);
    let offset = 0;
    for (const chunk of chunks) {
      view.set(chunk, offset);
      offset += chunk.length;
    }

    // Cache in IndexedDB
    try {
      await this.cache.set(cacheKey, buffer);
    } catch(e) {
      console.warn('[WeightLoader] Failed to cache shard:', e.message);
    }

    return buffer;
  }

  /**
   * Extract tensors needed for a specific layer from shard files.
   * 
   * @param {number} layerIdx - Layer index
   * @param {Object} index - Model index from fetchIndex()
   * @returns {Object} Map of tensor name to Float32Array
   */
  async fetchLayerWeights(layerIdx, index) {
    const prefix = `model.layers.${layerIdx}.`;
    const tensorNames = [
      `${prefix}self_attn.q_proj.weight`,
      `${prefix}self_attn.k_proj.weight`,
      `${prefix}self_attn.v_proj.weight`,
      `${prefix}self_attn.o_proj.weight`,
      `${prefix}mlp.gate_proj.weight`,
      `${prefix}mlp.up_proj.weight`,
      `${prefix}mlp.down_proj.weight`,
      `${prefix}input_layernorm.weight`,
      `${prefix}post_attention_layernorm.weight`,
    ];

    // Determine which shard files contain these tensors
    const filesToFetch = new Map(); // filename -> Set<tensorName>
    for (const tname of tensorNames) {
      const filename = index.weightMap?.[tname] || index.files[0];
      if (!filesToFetch.has(filename)) filesToFetch.set(filename, new Set());
      filesToFetch.get(filename).add(tname);
    }

    const weights = {};
    for (const [filename, tensors] of filesToFetch) {
      if (this.onProgress) this.onProgress({ file: filename, layer: layerIdx });
      
      const buffer = await this.fetchShard(filename, (pct, info) => {
        if (this.onProgress) this.onProgress({ file: filename, layer: layerIdx, pct, info });
      });

      const { tensors: tensorMeta } = parseSafetensorHeader(buffer);
      
      for (const tname of tensors) {
        const meta = tensorMeta[tname];
        if (!meta) {
          console.warn(`[WeightLoader] Tensor ${tname} not found in ${filename}`);
          continue;
        }
        
        const raw = extractTensor(buffer, meta);
        
        // Convert to Float32 if needed
        let float32;
        if (meta.dtype === 'F32') {
          float32 = new Float32Array(raw);
        } else if (meta.dtype === 'F16' || meta.dtype === 'BF16') {
          // Convert FP16/BF16 to FP32
          float32 = convertToFloat32(raw, meta.dtype);
        } else {
          console.warn(`[WeightLoader] Unsupported dtype ${meta.dtype} for ${tname}`);
          continue;
        }

        // Map to short name
        const shortName = tname.replace(prefix, '').replace('.weight', '')
          .replace('self_attn.', '').replace('mlp.', '');
        weights[shortName] = float32;
      }
    }

    return weights;
  }
}

/**
 * Convert FP16 or BF16 buffer to Float32Array.
 */
function convertToFloat32(buffer, dtype) {
  const input = new Uint16Array(buffer);
  const output = new Float32Array(input.length);
  const tmpBuf = new ArrayBuffer(4);
  const tmpView = new DataView(tmpBuf);

  for (let i = 0; i < input.length; i++) {
    if (dtype === 'BF16') {
      // BF16: just shift left by 16 bits to get FP32
      tmpView.setUint32(0, input[i] << 16, false);
      output[i] = tmpView.getFloat32(0, false);
    } else {
      // FP16 → FP32 conversion
      const h = input[i];
      const sign = (h >> 15) & 1;
      const exp = (h >> 10) & 0x1F;
      const frac = h & 0x3FF;

      if (exp === 0) {
        // Subnormal or zero
        output[i] = (sign ? -1 : 1) * Math.pow(2, -14) * (frac / 1024);
      } else if (exp === 31) {
        // Inf or NaN
        output[i] = frac === 0 ? (sign ? -Infinity : Infinity) : NaN;
      } else {
        output[i] = (sign ? -1 : 1) * Math.pow(2, exp - 15) * (1 + frac / 1024);
      }
    }
  }
  return output;
}


// ═══════════════════════════════════════════════════════════════════════════
//  PEER WEIGHT DISTRIBUTOR
// ═══════════════════════════════════════════════════════════════════════════

/**
 * High-level manager for distributed weight storage.
 * Combines HuggingFaceShardFetcher with ShardDistributor.
 */
export class PeerWeightDistributor {
  constructor(repoId, rsN = 6, rsK = 4) {
    this.fetcher = new HuggingFaceShardFetcher(repoId);
    this.rsN = rsN;
    this.rsK = rsK;
    this.layerAssignments = new Map(); // layerIdx -> { peerId -> blockIndices[] }
    this.localBlocks = new Map(); // `layer_${i}_${tensor}_${blockIdx}` -> ArrayBuffer
  }

  async init() {
    await this.fetcher.init();
  }

  /**
   * Load and distribute a single layer's weights.
   * @returns {Object} { layerIdx, distribution, localWeights }
   */
  async loadAndDistributeLayer(layerIdx, modelIndex, peers, peerId, gpuEngine) {
    const weights = await this.fetcher.fetchLayerWeights(layerIdx, modelIndex);
    
    // For each weight tensor, RS-encode and distribute
    const { ShardDistributor } = await import('./reed_solomon.js');
    const distributor = new ShardDistributor(this.rsN, this.rsK);
    
    const localWeights = {};
    
    for (const [name, float32] of Object.entries(weights)) {
      const shardId = `layer_${layerIdx}_${name}`;
      const plan = distributor.distribute(shardId, float32.buffer, peers, peerId);
      
      // Store our assigned blocks locally
      for (const block of plan.plan) {
        if (block.peerId === peerId) {
          const blockKey = `${shardId}_block_${block.blockIndex}`;
          this.localBlocks.set(blockKey, block.data);
        }
        // TODO: Send remote blocks via WebRTC DataChannel
      }

      // Keep full weight locally for GPU engine
      localWeights[name] = float32;
    }

    // Load into GPU if engine is available
    if (gpuEngine) {
      gpuEngine.loadLayerWeights(layerIdx, localWeights);
    }

    return { layerIdx, localWeights, tensorCount: Object.keys(weights).length };
  }

  /**
   * Get cache stats.
   */
  async cacheStats() {
    const totalSize = await this.fetcher.cache.totalSize();
    return {
      localBlocks: this.localBlocks.size,
      cachedMB: Math.round(totalSize / 1024 / 1024),
      layers: this.layerAssignments.size,
    };
  }
}
