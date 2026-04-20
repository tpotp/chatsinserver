// -- PETALS RUNTIME (Full-context recomputation, optimal for 135M) --

async function petalsRunLayer(component, inputData, seqLen) {
  const session = petalsONNXSessions[component];
  if (!session) return null;
  if (component === 'embeds') {
    const inputTensor = new ort.Tensor('int64', BigInt64Array.from(inputData.map(BigInt)), [1, seqLen]);
    const results = await session.run({ input_ids: inputTensor });
    return { data: Array.from(results.hidden_states.data), seqLen };
  } else if (component === 'head') {
    const hiddenTensor = new ort.Tensor('float32', new Float32Array(inputData), [1, seqLen, PETALS_HIDDEN_SIZE]);
    const results = await session.run({ hidden_states: hiddenTensor });
    const logits = results.logits.data;
    let maxIdx = 0, maxVal = -Infinity;
    for (let i = 0; i < logits.length; i++) { if (logits[i] > maxVal) { maxVal = logits[i]; maxIdx = i; } }
    return { tokenId: maxIdx, tokenStr: petalsTokenizer.decode([maxIdx]) };
  } else {
    // layer_N with causal mask
    const hiddenTensor = new ort.Tensor('float32', new Float32Array(inputData), [1, seqLen, PETALS_HIDDEN_SIZE]);
    const mask = new Float32Array(seqLen * seqLen);
    for (let i = 0; i < seqLen; i++) for (let j = 0; j <= i; j++) mask[i * seqLen + j] = 1.0;
    const maskTensor = new ort.Tensor('float32', mask, [1, 1, seqLen, seqLen]);
    const posTensor = new ort.Tensor('int64', BigInt64Array.from(Array.from({length: seqLen}, (_, i) => BigInt(i))), [1, seqLen]);
    const results = await session.run({ input_hidden: hiddenTensor, attention_mask: maskTensor, position_ids: posTensor });
    return { data: Array.from(results.output_hidden.data), seqLen };
  }
}

function petalsGetNextComponent(current) {
  if (current === 'embeds') return 'layer_0';
  if (current === 'head') return null;
  const m = current.match(/layer_(\d+)/);
  if (m) {
    const next = parseInt(m[1]) + 1;
    if (next >= PETALS_TOTAL_LAYERS) return 'head';
    return `layer_${next}`;
  }
  return null;
}

function petalsFindPeerForComponent(comp) {
  for (const [pid, info] of petalsReadyPeers) {
    if (info.layers.includes(comp)) {
      const peer = peers.get(pid);
      if (peer?.dc?.readyState === 'open') return pid;
    }
  }
  return null;
}

// Full forward pass through all layers for a token sequence
async function petalsForwardPass(tokenIds) {
  const seqLen = tokenIds.length;
  let currentComp = 'embeds';
  let currentData = tokenIds;
  let currentSeqLen = seqLen;

  while (currentComp) {
    if (petalsONNXSessions[currentComp]) {
      const result = await petalsRunLayer(currentComp, currentData, currentSeqLen);
      if (currentComp === 'head') return result;
      currentData = result.data;
      currentSeqLen = result.seqLen;
      currentComp = petalsGetNextComponent(currentComp);
    } else {
      const target = petalsFindPeerForComponent(currentComp);
      if (!target) return null;
      const peer = peers.get(target);
      const reqId = Math.floor(Math.random() * 0xFFFFFFFF);
      return new Promise((resolve) => {
        const timeout = setTimeout(() => { fedPending.delete(reqId); resolve(null); }, 15000);
        fedPending.set(reqId, {
          resolve: (res) => { clearTimeout(timeout); resolve(res); },
          onChunk: () => {}
        });
        peer.dc.send(encodeMsg(MSG.PETALS_TENSOR, reqId, 0, { component: currentComp, data: currentData, seqLen: currentSeqLen }));
      });
    }
  }
  return null;
}

async function runPetalsInference(userMessage) {
  isGenerating = true;
  metrics.totalRequests++;
  const aiMsgEl = addMessageToUI("assistant", null, true);
  const startTime = performance.now();
  let fullResponse = "", tokenCount = 0;

  const sysPrompt = "You are a helpful AI assistant running in a decentralized peer-to-peer swarm.";
  const promptText = sysPrompt + "\nUser: " + userMessage + "\nAssistant:";
  if (!petalsTokenizer) { isGenerating = false; return; }
  const encoded = await petalsTokenizer(promptText);
  let allTokenIds = Array.from(encoded.input_ids.data);

  aiMsgEl.querySelector('.typing-indicator')?.remove();
  const contentEl = aiMsgEl.querySelector('.msg-content');
  const pill = document.createElement('div');
  pill.className = 'fed-pill';
  pill.textContent = '\uD83C\uDF38 Web-Petals Swarm \u00B7 ' + petalsMyLayers.length + ' local layers';
  aiMsgEl.querySelector('.msg-body').appendChild(pill);

  try {
    while (tokenCount < 200) {
      const result = await petalsForwardPass(allTokenIds);
      if (!result || result.tokenId === undefined) { fullResponse += " [P2P timeout]"; break; }

      const { tokenId, tokenStr } = result;
      // Check for end-of-text tokens
      if (tokenId === 0 || tokenStr.includes('endoftext') || tokenStr.includes('im_end')) break;

      fullResponse += tokenStr;
      contentEl.textContent = fullResponse;
      document.getElementById('messages').scrollTop = 99999;
      tokenCount++;

      // Append new token and recompute full context next iteration
      allTokenIds.push(tokenId);

      // Safety: cap context at 256 tokens to keep mobile responsive
      if (allTokenIds.length > 256) break;
    }
  } catch(e) { contentEl.textContent += '\n[P2P Error: ' + e.message + ']'; }

  const ms = performance.now() - startTime;
  localTPS = Math.round((tokenCount / (ms / 1000)) * 10) / 10;
  const metaEl = aiMsgEl.querySelector('.msg-meta');
  if (metaEl) metaEl.textContent = 'Swarm-135M \u00B7 ' + petalsMyLayers.length + ' local layers \u00B7 ' + localTPS + ' tok/s \u00B7 ' + Math.round(ms) + 'ms';

  chatHistory.push({ role: "assistant", content: fullResponse });
  broadcastChat("assistant", fullResponse, peerId, "Swarm-135M");
  isGenerating = false;
}

// Handle incoming Petals tensor from peer
async function handlePetalsTensor(fromId, m) {
  const { component, data, seqLen } = m.payload;
  if (!petalsONNXSessions[component]) return;

  const result = await petalsRunLayer(component, data, seqLen);
  const nextComp = petalsGetNextComponent(component);

  if (component === 'head' || !nextComp) {
    const peer = peers.get(fromId);
    if (peer?.dc?.readyState === 'open') {
      peer.dc.send(encodeMsg(MSG.TOK_RES, m.reqId, 0, result));
    }
    return;
  }

  // Chain through local layers
  if (petalsONNXSessions[nextComp]) {
    let comp = nextComp;
    let chainData = result;
    while (true) {
      const nc = petalsGetNextComponent(comp);
      if (!nc || !petalsONNXSessions[nc]) {
        if (comp === 'head' || !nc) {
          const peer = peers.get(fromId);
          if (peer?.dc?.readyState === 'open') peer.dc.send(encodeMsg(MSG.TOK_RES, m.reqId, 0, chainData));
          return;
        }
        const target = petalsFindPeerForComponent(nc);
        if (target) {
          const tp = peers.get(target);
          if (tp?.dc?.readyState === 'open') tp.dc.send(encodeMsg(MSG.PETALS_TENSOR, m.reqId, 0, { component: nc, data: chainData.data, seqLen: chainData.seqLen }));
        }
        return;
      }
      chainData = await petalsRunLayer(nc, chainData.data, chainData.seqLen);
      comp = nc;
    }
  } else {
    const target = petalsFindPeerForComponent(nextComp);
    if (target) {
      const tp = peers.get(target);
      if (tp?.dc?.readyState === 'open') tp.dc.send(encodeMsg(MSG.PETALS_TENSOR, m.reqId, 0, { component: nextComp, data: result.data, seqLen: result.seqLen }));
    }
  }
}
