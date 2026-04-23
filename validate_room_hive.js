const fs = require('fs');
const path = require('path');

const root = process.cwd();
const indexPath = path.join(root, 'index.html');
const hostPath = path.join(root, 'host.html');

const indexHtml = fs.readFileSync(indexPath, 'utf8');
const hostHtml = fs.readFileSync(hostPath, 'utf8');

function section(source, startMarker, endMarker) {
  const start = source.indexOf(startMarker);
  if (start === -1) throw new Error(`Missing marker: ${startMarker}`);
  const end = source.indexOf(endMarker, start);
  if (end === -1) throw new Error(`Missing end marker after: ${startMarker}`);
  return source.slice(start, end);
}

const checks = [];
function check(ok, label, detail = '') {
  checks.push({ ok, label, detail });
}

try {
  const stageBlock = section(indexHtml, 'const PUDU_HIVE_STAGES = [', 'const PUDU_PERSONALITIES = {');
  const stageModels = [...stageBlock.matchAll(/modelLabel:\s*'([^']+)'/g)].map(match => match[1]);
  const stageMins = [...stageBlock.matchAll(/minPeers:\s*(\d+)/g)].map(match => Number(match[1]));
  check(stageModels.length >= 5, 'Hay suficientes etapas de sala', `count=${stageModels.length}`);
  check(stageModels.every(label => label.includes('SmolLM2-1.7B-Q8')), 'Todas las etapas usan SmolLM2-1.7B-Q8', stageModels.join(' | '));
  check(stageMins[0] === 5, 'El quorum inicial exige 5 peers reales', `minPeers[0]=${stageMins[0]}`);

  check(/const ROOM_CAPACITY = 15;/.test(indexHtml), 'index.html mantiene 15 usuarios por sala');
  check(/const ROOM_CAPACITY = 15;/.test(hostHtml), 'host.html mantiene 15 usuarios por sala');

  const runPetalsBlock = section(indexHtml, 'async function runPetalsInference(', '// Handle incoming Petals tensor');
  check(!runPetalsBlock.includes('runSimulationCollectiveInference'), 'runPetalsInference ya no deriva al motor local simulado');
  check(!runPetalsBlock.includes('ensureSimulationCollectiveEngine'), 'runPetalsInference ya no precalienta cerebro simulado');
  check(runPetalsBlock.includes('petalsForwardPass') || runPetalsBlock.includes('generatePetalsTextFromMessages'), 'runPetalsInference sigue usando Petals real');

  const hostBridgeBlock = section(indexHtml, 'function setupHostEmbedBridge()', 'function showModelProgress');
  check(hostBridgeBlock.includes("await petalsActivate('host-warm');"), 'host embed calienta shards Petals reales');

  const bootBlock = section(indexHtml, 'async function boot()', 'boot().catch');
  check(bootBlock.includes('if (!STRICT_PUDU_HIVE) {') && bootBlock.includes('await initEngine();'), 'El engine local quedó detrás del modo no estricto');

  const hostFrameBlock = section(hostHtml, 'function buildHostFrameUrl(', 'function getActiveLobbyEntries');
  check(!hostFrameBlock.includes("searchParams.set('simulate'"), 'host.html ya no crea iframes con simulate=');
  check(hostFrameBlock.includes("searchParams.set('hostEmbed', '1');"), 'host.html sigue creando un nodo real por sala');

  const hostStatusBlock = section(hostHtml, 'function updateControllerFromStatus(status)', 'function renderRooms()');
  check(hostStatusBlock.includes('status.petalsReady'), 'host.html marca lista una sala por shards reales');
  check(hostStatusBlock.includes('status.coverageCount') && hostStatusBlock.includes('coverageTotal'), 'host.html exige cobertura completa antes de marcar ready');

  const strictBranchBlock = section(indexHtml, 'case MSG.CONSENSUS_REQ:', 'case MSG.CACHE_STORE:');
  check(strictBranchBlock.includes('petalsActive && petalsMyLayers.length > 0 && petalsTokenizer'), 'Los requests remotos estrictos se sirven desde shards Petals');
  check(!strictBranchBlock.includes('collective-sim'), 'El handler remoto estricto ya no usa collective-sim');
} catch (error) {
  console.error(`✗ Validation setup failed: ${error.message}`);
  process.exit(1);
}

const failed = checks.filter(item => !item.ok);
for (const item of checks) {
  const prefix = item.ok ? '✓' : '✗';
  console.log(`${prefix} ${item.label}${item.detail ? ` :: ${item.detail}` : ''}`);
}

if (failed.length > 0) {
  console.error(`\n${failed.length} validation check(s) failed.`);
  process.exit(1);
}

console.log(`\n${checks.length} validation checks passed.`);
