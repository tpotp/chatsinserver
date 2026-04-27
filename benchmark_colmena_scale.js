const http = require('http');
const { spawn } = require('child_process');
const path = require('path');
const { chromium, devices } = require('playwright');

const BASE_URL = 'http://127.0.0.1:4173/index.html';
const SCALES = [5, 10, 50, 100, 300, 500, 1000];
const WEBGPU_ARGS = ['--enable-unsafe-webgpu', '--ignore-gpu-blocklist', '--enable-features=Vulkan'];

function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

function httpReady(url) {
  return new Promise(resolve => {
    const req = http.get(url, res => {
      res.resume();
      resolve(res.statusCode >= 200 && res.statusCode < 500);
    });
    req.on('error', () => resolve(false));
    req.setTimeout(3000, () => {
      req.destroy();
      resolve(false);
    });
  });
}

async function ensureServer() {
  if (await httpReady(BASE_URL)) return null;
  const child = spawn(process.execPath, ['local_dev_server.js'], {
    cwd: __dirname,
    stdio: 'ignore',
    windowsHide: true,
  });
  for (let i = 0; i < 40; i++) {
    if (await httpReady(BASE_URL)) return child;
    await sleep(500);
  }
  throw new Error('No pude levantar el servidor local en http://127.0.0.1:4173');
}

async function waitForCollectiveReady(page, requestedCount, timeoutMs) {
  await page.waitForFunction((expected) => {
    const dbg = window.__PUDU_DEBUG__;
    if (!dbg) return false;
    const sim = dbg.getSimulationState?.();
    const stage = dbg.getStageState?.();
    const label = dbg.getCurrentModelLabel?.() || '';
    return !!sim && !!stage && sim.requestedCount === expected && stage.ready && !!label && label !== 'none';
  }, requestedCount, { timeout: timeoutMs });
}

async function collectHostSnapshot(page) {
  return page.evaluate(() => {
    const dbg = window.__PUDU_DEBUG__;
    const stage = dbg?.getStageState?.() || null;
    const metrics = dbg?.getCollectiveMetrics?.() || null;
    const sim = dbg?.getSimulationState?.() || null;
    return {
      stage,
      metrics,
      simulation: sim,
      modelLabel: dbg?.getCurrentModelLabel?.() || null,
      petalsStatus: document.querySelector('#petals-status')?.textContent || null,
      petalsLayers: document.querySelector('#petals-my-layers')?.textContent || null,
    };
  });
}

async function sendClientPrompt(page, prompt, timeoutMs) {
  const beforeCount = await page.evaluate(() => document.querySelectorAll('.msg-content').length);
  const started = Date.now();
  await page.fill('#chat-input', prompt);
  await page.click('#send-btn');
  await page.waitForFunction((previousCount) => {
    const nodes = Array.from(document.querySelectorAll('.msg-content'));
    if (nodes.length <= previousCount) return false;
    const last = (nodes[nodes.length - 1]?.textContent || '').trim();
    if (!last || last.length < 20) return false;
    if (/^⏳/.test(last)) return false;
    if (/Pudu uplink pendiente/i.test(last)) return false;
    return true;
  }, beforeCount, { timeout: timeoutMs });
  const latencyMs = Date.now() - started;
  return page.evaluate((latency) => {
    const contents = Array.from(document.querySelectorAll('.msg-content')).map(node => node.textContent || '');
    const metas = Array.from(document.querySelectorAll('.msg-meta')).map(node => node.textContent || '');
    const pills = Array.from(document.querySelectorAll('.fed-pill')).map(node => node.textContent || '');
    return {
      latencyMs: latency,
      response: contents[contents.length - 1] || '',
      meta: metas[metas.length - 1] || '',
      pill: pills[pills.length - 1] || '',
    };
  }, latencyMs);
}

async function run() {
  const server = await ensureServer();
  const browser = await chromium.launch({
    headless: true,
    channel: 'chrome',
    args: WEBGPU_ARGS,
  });

  const hostContext = await browser.newContext({ viewport: { width: 1440, height: 960 } });
  const clientContext = await browser.newContext({ ...devices['Pixel 7'] });
  const hostPage = await hostContext.newPage();
  const clientPage = await clientContext.newPage();
  const room = `bench-${Date.now()}`;

  try {
    await Promise.all([
      hostPage.goto(`${BASE_URL}?room=${room}`, { waitUntil: 'domcontentloaded', timeout: 180000 }),
      clientPage.goto(`${BASE_URL}?room=${room}&assistOnly=1`, { waitUntil: 'domcontentloaded', timeout: 180000 }),
    ]);
    await hostPage.waitForTimeout(4000);
    await clientPage.waitForTimeout(4000);

    const results = [];
    for (const scale of SCALES) {
      const commandStart = Date.now();
      await hostPage.fill('#chat-input', `/colmena ${scale}`);
      await hostPage.click('#send-btn');
      await waitForCollectiveReady(hostPage, scale, scale >= 300 ? 900000 : 600000);
      await hostPage.waitForTimeout(2500);
      const host = await collectHostSnapshot(hostPage);
      const client = await sendClientPrompt(
        clientPage,
        `Pudu, resume tu nivel actual con ${scale} nodos y una accion concreta para cuidar el bosque chileno.`,
        300000
      );
      const readyLatencyMs = Date.now() - commandStart;
      const stage = host.stage?.activeStage || {};
      const metrics = host.metrics || {};
      const row = {
        scale,
        readyLatencyMs,
        stage: stage.label || '—',
        stageId: stage.id || '—',
        modelLabel: host.modelLabel || stage.modelLabel || '—',
        peerCount: host.stage?.peerCount || 0,
        collectiveTPS: host.stage?.collectiveTPS || metrics.collectiveTPS || 0,
        coreSize: host.stage?.coreSize || metrics.coreSize || 0,
        helpers: host.stage?.helperCount || metrics.helperCount || 0,
        draftHelpers: host.stage?.draftHelperCount || metrics.draftHelperCount || 0,
        cacheHelpers: host.stage?.cacheHelperCount || metrics.cacheHelperCount || 0,
        responseLatencyMs: client.latencyMs,
        response: client.response,
        meta: client.meta,
        pill: client.pill,
      };
      results.push(row);
      console.log(JSON.stringify(row));
      await hostPage.waitForTimeout(2000);
      await clientPage.waitForTimeout(2000);
    }

    console.log('\nRESULTS_TABLE');
    console.log('scale\tstage\tmodel\tpeers\tcollectiveTPS\tcore\thelpers\treadyMs\tresponseMs');
    for (const row of results) {
      console.log([
        row.scale,
        row.stage,
        row.modelLabel,
        row.peerCount,
        row.collectiveTPS,
        row.coreSize,
        row.helpers,
        row.readyLatencyMs,
        row.responseLatencyMs,
      ].join('\t'));
    }
  } finally {
    await browser.close();
    if (server) {
      server.kill();
      await sleep(1000);
    }
  }
}

run().catch(err => {
  console.error(err.stack || err.message);
  process.exit(1);
});
