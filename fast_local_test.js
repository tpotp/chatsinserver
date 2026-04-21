const http = require('http');
const fs = require('fs');
const path = require('path');
const { chromium } = require('playwright');

const PORT = 8888;
const server = http.createServer((req, res) => {
  if (req.url.startsWith('/') || req.url.includes('index.html')) {
    fs.readFile(path.join(__dirname, 'index.html'), (err, data) => {
      if (err) { res.writeHead(404); res.end(JSON.stringify(err)); return; }
      res.writeHead(200, { 'Content-Type': 'text/html' });
      res.end(data);
    });
  } else {
    res.writeHead(404); res.end();
  }
});

server.listen(PORT, async () => {
  console.log(`🌍 Servidor local iniciado http://localhost:${PORT}`);
  
  try {
    const browser = await chromium.launch({ headless: true });
    
    // HOST
    console.log('1️⃣ Levantando HOST simulador (10 celulares)...');
    const hostContext = await browser.newContext();
    const hostPage = await hostContext.newPage();
    hostPage.setDefaultTimeout(120000);
    //hostPage.on('console', msg => console.log('[HOST] ' + msg.text()));
    
    await hostPage.goto(`http://localhost:${PORT}/?simulate=10`, { timeout: 120000 });
    await hostPage.waitForFunction(() => {
      const log = document.getElementById('sim-log');
      return log && log.textContent && log.textContent.includes('chilenos registrados');
    }, { timeout: 120000 });
    console.log('✅ HOST simulador listo.');

    // CLIENT
    console.log('2️⃣ Levantando CLIENTE...');
    const clientContext = await browser.newContext();
    const clientPage = await clientContext.newPage();
    clientPage.setDefaultTimeout(120000);
    //clientPage.on('console', msg => console.log('[CLIENT] ' + msg.text()));
    
    await clientPage.goto(`http://localhost:${PORT}/?client=true`, { timeout: 120000 });
    
    console.log('⏳ Esperando a que el cliente descubra al enjambre en Nostr (12s)...');
    await clientPage.waitForTimeout(12000);
    
    const hiveCount = await clientPage.evaluate(() => document.querySelector('.header-stats span:nth-child(4)')?.textContent || '0');
    console.log(`🌐 Cliente conectado. HIVE estadistica: ${hiveCount}`);

    console.log('3️⃣ Enviando mensaje de prueba ("hola")...');
    await clientPage.fill('#chat-input', 'hola enjambre chileno!');
    await clientPage.click('#send-btn');
    
    console.log('⏳ Esperando la respuesta automática del enjambre (esto fallará Web-Petals localmente y probará el Fallback a Nostr automáticamente)...');
    
    await clientPage.waitForFunction(() => {
      const msgs = document.querySelectorAll('.msg.ai .msg-content');
      if (msgs.length === 0) return false;
      const text = msgs[msgs.length - 1].textContent;
      return text && text.trim() !== '' && !text.includes('Conectando P2P') && !text.includes('Intercepción Sharding') && !text.includes('Petals interrumpido');
    }, { timeout: 45000 });
    
    const result = await clientPage.evaluate(() => {
      const msgs = document.querySelectorAll('.msg.ai');
      const lastMsg = msgs[msgs.length - 1];
      return {
        text: lastMsg.querySelector('.msg-content')?.textContent,
        routePill: lastMsg.querySelector('.fed-pill')?.textContent,
        meta: lastMsg.querySelector('.msg-meta')?.textContent
      };
    });
    
    console.log('\n================ RESULTADOS ================');
    console.log(`💬 RESPUESTA : ${result.text}`);
    console.log(`💊 RUTA      : ${result.routePill}`);
    console.log(`📊 METADATA  : ${result.meta}`);
    console.log('============================================\n');
    
    if (result.routePill && result.routePill.includes('Fallback')) {
       console.log('🎉 ¡ÉXITO! La inferencia cayó elegantemente en el Fallback de Nostr, sorteando los timeouts P2P o falta de GPU de la simulación.');
    } else {
       console.log('❌ FALLO: No se detectó correctamente el Fallback. Pill actual:', result.routePill);
    }
    
    await browser.close();
  } catch(e) {
    console.error('❌ Error general del test:', e.message);
  } finally {
    server.close();
    process.exit(0);
  }
});
