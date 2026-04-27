const { chromium } = require('playwright');

(async () => {
  console.log('🚀 Iniciando test automático (Headless)...');
  const browser = await chromium.launch({ headless: true });
  
  // Create host context
  const hostContext = await browser.newContext();
  const hostPage = await hostContext.newPage();
  
  console.log('1️⃣ Levantando HOST simulador...');
  hostPage.on('console', msg => console.log('[HOST] ' + msg.text()));
  await hostPage.goto('http://localhost:8080/index.html?simulate=5&room=sala-test-123', { waitUntil: 'load', timeout: 120000 });
  
  // Wait for simulator to be ready
  await hostPage.waitForTimeout(5000);
  console.log('✅ HOST simulador inicializado en sala-test-123.');

  // Create client context
  const clientContext = await browser.newContext();
  const clientPage = await clientContext.newPage();
  clientPage.on('console', msg => console.log('[CLIENT] ' + msg.text()));
  clientPage.on('pageerror', err => console.log('[CLIENT ERROR] ' + err.message));
  clientPage.on('requestfailed', request => console.log('[CLIENT REQ FAILED] ' + request.url()));
  
  console.log('2️⃣ Levantando CLIENTE index2.html...');
  await clientPage.goto('http://localhost:8080/index2.html?room=sala-test-123', { waitUntil: 'load', timeout: 120000 });
  
  console.log('⏳ Esperando a que el cliente se una y descubra a la colmena (10s)...');
  
  // Simulate clicking the "Enter" button if needed
  try {
     await clientPage.waitForSelector('#enter-btn-lite', { timeout: 10000 });
     await clientPage.click('#enter-btn-lite'); // bypass 3D waiting load
     console.log('👆 Clicked Enter Lite');
  } catch(e) {
     console.log('No enter button needed or not found');
  }

  await clientPage.waitForTimeout(10000);
  
  console.log('3️⃣ Enviando mensaje @pudu de prueba...');
  await clientPage.fill('#chat-input', '@pudu hola estás ahí?');
  await clientPage.click('.send-btn');
  
  console.log('⏳ Esperando la respuesta P2P (hasta 30s)...');
  
  try {
    await clientPage.waitForFunction(() => {
      const msgs = document.querySelectorAll('.msg-row .msg-bubble');
      if (msgs.length === 0) return false;
      const text = msgs[msgs.length - 1].textContent;
      return text && text.trim() !== '' && text.includes('🦌') && !text.includes('procesando');
    }, { timeout: 30000 });
    
    // Extract results
    const result = await clientPage.evaluate(() => {
      const msgs = document.querySelectorAll('.msg-row');
      const lastMsg = msgs[msgs.length - 1];
      return {
        text: lastMsg.querySelector('.msg-bubble')?.textContent,
      };
    });
    
    console.log('\n================ RESULTADOS ================');
    console.log(`💬 RESPUESTA : ${result.text}`);
    console.log('============================================\n');
    console.log('🎉 ¡ÉXITO! Respuesta de la colmena.');
    await clientPage.screenshot({ path: 'test_success.png' });
  } catch(e) {
    console.error('❌ Timeout esperando respuesta de la IA', e.message);
    await clientPage.screenshot({ path: 'test_error.png' });
  }

  await browser.close();
  console.log('🛑 Test finalizado.');
  process.exit(0);
})();
