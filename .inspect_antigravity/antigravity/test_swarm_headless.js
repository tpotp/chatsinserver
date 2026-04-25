const { chromium } = require('playwright');

(async () => {
  console.log('🚀 Iniciando test automático (Headless)...');
  const browser = await chromium.launch({ headless: true });
  
  // Create host context
  const hostContext = await browser.newContext();
  const hostPage = await hostContext.newPage();
  
  console.log('1️⃣ Levantando HOST simulador (10 celulares)...');
  hostPage.on('console', msg => console.log('[HOST] ' + msg.text()));
  await hostPage.goto('https://chatsinserver.vercel.app/?simulate=10&force_deploy=111', { waitUntil: 'load', timeout: 120000 });
  
  // Wait for simulator to be ready
  await hostPage.setDefaultTimeout(120000);
  await hostPage.waitForFunction(() => {
    const log = document.getElementById('sim-log');
    return log && log.textContent && log.textContent.includes('chilenos registrados');
  }, { timeout: 120000 });
  console.log('✅ HOST simulador inicializado.');

  // Create client context (different profile/storage because it's a new context)
  const clientContext = await browser.newContext();
  const clientPage = await clientContext.newPage();
  clientPage.on('console', msg => console.log('[CLIENT] ' + msg.text()));
  
  console.log('2️⃣ Levantando CLIENTE...');
  await clientPage.goto('https://chatsinserver.vercel.app/?client=true&force_deploy=222', { waitUntil: 'load', timeout: 120000 });
  
  console.log('⏳ Esperando a que el cliente descubra al enjambre en Nostr (15s)...');
  await clientPage.waitForTimeout(15000);
  
  const hiveCount = await clientPage.evaluate(() => {
    return document.querySelector('.header-stats span:nth-child(4)')?.textContent || '0';
  });
  console.log(`🌐 Cliente conectado. HIVE estadistica: ${hiveCount}`);

  console.log('3️⃣ Enviando mensaje de prueba ("hola")...');
  await clientPage.fill('#chat-input', 'hola enjambre chileno!');
  await clientPage.click('#send-btn');
  
  console.log('⏳ Esperando la respuesta P2P (hasta 120s)...');
  
  // Wait for AI response message bubble to appear and not be empty
  try {
    await clientPage.waitForFunction(() => {
      const msgs = document.querySelectorAll('.msg.ai .msg-content');
      if (msgs.length === 0) return false;
      const text = msgs[msgs.length - 1].textContent;
      return text && text.trim() !== '' && !text.includes('Conectando P2P') && !text.includes('Intercepción Sharding');
    }, { timeout: 45000 });
    
    // Extract results
    const result = await clientPage.evaluate(() => {
      const msgs = document.querySelectorAll('.msg.ai');
      const lastMsg = msgs[msgs.length - 1];
      return {
        text: lastMsg.querySelector('.msg-content')?.textContent,
        routePill: lastMsg.querySelector('.msg-body .fed-pill')?.textContent,
        meta: lastMsg.querySelector('.msg-meta')?.textContent
      };
    });
    
    console.log('\n================ RESULTADOS ================');
    console.log(`💬 RESPUESTA : ${result.text}`);
    console.log(`💊 RUTA      : ${result.routePill}`);
    console.log(`📊 METADATA  : ${result.meta}`);
    console.log('============================================\n');
    
    if (result.routePill && result.routePill.includes('Web-Petals')) {
       console.log('🎉 ¡ÉXITO! La inferencia usó correctamente el enjambre.');
    } else {
       console.log('❌ FALLO: No se usó Web-Petals. La derivación falló.');
    }
    await clientPage.screenshot({ path: 'test_result_success.png' });
    
  } catch(e) {
    console.error('❌ Timeout esperando respuesta de la IA', e.message);
    await clientPage.screenshot({ path: 'test_result_error.png' });
  }

  await browser.close();
  console.log('🛑 Test finalizado.');
})();
