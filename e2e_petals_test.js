const { chromium } = require('playwright');

(async () => {
  const browser = await chromium.launch({ headless: true });
  console.log("Browser launched.");

  // Context 1: The Host (Motherboard)
  const hostContext = await browser.newContext();
  const hostPage = await hostContext.newPage();
  
  console.log("Navigating Host to ?simulate=10 ...");
  await hostPage.goto('https://chatsinserver.vercel.app/?simulate=10');
  
  // Wait for the host to successfully load the 26 layers
  console.log("Waiting for Host to load ONNX WebGPU Sessions...");
  
  try {
      await hostPage.waitForSelector('text=✅ SmolLM2-1.7B COMPLETO', { timeout: 180000 });
      console.log("Host loaded successfully!");
  } catch(e) {
      console.log("Host failed to load completely. Check #sim-log:");
      const log = await hostPage.locator('#sim-log').innerText();
      console.log(log);
      await browser.close();
      return;
  }

  // Context 2: The Client (Mobile device)
  const clientContext = await browser.newContext({
    userAgent: 'Mozilla/5.0 (Linux; Android 10; Pixel 3 XL) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Mobile Safari/537.36',
    viewport: { width: 412, height: 823 }
  });
  const clientPage = await clientContext.newPage();
  console.log("Navigating Client to chat room...");
  await clientPage.goto('https://chatsinserver.vercel.app/');
  
  // Wait to connect
  await clientPage.waitForTimeout(5000);
  
  // Type message
  console.log("Client typing message...");
  await clientPage.fill('#chat-input', 'Escribe un poema corto sobre un pudú chileno');
  await clientPage.click('#send-btn');
  
  console.log("Waiting for Swarm response...");
  await clientPage.waitForTimeout(3000); // give time for intercept and fetch
  
  let attempts = 0;
  let text = "";
  while (attempts < 20) { // 20*2s = 40 seconds wait
     await clientPage.waitForTimeout(2000);
     const msgs = await clientPage.locator('.message.assistant .msg-content').allInnerTexts();
     text = msgs[msgs.length - 1] || "";
     console.log(`Current response (len ${text.length}): ${text.substring(0, 50)}...`);
     
     if (text.includes('[P2P Error')) {
        console.log("ERROR Detected!");
        break;
     }

     const logs = await hostPage.locator('#sim-log').innerText();
     if (logs.includes('Error')) {
        console.log("HOST ERROR Detected!");
        console.log(logs.slice(-500));
     }

     // If response is somewhat fleshed out and contains standard output
     if (text.length > 50 && !text.includes('...')) {
         break;
     }
     attempts++;
  }

  console.log("Final Response on Mobile:");
  console.log(text);
  
  await browser.close();
})();
