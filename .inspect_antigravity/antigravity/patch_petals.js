const fs = require('fs');
const file = 'index.html';
let content = fs.readFileSync(file, 'utf-8');
const lines = content.split('\n');

// Find the broken section: starts after "}" on the line after petalsActivate closes,
// and ends before "// ── METRICS SYSTEM"
let startLine = -1, endLine = -1;
for (let i = 0; i < lines.length; i++) {
  // The petalsActivate function ends, then there's a blank line, then broken code
  if (lines[i].includes('
