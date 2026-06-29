const fs = require('fs');
const html = fs.readFileSync('frontend/index.html', 'utf-8');

// evaluate the scripts
// we can just extract the JS from the HTML
const scripts = html.match(/<script>([\s\S]*?)<\/script>/)[1];
fs.writeFileSync('frontend_script.js', scripts);
