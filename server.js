// server.js
// A tiny local server that runs a Python script with the topic/bitstream
// from the webpage, and streams its output back live using
// Server-Sent Events (SSE).
// Run with: node server.js
// Then open index.html in your browser.

const http = require('http');
const fs = require('fs');
const url = require('url');
const { spawn } = require('child_process');

const PORT = 3000;

// ---------------------------------------------------------------------
// EDIT THESE for your setup.
// ---------------------------------------------------------------------

// The folder to run the script in.
const WORKING_DIRECTORY = // SET WORKING DIRECTORY

// The script + fixed args (topic/bits are appended safely below —
// never edit this to build a command string with string concatenation).
const SCRIPT = 'sentence_encode.py';

// Track at most one running process at a time.
let currentProcess = null;

function stopCurrentProcess() {
  if (currentProcess) {
    currentProcess.kill();
    currentProcess = null;
  }
}

const server = http.createServer((req, res) => {
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET, OPTIONS');

  if (req.method === 'OPTIONS') {
    res.writeHead(204);
    return res.end();
  }

  const parsedUrl = url.parse(req.url, true); // true => parse query string
  const { pathname, query } = parsedUrl;

  // IMPORTANT: check pathname, not req.url — req.url includes the
  // "?topic=...&bitstream=..." query string, so it never equals
  // '/stream-command' exactly.
  if (pathname === '/stream-command') {
    // Only one active stream at a time; kill any previous one.
    stopCurrentProcess();

    res.writeHead(200, {
      'Content-Type': 'text/event-stream',
      'Cache-Control': 'no-cache',
      Connection: 'keep-alive',
    });

    if (!fs.existsSync(WORKING_DIRECTORY)) {
      res.write(
        `data: [error] WORKING_DIRECTORY does not exist: ${WORKING_DIRECTORY}\n\n`
      );
      res.write(`data: [process exited with code 1]\n\n`);
      return res.end();
    }

    // Read the inputs sent from the page. These are just strings — never
    // built into a shell command, so there's nothing for anyone to inject.
    const topic = '"' + query.topic + '"' || '';
    const bitstream = '"' + query.bitstream + '"' || '';
    const key = query.key || '';

    // Each element of this array is passed to the program as a single,
    // literal argument — NOT interpreted by a shell. So even if topic
    // contains quotes, semicolons, "&&", etc., it's just treated as text,
    // not executed. This is why there's no `shell: true` here.
    const args = [SCRIPT, '--topic', topic, '--message', bitstream, '--key',  key, '--attempts', 6, '--clean'];

    const sendEvent = (line) => {
      // SSE format: each message is "data: <text>\n\n"
      res.write(`data: ${line}\n\n`);
    };

    sendEvent(`[running: python3 ${args.join(' ')}]`);

    const child = spawn('python3', args, {
      cwd: WORKING_DIRECTORY,
      // no shell: true — args are passed directly to python3, not through
      // a shell, so nothing in topic/bitstream can be interpreted as
      // shell syntax.
    });
    currentProcess = child;

    child.stdout.on('data', (chunk) => {
      chunk
        .toString()
        .split('\n')
        .filter((line) => line.trim() !== '')
        .forEach(sendEvent);
    });

    child.stderr.on('data', (chunk) => {
      sendEvent(`[stderr] ${chunk.toString().trim()}`);
    });

    child.on('close', (code) => {
      sendEvent(`[process exited with code ${code}]`);
      res.end();
      if (currentProcess === child) currentProcess = null;
    });

    child.on('error', (err) => {
      sendEvent(`[error starting command: ${err.message}]`);
      res.end();
      if (currentProcess === child) currentProcess = null;
    });

    // If the browser closes the connection (e.g. tab closed, Stop clicked),
    // kill the process so it doesn't keep running in the background.
    req.on('close', () => {
      if (currentProcess === child) {
        child.kill();
        currentProcess = null;
      }
    });

    return;
  }

  if (pathname === '/stop-command') {
    stopCurrentProcess();
    res.setHeader('Content-Type', 'application/json');
    res.writeHead(200);
    return res.end(JSON.stringify({ stopped: true }));
  }

  res.writeHead(404);
  res.end('Not found');
});

server.listen(PORT, () => {
  console.log(`Local command server running at http://localhost:${PORT}`);
  console.log(`It will "cd" into: ${WORKING_DIRECTORY}`);
  console.log(`Then run: python3 ${SCRIPT} --topic <topic> --message <bitstream> --key <key> --clean`);
});
