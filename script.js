import { decodeMessage } from './decode.js';

const startBtn = document.getElementById('startBtn');
const stopBtn = document.getElementById('stopBtn');
const output = document.getElementById('output');
const statusText = document.getElementById('statusText');
const decodeBtn = document.getElementById('decodeBtn');
const decodeOutput = document.getElementById('decodeOutput');

let eventSource = null;

let first_line = true;

function appendLine(line) {
  if (first_line) {
    first_line = false;
  }
  else {
    output.textContent += line;
  }
  output.scrollTop = output.scrollHeight; // auto-scroll to bottom
}

function stopStream() {
  if (eventSource) {
    eventSource.close();
    eventSource = null;
  }
  startBtn.disabled = false;
  stopBtn.disabled = true;
  first_line = true;
  statusText.textContent = 'Stopped.';
}

startBtn.addEventListener('click', () => {
  output.textContent = '';
  statusText.textContent = 'Connecting…';
  startBtn.disabled = true;
  stopBtn.disabled = false;

  // Read the two input fields.
  const topic = document.getElementById('ftopic').value;
  const bitstream = document.getElementById('fbitstream').value;
  const key = document.getElementById('fkey').value;

  // Send them as query parameters. encodeURIComponent keeps special
  // characters (spaces, &, etc.) from breaking the URL. This is safe:
  // the server treats them as plain string data, never as shell
  // commands, so there's nothing to inject here.
  const params = new URLSearchParams({ topic, bitstream });

  // EventSource opens a persistent connection and fires onmessage
  // every time the server sends a new "data: ..." event.
  eventSource = new EventSource(
    `http://localhost:3000/stream-command?${params.toString()}`
  );

  eventSource.onopen = () => {
    statusText.textContent = 'Streaming…';
  };

  eventSource.onmessage = (event) => {
    appendLine(event.data);
  };

  eventSource.onerror = () => {
    statusText.textContent =
      'Connection closed or lost (is server.js running?).';
    stopStream();
  };
});

stopBtn.addEventListener('click', async () => {
  // Tell the server to kill the running process, then close our connection.
  try {
    await fetch('http://localhost:3000/stop-command');
  } catch (err) {
    // Server may already be down; ignore.
  }
  stopStream();
});

decodeBtn.addEventListener('click', () => {
  // Decoding runs entirely in the browser — no server needed.
  const encoded = document.getElementById('fencoded').value;
  const key = document.getElementById('fkey').value;

  // unishox2.js was loaded as a classic script, so its functions are on window,
  // which is all decodeMessage needs from the module.
  try {
    decodeOutput.textContent = decodeMessage(encoded, key, window);
  } catch (err) {
    // A wrong key produces arbitrary bits, so a failed decode is expected here
    // rather than exceptional; show why instead of failing silently.
    decodeOutput.textContent = `Could not decode: ${err.message}`;
  }
});
