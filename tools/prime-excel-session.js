#!/usr/bin/env node

const RESPONSES_URL = "https://bps.openai.com/basispoints/api/responses";
const ALLOWED_HEADERS = new Set([
  "authorization",
  "chatgpt-account-id",
  "user-agent",
  "x-basispoints-auth-mode",
  "x-openai-account-id",
  "x-openai-account-user-id",
  "x-openai-internal-basispoints-browser-name",
  "x-openai-internal-basispoints-browser-ua-brands",
  "x-openai-internal-basispoints-browser-ua-mobile",
  "x-openai-internal-basispoints-browser-ua-platform",
  "x-openai-internal-basispoints-client-agent-profile",
  "x-openai-internal-basispoints-client-editor",
  "x-openai-internal-basispoints-client-host",
  "x-openai-internal-basispoints-client-platform",
  "x-openai-internal-basispoints-client-platform-class",
  "x-openai-internal-basispoints-client-product",
  "x-openai-internal-basispoints-client-runtime",
  "x-openai-internal-basispoints-office-host",
  "x-openai-internal-basispoints-office-platform",
  "x-stainless-arch",
  "x-stainless-lang",
  "x-stainless-os",
  "x-stainless-package-version",
  "x-stainless-retry-count",
  "x-stainless-runtime",
  "x-stainless-runtime-version",
]);

function parseArguments(argv) {
  const options = {
    devtoolsPort: 9222,
    proxyUrl: "http://127.0.0.1:8000/api/config/excel-session",
    timeoutMs: 300_000,
  };
  for (let index = 0; index < argv.length; index += 1) {
    const argument = argv[index];
    const value = argv[index + 1];
    if (argument === "--devtools-port") {
      options.devtoolsPort = Number.parseInt(value, 10);
    } else if (argument === "--proxy-url") {
      options.proxyUrl = value;
    } else if (argument === "--timeout-ms") {
      options.timeoutMs = Number.parseInt(value, 10);
    } else {
      throw new Error(`Unknown argument: ${argument}`);
    }
    index += 1;
  }
  if (!Number.isInteger(options.devtoolsPort) || options.devtoolsPort < 1 || options.devtoolsPort > 65535) {
    throw new Error("Invalid DevTools port");
  }
  if (!Number.isInteger(options.timeoutMs) || options.timeoutMs < 1) {
    throw new Error("Invalid timeout");
  }
  const proxy = new URL(options.proxyUrl);
  if (
    proxy.protocol !== "http:"
    || !["127.0.0.1", "localhost", "[::1]"].includes(proxy.hostname)
  ) {
    throw new Error("The session endpoint must be an HTTP loopback URL");
  }
  return options;
}

function selectedHeaders(rawHeaders = {}) {
  const result = {};
  for (const [rawName, rawValue] of Object.entries(rawHeaders)) {
    const name = rawName.toLowerCase();
    if (ALLOWED_HEADERS.has(name) && typeof rawValue === "string") {
      result[name] = rawValue;
    }
  }
  return result;
}

function hasRequiredHeaders(headers) {
  return (
    typeof headers.authorization === "string"
    && headers.authorization.toLowerCase().startsWith("bearer ")
    && Boolean(headers["chatgpt-account-id"] || headers["x-openai-account-id"])
  );
}

async function discoverTarget(port) {
  const response = await fetch(`http://127.0.0.1:${port}/json/list`);
  if (!response.ok) {
    throw new Error(`DevTools discovery returned HTTP ${response.status}`);
  }
  const targets = await response.json();
  const target = targets.find((candidate) => (
    candidate.type === "page"
    && candidate.webSocketDebuggerUrl
    && /^https:\/\/bps\.openai\.com\/basispoints\//i.test(candidate.url)
  ));
  if (!target) {
    throw new Error("No ChatGPT Excel add-in WebView was found");
  }
  return target;
}

async function prime(options) {
  const target = await discoverTarget(options.devtoolsPort);
  const socket = new WebSocket(target.webSocketDebuggerUrl);
  const pending = new Map();
  const requests = new Map();
  let nextCommandId = 0;
  let submitting = false;
  let succeeded = false;
  let submitError = null;

  function send(method, params = {}) {
    const id = ++nextCommandId;
    const promise = new Promise((resolve, reject) => {
      pending.set(id, { method, resolve, reject });
    });
    socket.send(JSON.stringify({ id, method, params }));
    return promise;
  }

  async function submit(headers) {
    if (submitting || succeeded || !hasRequiredHeaders(headers)) {
      return;
    }
    submitting = true;
    const response = await fetch(options.proxyUrl, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ headers }),
    });
    const payload = await response.json().catch(() => ({}));
    if (!response.ok) {
      throw new Error(payload.detail || `Proxy returned HTTP ${response.status}`);
    }
    succeeded = true;
    console.log(
      `GPT Excel session primed in memory${payload.expires_at ? `; token expires at ${new Date(payload.expires_at * 1000).toISOString()}` : ""}.`,
    );
    socket.close();
  }

  socket.addEventListener("message", (event) => {
    const payload = JSON.parse(String(event.data));
    if (payload.id) {
      const command = pending.get(payload.id);
      if (!command) {
        return;
      }
      pending.delete(payload.id);
      if (payload.error) {
        command.reject(new Error(`${command.method}: ${payload.error.message}`));
      } else {
        command.resolve(payload.result);
      }
      return;
    }

    if (payload.method === "Network.requestWillBeSent") {
      const request = payload.params?.request;
      if (request?.method === "POST" && request.url === RESPONSES_URL) {
        const headers = selectedHeaders(request.headers);
        requests.set(payload.params.requestId, headers);
        submit(headers).catch((error) => {
          submitError = error;
          socket.close();
        });
      }
      return;
    }

    if (payload.method === "Network.requestWillBeSentExtraInfo") {
      const prior = requests.get(payload.params?.requestId);
      if (!prior) {
        return;
      }
      const headers = { ...prior, ...selectedHeaders(payload.params.headers) };
      requests.set(payload.params.requestId, headers);
      submit(headers).catch((error) => {
        submitError = error;
        socket.close();
      });
    }
  });

  await new Promise((resolve, reject) => {
    socket.addEventListener("open", resolve, { once: true });
    socket.addEventListener("error", () => reject(new Error("Failed to connect to the Excel WebView")), { once: true });
  });
  await send("Network.enable", {
    maxTotalBufferSize: 1_000_000,
    maxResourceBufferSize: 1_000_000,
    maxPostDataSize: 1_000_000,
  });
  console.log("Waiting for the next ChatGPT Excel prompt. Send one message in the Excel add-in now.");

  await new Promise((resolve, reject) => {
    const timeout = setTimeout(() => {
      if (!succeeded) {
        reject(new Error("Timed out waiting for an Excel Responses request"));
        socket.close();
      }
    }, options.timeoutMs);
    socket.addEventListener("close", () => {
      clearTimeout(timeout);
      if (succeeded) {
        resolve();
      } else {
        reject(submitError || new Error("Excel WebView connection closed before a session was captured"));
      }
    }, { once: true });
  });
}

const options = parseArguments(process.argv.slice(2));
await prime(options);
