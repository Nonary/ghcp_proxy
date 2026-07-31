#!/usr/bin/env node

const childProcess = require("node:child_process");
const fs = require("node:fs");
const http = require("node:http");
const https = require("node:https");
const net = require("node:net");
const path = require("node:path");
const tls = require("node:tls");

const TARGET_HOST = "bps.openai.com";
const TARGET_PATH = "/basispoints/api/responses";
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
    certDir: "",
    listenHost: "127.0.0.1",
    listenPort: 8899,
    proxyUrl: "http://127.0.0.1:8000/api/config/excel-session",
    timeoutMs: 300_000,
  };
  for (let index = 0; index < argv.length; index += 1) {
    const argument = argv[index];
    const value = argv[index + 1];
    if (argument === "--cert-dir") {
      options.certDir = value;
    } else if (argument === "--listen-host") {
      options.listenHost = value;
    } else if (argument === "--listen-port") {
      options.listenPort = Number.parseInt(value, 10);
    } else if (argument === "--proxy-url") {
      options.proxyUrl = value;
    } else if (argument === "--timeout-ms") {
      options.timeoutMs = Number.parseInt(value, 10);
    } else {
      throw new Error(`Unknown argument: ${argument}`);
    }
    index += 1;
  }
  if (!["127.0.0.1", "localhost", "::1"].includes(options.listenHost)) {
    throw new Error("The capture proxy must listen on a loopback address");
  }
  if (!Number.isInteger(options.listenPort) || options.listenPort < 1 || options.listenPort > 65535) {
    throw new Error("Invalid capture proxy port");
  }
  if (!Number.isInteger(options.timeoutMs) || options.timeoutMs < 1) {
    throw new Error("Invalid timeout");
  }
  if (!options.certDir) {
    throw new Error("A certificate directory is required");
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

function runOpenSsl(args) {
  const result = childProcess.spawnSync("openssl", args, {
    encoding: "utf-8",
    stdio: ["ignore", "pipe", "pipe"],
  });
  if (result.error) {
    throw new Error(`Unable to run openssl: ${result.error.message}`);
  }
  if (result.status !== 0) {
    throw new Error((result.stderr || result.stdout || "openssl failed").trim());
  }
}

function ensureCertificateMaterial(certDir) {
  fs.mkdirSync(certDir, { recursive: true, mode: 0o700 });
  const caKey = path.join(certDir, "ghcp-excel-capture-ca.key");
  const caCert = path.join(certDir, "ghcp-excel-capture-ca.pem");
  const leafKey = path.join(certDir, `${TARGET_HOST}.key`);
  const leafCert = path.join(certDir, `${TARGET_HOST}.pem`);
  if (
    fs.existsSync(caKey)
    && fs.existsSync(caCert)
    && fs.existsSync(leafKey)
    && fs.existsSync(leafCert)
  ) {
    return { caCert, caKey, leafCert, leafKey };
  }

  const caConfig = path.join(certDir, "ca.cnf");
  const leafConfig = path.join(certDir, "leaf.cnf");
  const leafCsr = path.join(certDir, `${TARGET_HOST}.csr`);
  fs.writeFileSync(
    caConfig,
    [
      "[req]",
      "distinguished_name=dn",
      "x509_extensions=v3_ca",
      "prompt=no",
      "[dn]",
      "CN=GHCP Proxy Excel Capture CA",
      "[v3_ca]",
      "basicConstraints=critical,CA:true",
      "keyUsage=critical,keyCertSign,cRLSign",
      "subjectKeyIdentifier=hash",
      "",
    ].join("\n"),
    { mode: 0o600 },
  );
  fs.writeFileSync(
    leafConfig,
    [
      "[req]",
      "distinguished_name=dn",
      "req_extensions=req_ext",
      "prompt=no",
      "[dn]",
      `CN=${TARGET_HOST}`,
      "[req_ext]",
      `subjectAltName=DNS:${TARGET_HOST}`,
      "basicConstraints=critical,CA:false",
      "keyUsage=critical,digitalSignature,keyEncipherment",
      "extendedKeyUsage=serverAuth",
      "",
    ].join("\n"),
    { mode: 0o600 },
  );
  runOpenSsl([
    "req",
    "-x509",
    "-newkey",
    "rsa:2048",
    "-nodes",
    "-days",
    "3650",
    "-config",
    caConfig,
    "-keyout",
    caKey,
    "-out",
    caCert,
  ]);
  runOpenSsl([
    "req",
    "-new",
    "-newkey",
    "rsa:2048",
    "-nodes",
    "-config",
    leafConfig,
    "-keyout",
    leafKey,
    "-out",
    leafCsr,
  ]);
  runOpenSsl([
    "x509",
    "-req",
    "-in",
    leafCsr,
    "-CA",
    caCert,
    "-CAkey",
    caKey,
    "-CAcreateserial",
    "-days",
    "825",
    "-sha256",
    "-extfile",
    leafConfig,
    "-extensions",
    "req_ext",
    "-out",
    leafCert,
  ]);
  for (const keyPath of [caKey, leafKey]) {
    fs.chmodSync(keyPath, 0o600);
  }
  return { caCert, caKey, leafCert, leafKey };
}

function selectedHeaders(rawHeaders = {}) {
  const result = {};
  for (const [rawName, rawValue] of Object.entries(rawHeaders)) {
    const name = rawName.toLowerCase();
    if (!ALLOWED_HEADERS.has(name)) {
      continue;
    }
    if (Array.isArray(rawValue)) {
      result[name] = rawValue.join(", ");
    } else if (typeof rawValue === "string") {
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

function validToolsVersionId(value) {
  return typeof value === "string" && /^[A-Za-z0-9._-]{1,160}$/.test(value);
}

function requestToolsVersionId(body) {
  if (!body.length) {
    return null;
  }
  try {
    const value = JSON.parse(body.toString("utf-8"))?.metadata?.bps_tools_version_id;
    return validToolsVersionId(value) ? value : null;
  } catch {
    return null;
  }
}

function requestTarget(req) {
  const host = String(req.headers.host || "").split(":")[0].toLowerCase();
  let pathname = req.url || "/";
  let search = "";
  try {
    const parsed = new URL(req.url, `https://${req.headers.host || TARGET_HOST}`);
    pathname = parsed.pathname;
    search = parsed.search;
  } catch {
    const split = pathname.indexOf("?");
    if (split >= 0) {
      search = pathname.slice(split);
      pathname = pathname.slice(0, split);
    }
  }
  return { host, path: `${pathname}${search}`, pathname };
}

function isExcelResponsesRequest(req) {
  const target = requestTarget(req);
  return (
    req.method === "POST"
    && target.host === TARGET_HOST
    && target.pathname.replace(/\/+$/, "") === TARGET_PATH
  );
}

function upstreamHeaders(headers) {
  const result = { ...headers };
  delete result["proxy-authorization"];
  delete result["proxy-connection"];
  return result;
}

async function submitSession(proxyUrl, headers, body) {
  const response = await fetch(proxyUrl, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({
      headers,
      tools_version_id: requestToolsVersionId(body),
    }),
  });
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(payload.detail || `Proxy returned HTTP ${response.status}`);
  }
  console.log(
    `GPT Excel session captured through local proxy${payload.expires_at ? `; token expires at ${new Date(payload.expires_at * 1000).toISOString()}` : ""}.`,
  );
}

function closeServer(server) {
  server.close(() => {
    process.exit(0);
  });
}

function forwardRequest(req, res, body, { forceConnectionClose = false, onComplete = null } = {}) {
  const target = requestTarget(req);
  const transport = target.host === TARGET_HOST ? https : http;
  let completed = false;
  function complete() {
    if (completed) {
      return;
    }
    completed = true;
    if (typeof onComplete === "function") {
      onComplete();
    }
  }
  const upstream = transport.request(
    {
      hostname: target.host,
      port: target.host === TARGET_HOST ? 443 : 80,
      method: req.method,
      path: target.path,
      headers: upstreamHeaders(req.headers),
    },
    (upstreamResponse) => {
      const responseHeaders = { ...upstreamResponse.headers };
      if (forceConnectionClose) {
        responseHeaders.connection = "close";
      }
      res.writeHead(upstreamResponse.statusCode || 502, responseHeaders);
      upstreamResponse.pipe(res);
      upstreamResponse.on("end", complete);
      upstreamResponse.on("error", complete);
    },
  );
  upstream.on("error", (error) => {
    if (!res.headersSent) {
      res.writeHead(502, { "content-type": "text/plain" });
    }
    res.end(`Proxy upstream error: ${error.message}`);
    complete();
  });
  res.on("finish", complete);
  res.on("close", complete);
  upstream.end(body);
}

function tunnelRequest(req, clientSocket, head) {
  const [hostname, rawPort] = String(req.url || "").split(":");
  const port = Number.parseInt(rawPort || "443", 10);
  if (!hostname || !Number.isInteger(port)) {
    clientSocket.end("HTTP/1.1 400 Bad Request\r\n\r\n");
    return;
  }
  const serverSocket = net.connect(port, hostname, () => {
    clientSocket.write("HTTP/1.1 200 Connection Established\r\n\r\n");
    if (head?.length) {
      serverSocket.write(head);
    }
    clientSocket.pipe(serverSocket);
    serverSocket.pipe(clientSocket);
  });
  serverSocket.on("error", () => {
    clientSocket.end("HTTP/1.1 502 Bad Gateway\r\n\r\n");
  });
}

async function run(options) {
  const certs = ensureCertificateMaterial(options.certDir);
  const secureContext = tls.createSecureContext({
    cert: fs.readFileSync(certs.leafCert),
    key: fs.readFileSync(certs.leafKey),
  });
  let succeeded = false;
  let submitError = null;

  const server = http.createServer((req, res) => {
    const chunks = [];
    req.on("data", (chunk) => {
      chunks.push(chunk);
    });
    req.on("end", () => {
      const body = Buffer.concat(chunks);
      const shouldCapture = !succeeded && isExcelResponsesRequest(req);
      let captureComplete = false;
      let responseComplete = false;
      const maybeClose = () => {
        if (succeeded && captureComplete && responseComplete) {
          closeServer(server);
        }
      };
      if (shouldCapture) {
        const headers = selectedHeaders(req.headers);
        if (hasRequiredHeaders(headers)) {
          submitSession(options.proxyUrl, headers, body)
            .then(() => {
              succeeded = true;
              captureComplete = true;
              setTimeout(() => closeServer(server), 15_000).unref();
              maybeClose();
            })
            .catch((error) => {
              submitError = error;
              console.error(error?.message || String(error));
            });
        }
      }
      forwardRequest(req, res, body, {
        forceConnectionClose: shouldCapture,
        onComplete: () => {
          responseComplete = true;
          maybeClose();
        },
      });
    });
  });

  server.on("connect", (req, clientSocket, head) => {
    const [hostname, rawPort] = String(req.url || "").split(":");
    const port = Number.parseInt(rawPort || "443", 10);
    if (hostname?.toLowerCase() !== TARGET_HOST || port !== 443) {
      tunnelRequest(req, clientSocket, head);
      return;
    }
    clientSocket.write("HTTP/1.1 200 Connection Established\r\n\r\n");
    const tlsSocket = new tls.TLSSocket(clientSocket, {
      isServer: true,
      ALPNProtocols: ["http/1.1"],
      secureContext,
    });
    tlsSocket.on("error", () => {});
    if (head?.length) {
      tlsSocket.unshift(head);
    }
    server.emit("connection", tlsSocket);
  });

  await new Promise((resolve, reject) => {
    server.once("error", reject);
    server.listen(options.listenPort, options.listenHost, resolve);
  });
  console.log(`GPT Excel local capture proxy listening on ${options.listenHost}:${options.listenPort}.`);
  console.log(`Trust this local CA certificate before sending the Excel prompt: ${certs.caCert}`);

  await new Promise((resolve, reject) => {
    const timeout = setTimeout(() => {
      server.close();
      reject(submitError || new Error("Timed out waiting for a proxied ChatGPT Excel Responses request"));
    }, options.timeoutMs);
    server.on("close", () => {
      clearTimeout(timeout);
      if (succeeded) {
        resolve();
      }
    });
  });
}

async function main() {
  const options = parseArguments(process.argv.slice(2));
  await run(options);
}

main().catch((error) => {
  console.error(error?.message || String(error));
  process.exitCode = 1;
});
