// A stand-in for the agent's service in the end-to-end tests. It records each POST
// from the backend and lists them at GET /calls.
import { createServer } from "node:http";

const PORT = 8003;
const calls = [];

createServer((request, response) => {
  response.setHeader("content-type", "application/json");
  if (request.method === "POST") {
    calls.push(`POST ${request.url}`);
    response.end(JSON.stringify({ status: "recorded" }));
  } else if (request.url === "/calls") {
    response.end(JSON.stringify(calls));
  } else {
    response.end(JSON.stringify({ status: "ok" }));
  }
}).listen(PORT, "127.0.0.1");
