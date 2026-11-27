import type { APIRequestContext } from "@playwright/test";
import { AGENT_STANDIN } from "../playwright.config";

export async function agentCalls(request: APIRequestContext) {
  const response = await request.get(`${AGENT_STANDIN}/calls`);
  return (await response.json()) as string[];
}
