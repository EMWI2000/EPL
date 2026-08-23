import {
  AI_REVIEW_RESPONSE_SCHEMA_VERSION,
  assertAiReviewRequestIsFresh,
  parseAiReviewRequest,
} from "@/lib/ai-review-contract";
import { configuredFplManagerId } from "@/lib/fpl-manager-config";
import {
  DEFAULT_OPENAI_REVIEW_MODEL,
  OpenAiReviewError,
  requestOpenAiReview,
} from "@/lib/openai-ai-review";
import { parsePlannerPayload } from "@/lib/planner-contract";
import { hasValidPublicOrigin } from "@/lib/public-request-origin";
import { getCurrentSession } from "@/lib/require-user";

export const runtime = "nodejs";
export const maxDuration = 60;

const MAX_REQUEST_BYTES = 65_536;
const COOLDOWN_MS = 20_000;
const recentRequests = new Map<string, number>();

function responseHeaders(extra: Record<string, string> = {}) {
  return {
    "Cache-Control": "no-store",
    "X-Content-Type-Options": "nosniff",
    ...extra,
  };
}

function errorResponse(
  status: number,
  code: string,
  message: string,
  extraHeaders: Record<string, string> = {},
) {
  return Response.json(
    { error: { code, message } },
    { status, headers: responseHeaders(extraHeaders) },
  );
}

function consumeCooldown(key: string, now = Date.now()): number {
  const previous = recentRequests.get(key) ?? 0;
  const remaining = previous + COOLDOWN_MS - now;
  if (remaining > 0) return remaining;
  recentRequests.set(key, now);
  if (recentRequests.size > 200) {
    for (const [candidate, timestamp] of recentRequests) {
      if (timestamp + COOLDOWN_MS <= now) recentRequests.delete(candidate);
    }
  }
  return 0;
}

export async function POST(request: Request) {
  const session = await getCurrentSession();
  if (!session) {
    return errorResponse(401, "unauthorized", "Du skal være logget ind for at få en AI-kvalificering.");
  }
  if (!request.headers.get("origin") || !hasValidPublicOrigin(request)) {
    return errorResponse(403, "forbidden_origin", "Forespørgslen blev afvist.");
  }

  const contentType = request.headers
    .get("content-type")
    ?.split(";", 1)[0]
    .trim()
    .toLowerCase();
  if (contentType !== "application/json") {
    return errorResponse(415, "unsupported_media_type", "Content-Type skal være application/json.");
  }
  const declaredLength = Number(request.headers.get("content-length") ?? "0");
  if (
    !Number.isFinite(declaredLength) ||
    declaredLength < 0 ||
    declaredLength > MAX_REQUEST_BYTES
  ) {
    return errorResponse(413, "request_too_large", "Forespørgslen er for stor.");
  }

  const bodyText = await request.text();
  if (Buffer.byteLength(bodyText, "utf8") > MAX_REQUEST_BYTES) {
    return errorResponse(413, "request_too_large", "Forespørgslen er for stor.");
  }

  let body: unknown;
  try {
    body = JSON.parse(bodyText) as unknown;
  } catch {
    return errorResponse(400, "invalid_json", "Forespørgslen indeholder ugyldig JSON.");
  }

  let reviewRequest;
  try {
    reviewRequest = parseAiReviewRequest(body, parsePlannerPayload);
  } catch {
    return errorResponse(400, "invalid_request", "AI-konteksten matcher ikke den beregnede anbefaling.");
  }

  try {
    assertAiReviewRequestIsFresh(reviewRequest);
  } catch {
    return errorResponse(
      409,
      "stale_recommendation",
      "Planen er for gammel til et aktuelt AI-review. Synkronisér og beregn planen igen.",
    );
  }

  const configuredManagerId = configuredFplManagerId(process.env.FPL_MANAGER_ID, {
    required: process.env.VERCEL_ENV === "production",
  });
  if (configuredManagerId !== null && reviewRequest.manager_id !== configuredManagerId) {
    return errorResponse(403, "manager_mismatch", "Forespørgslen matcher ikke det konfigurerede FPL-hold.");
  }

  const apiKey = process.env.OPENAI_API_KEY?.trim();
  if (!apiKey) {
    return errorResponse(
      503,
      "ai_unconfigured",
      "AI-kvalificeringen mangler sikker serverkonfiguration.",
    );
  }

  const cooldownRemaining = consumeCooldown(String(session.user.id));
  if (cooldownRemaining > 0) {
    const retryAfter = Math.max(1, Math.ceil(cooldownRemaining / 1_000));
    return errorResponse(
      429,
      "ai_cooldown",
      `Vent ${retryAfter} sekunder, før du opdaterer AI-kvalificeringen.`,
      { "Retry-After": String(retryAfter) },
    );
  }

  const model = process.env.OPENAI_MODEL?.trim() || DEFAULT_OPENAI_REVIEW_MODEL;
  try {
    const result = await requestOpenAiReview(reviewRequest, { apiKey, model });
    return Response.json(
      {
        schema_version: AI_REVIEW_RESPONSE_SCHEMA_VERSION,
        generated_at: new Date().toISOString(),
        recommendation_generated_at: reviewRequest.recommendation_generated_at,
        target_event: reviewRequest.planner.target_event,
        model,
        review: result.review,
        research: result.research,
      },
      { headers: responseHeaders() },
    );
  } catch (error) {
    const kind = error instanceof OpenAiReviewError ? error.kind : "upstream";
    console.error("AI review failed", kind);
    if (kind === "configuration") {
      return errorResponse(503, "ai_unconfigured", "AI-kvalificeringen er ikke konfigureret korrekt.");
    }
    if (kind === "rate_limited") {
      return errorResponse(429, "ai_rate_limited", "AI-tjenesten har travlt. Prøv igen om lidt.");
    }
    if (kind === "timeout") {
      return errorResponse(504, "ai_timeout", "AI-kvalificeringen nåede ikke at blive færdig. Prøv igen.");
    }
    return errorResponse(
      502,
      "ai_unavailable",
      "AI-kvalificeringen kunne ikke færdiggøres. Den beregnede FPL-plan er stadig gyldig.",
    );
  }
}
