import { getCurrentSession } from "@/lib/require-user";
import { hasValidPublicOrigin } from "@/lib/public-request-origin";

const MAX_REQUEST_BYTES = 16_384;

function errorResponse(status: number, code: string, message: string) {
  return Response.json(
    { error: { code, message } },
    {
      status,
      headers: {
        "Cache-Control": "no-store",
        "X-Content-Type-Options": "nosniff",
      },
    },
  );
}

type ProxyOptions = {
  endpoint: `/${string}`;
  unauthorizedMessage: string;
  unavailableMessage: string;
};

export async function proxyInternalJson(
  request: Request,
  options: ProxyOptions,
) {
  const session = await getCurrentSession();
  if (!session) {
    return errorResponse(401, "unauthorized", options.unauthorizedMessage);
  }

  const requestUrl = new URL(request.url);
  if (!hasValidPublicOrigin(request)) {
    return errorResponse(403, "forbidden_origin", "Forespørgslen blev afvist.");
  }

  const contentType = request.headers
    .get("content-type")
    ?.split(";", 1)[0]
    .trim()
    .toLowerCase();
  if (contentType !== "application/json") {
    return errorResponse(
      415,
      "unsupported_media_type",
      "Content-Type skal være application/json.",
    );
  }

  const declaredLength = Number(request.headers.get("content-length") ?? "0");
  if (
    !Number.isFinite(declaredLength) ||
    declaredLength < 0 ||
    declaredLength > MAX_REQUEST_BYTES
  ) {
    return errorResponse(413, "request_too_large", "Forespørgslen er for stor.");
  }

  const internalToken = process.env.INTERNAL_API_TOKEN;
  if (!internalToken) {
    console.error("INTERNAL_API_TOKEN is not configured");
    return errorResponse(
      503,
      "service_unconfigured",
      "Beregningstjenesten er ikke konfigureret.",
    );
  }

  const body = await request.text();
  if (Buffer.byteLength(body, "utf8") > MAX_REQUEST_BYTES) {
    return errorResponse(413, "request_too_large", "Forespørgslen er for stor.");
  }

  let upstream: Response;
  try {
    upstream = await fetch(
      new URL(options.endpoint, requestUrl.origin),
      {
        method: "POST",
        headers: {
          Accept: "application/json",
          "Content-Type": "application/json",
          "X-Internal-Token": internalToken,
        },
        body,
        cache: "no-store",
        redirect: "manual",
        signal: AbortSignal.timeout(55_000),
      },
    );
  } catch (error) {
    console.error(`Internal request to ${options.endpoint} failed`, error);
    return errorResponse(502, "compute_unavailable", options.unavailableMessage);
  }

  const responseBody = await upstream.text();
  return new Response(responseBody, {
    status: upstream.status,
    headers: {
      "Content-Type":
        upstream.headers.get("content-type") ?? "application/json; charset=utf-8",
      "Cache-Control": "no-store",
      "X-Content-Type-Options": "nosniff",
    },
  });
}
