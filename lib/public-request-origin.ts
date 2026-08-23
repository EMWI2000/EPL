function firstHeaderValue(value: string | null) {
  return value?.split(",", 1)[0]?.trim() || null;
}

function normalizeHost(value: string | null) {
  if (!value || /[\s\\/?#@]/.test(value)) return null;

  try {
    const parsed = new URL(`http://${value}`);
    if (parsed.pathname !== "/" || parsed.search || parsed.hash) return null;
    return parsed.host;
  } catch {
    return null;
  }
}

export function resolvePublicRequestOrigin(request: Request) {
  const requestUrl = new URL(request.url);
  const publicHost =
    normalizeHost(firstHeaderValue(request.headers.get("x-forwarded-host"))) ??
    normalizeHost(firstHeaderValue(request.headers.get("host"))) ??
    requestUrl.host;
  const forwardedProtocol = firstHeaderValue(
    request.headers.get("x-forwarded-proto"),
  );
  const protocol = forwardedProtocol ?? requestUrl.protocol.slice(0, -1);

  if (protocol !== "http" && protocol !== "https") {
    return requestUrl.origin;
  }

  return new URL(`${protocol}://${publicHost}`).origin;
}

export function hasValidPublicOrigin(request: Request) {
  const origin = request.headers.get("origin");
  if (!origin) return true;

  try {
    return new URL(origin).origin === resolvePublicRequestOrigin(request);
  } catch {
    return false;
  }
}
