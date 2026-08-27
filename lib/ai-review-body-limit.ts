// The browser sends the fully validated planner result. The separate request
// sent to OpenAI is compacted and keeps its stricter 64 KiB limit.
export const MAX_AI_REVIEW_INBOUND_REQUEST_BYTES = 256 * 1_024;

export function hasInvalidAiReviewContentLength(value: string | null): boolean {
  const declaredLength = Number(value ?? "0");
  return (
    !Number.isSafeInteger(declaredLength) ||
    declaredLength < 0 ||
    declaredLength > MAX_AI_REVIEW_INBOUND_REQUEST_BYTES
  );
}

export function isAiReviewBodyTooLarge(bodyText: string): boolean {
  return new TextEncoder().encode(bodyText).byteLength > MAX_AI_REVIEW_INBOUND_REQUEST_BYTES;
}
