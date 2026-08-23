const POSITIVE_INTEGER = /^[1-9]\d*$/;

export function configuredFplManagerId(
  value: string | undefined,
  { required = false }: { required?: boolean } = {},
): number | null {
  const candidate = value?.trim();
  if (!candidate) {
    if (required) {
      throw new Error("FPL_MANAGER_ID must be configured in Vercel.");
    }
    return null;
  }
  if (!POSITIVE_INTEGER.test(candidate)) {
    throw new Error("FPL_MANAGER_ID must be a positive integer.");
  }

  const managerId = Number(candidate);
  if (!Number.isSafeInteger(managerId)) {
    throw new Error("FPL_MANAGER_ID must be a safe positive integer.");
  }
  return managerId;
}

export function hasConfiguredFplManagerMismatch(
  payload: unknown,
  managerId: number,
  { required = false }: { required?: boolean } = {},
): boolean {
  if (typeof payload !== "object" || payload === null || Array.isArray(payload)) {
    return required;
  }
  const record = payload as Record<string, unknown>;
  const hasManagerId = Object.prototype.hasOwnProperty.call(record, "manager_id");
  return hasManagerId ? record.manager_id !== managerId : required;
}

export function resolveInitialFplManagerId(
  configuredManagerId: number | null,
  storedValue: string | null,
): number | null {
  if (configuredManagerId !== null) return configuredManagerId;
  try {
    return configuredFplManagerId(storedValue ?? undefined);
  } catch {
    return null;
  }
}
