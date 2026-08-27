export function canSubmitWeeklyPlanner(input: {
  hasManagerSync: boolean;
  squadConfirmed: boolean;
  hasFreeHitWarning: boolean;
  isSyncing: boolean;
  syncStateValid: boolean;
  hasSyncError: boolean;
}): boolean {
  return input.hasManagerSync &&
    input.squadConfirmed &&
    !input.hasFreeHitWarning &&
    !input.isSyncing &&
    input.syncStateValid &&
    !input.hasSyncError;
}
