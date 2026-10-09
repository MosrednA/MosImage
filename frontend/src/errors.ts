export function errorMessage(error: unknown): string { return error instanceof Error ? error.message : "The operation failed."; }
