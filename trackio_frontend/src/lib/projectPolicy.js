export const CANONICAL_MRCRA_PROJECT = "mrcra-fineweb";

export function canonicalProjectList(projects) {
  return Array.isArray(projects) && projects.includes(CANONICAL_MRCRA_PROJECT)
    ? [CANONICAL_MRCRA_PROJECT]
    : [];
}

export function canonicalProject(projects) {
  return canonicalProjectList(projects)[0] ?? null;
}
