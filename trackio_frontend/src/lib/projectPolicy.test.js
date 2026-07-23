import { describe, expect, test } from "vitest";
import {
  CANONICAL_MRCRA_PROJECT,
  canonicalProject,
  canonicalProjectList,
} from "./projectPolicy.js";

describe("single-project MRCRA dashboard policy", () => {
  test("removes unrelated and legacy projects", () => {
    expect(canonicalProjectList([
      "mrrn-fineweb",
      CANONICAL_MRCRA_PROJECT,
      "fold-st-coherent-solution-suite",
    ])).toEqual([CANONICAL_MRCRA_PROJECT]);
  });

  test("fails closed when the MRCRA project does not exist", () => {
    expect(canonicalProject(["mrrn-smoke-test"])).toBeNull();
    expect(canonicalProjectList(null)).toEqual([]);
  });
});
