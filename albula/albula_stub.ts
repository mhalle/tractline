// Stands in for Albula's SDK so ukf.ts runs on its own: dwi.ts imports these NRRD readers, which the
// comparison never calls (it hands trackUkf a UkfData built from UKF's own normalized signal).
export type Volume = unknown;
const absent = () => { throw new Error("albula SDK not present: this harness builds UkfData directly"); };
export const nrrdDecode = absent, nrrdGeometry = absent, nrrdVectors = absent, nrrdSampleReader = absent,
  nrrdSpaceFlip = absent, nrrdSplitHeader = absent;
export const NRRD_TYPE_BYTES: Record<string, number> = {};
