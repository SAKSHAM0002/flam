/**
 * Shape Detector Module
 * Exports all detector components and types
 */

export { ImageProcessor } from "./ImageProcessor.js";
export { ComponentLabeler } from "./ComponentLabeler.js";
export { GeometryUtils } from "./GeometryUtils.js";
export { ConvexHullComputer } from "./ConvexHullComputer.js";
export { PolygonSimplification } from "./PolygonSimplification.js";
export { ShapeClassifier } from "./ShapeClassifier.js";
export type { Point, DetectedShape, DetectionResult, RegionMetrics } from "./types.js";
