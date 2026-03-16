/**
 * Shape classification module
 * Determines shape type based on vertex count, angles, solidity, and circularity
 */

import type { DetectedShape } from "./types.js";

export class ShapeClassifier {
  /**
   * Classify a shape based on its geometry
   * Uses multi-stage classification: solidity-first, then vertex count, then metrics
   */
  static classifyShape(
    hullVertexCount: number,
    angles: number[],
    solidity: number,
    circularity: number,
    bboxWidth: number,
    bboxHeight: number
  ): { type: DetectedShape["type"]; confidence: number } {
    let type: DetectedShape["type"] = "star";

    // Detect concave shapes FIRST (stars, etc.) using solidity
    if (solidity < 0.85 && hullVertexCount >= 5) {
      type = "star";
      console.log(`[DEBUG]   → STAR (concave shape, solidity ${solidity.toFixed(3)} < 0.85)`);
    }
    // Normal convex polygons by vertex count
    else if (hullVertexCount === 3) {
      type = "triangle";
      console.log(`[DEBUG]   → TRIANGLE (${hullVertexCount} vertices)`);
    }
    else if (hullVertexCount === 4) {
      // Only classify as rectangle/square if 3+ angles are near 90° (true right angles)
      const rectAngles = angles.filter(a => Math.abs(a - 90) < 20).length;
      
      if (rectAngles >= 3) {
        // Distinguish square vs rectangle by aspect ratio
        const ratio = Math.max(bboxWidth, bboxHeight) / Math.min(bboxWidth, bboxHeight);
        
        if (ratio < 1.1) {
          type = "square";
          console.log(`[DEBUG]   → SQUARE (${hullVertexCount} vertices, ${rectAngles} right angles, ratio ${ratio.toFixed(3)})`);
        } else {
          type = "rectangle";
          console.log(`[DEBUG]   → RECTANGLE (${hullVertexCount} vertices, ${rectAngles} right angles, ratio ${ratio.toFixed(3)})`);
        }
      } else {
        // Skewed quad from curved triangle or other distorted shape
        type = "triangle";
        console.log(`[DEBUG]   → TRIANGLE (4-vertex shape with only ${rectAngles} right angles, likely curved triangle)`);
      }
    }
    else if (hullVertexCount === 5 || hullVertexCount === 6) {
      type = "pentagon";
      console.log(`[DEBUG]   → PENTAGON (${hullVertexCount} vertices, tolerant)`);
    }
    // Circles usually produce many vertices after simplification
    else if (hullVertexCount >= 7 && circularity > 0.88) {
      type = "circle";
      console.log(`[DEBUG]   → CIRCLE (${hullVertexCount} vertices, circularity ${circularity.toFixed(3)})`);
    }
    // Strict fallback: only circle if very round
    else if (circularity > 0.88) {
      type = "circle";
      console.log(`[DEBUG]   → CIRCLE (high circularity ${circularity.toFixed(3)})`);
    }
    else {
      type = "star";
      console.log(`[DEBUG]   → STAR (fallback safe classification)`);
    }

    // Confidence scoring (sequential like original)
    let confidence = 0.85;
    if (type === "circle") confidence = Math.min(0.97, 0.4 + circularity);
    if (type === "triangle" && hullVertexCount === 3) confidence = 0.92;
    if (type === "rectangle" && hullVertexCount === 4) confidence = 0.95;
    if (type === "square" && hullVertexCount === 4) confidence = 0.96;
    if (type === "pentagon" && hullVertexCount === 5) confidence = 0.92;
    if (type === "star") confidence = Math.max(0.65, 0.9 - solidity);

    confidence = Math.max(0.5, Math.min(0.97, confidence));

    return { type, confidence };
  }

  /**
   * Detect and fix degenerate quads (curved triangles simplified to 4 vertices)
   * Returns corrected vertex count
   */
  static correctDegenerateQuad(hullVertexCount: number, angles: number[]): number {
    if (hullVertexCount === 4 && angles.length === 4) {
      const largeAngles = angles.filter(a => a > 150).length;
      
      // if two+ angles are almost straight lines → degenerate quad from curved triangle
      if (largeAngles >= 2) {
        console.log(`[DEBUG] Degenerate quad → triangle (${largeAngles} angles > 150°)`);
        return 3;
      }
    }
    return hullVertexCount;
  }
}
