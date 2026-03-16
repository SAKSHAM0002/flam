/**
 * Convex hull computation module
 * Implements Monotone Chain algorithm for convex hull
 */

import type { Point } from "./types.js";

export class ConvexHullComputer {
  /**
   * Compute convex hull using Monotone Chain algorithm
   * Sorts points and constructs lower and upper hulls
   */
  static computeHull(points: Point[]): Point[] {
    if (!points || points.length <= 1) return points.slice();

    const pts = points.slice().sort((a, b) => a.x === b.x ? a.y - b.y : a.x - b.x);

    const cross = (o: Point, a: Point, b: Point) => (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x);

    const lower: Point[] = [];
    for (const p of pts) {
      while (lower.length >= 2 && cross(lower[lower.length - 2], lower[lower.length - 1], p) <= 0) {
        lower.pop();
      }
      lower.push(p);
    }

    const upper: Point[] = [];
    for (let i = pts.length - 1; i >= 0; i--) {
      const p = pts[i];
      while (upper.length >= 2 && cross(upper[upper.length - 2], upper[upper.length - 1], p) <= 0) {
        upper.pop();
      }
      upper.push(p);
    }

    lower.pop();
    upper.pop();
    return lower.concat(upper);
  }
}
