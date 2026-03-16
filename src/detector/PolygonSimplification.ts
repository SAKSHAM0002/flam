/**
 * Polygon simplification and cleaning module
 * Ramer-Douglas-Peucker simplification, deduplication, collinear vertex merging
 */

import type { Point } from "./types.js";

export class PolygonSimplification {
  /**
   * Ramer-Douglas-Peucker simplification for CLOSED polygons
   * Properly handles the closure by temporarily closing polygon
   */
  static simplifyPolygon(points: Point[], epsilon = 4): Point[] {
    if (!points || points.length < 3) return points.slice();

    const perpendicularDistance = (pt: Point, a: Point, b: Point) => {
      const dx = b.x - a.x;
      const dy = b.y - a.y;

      if (dx === 0 && dy === 0) {
        return Math.hypot(pt.x - a.x, pt.y - a.y);
      }

      const t = ((pt.x - a.x) * dx + (pt.y - a.y) * dy) / (dx * dx + dy * dy);
      const projX = a.x + t * dx;
      const projY = a.y + t * dy;

      return Math.hypot(pt.x - projX, pt.y - projY);
    };

    const rdp = (pts: Point[]): Point[] => {
      if (pts.length <= 2) return pts;

      let dmax = 0;
      let index = 0;

      for (let i = 1; i < pts.length - 1; i++) {
        const d = perpendicularDistance(pts[i], pts[0], pts[pts.length - 1]);
        if (d > dmax) {
          index = i;
          dmax = d;
        }
      }

      if (dmax > epsilon) {
        const left = rdp(pts.slice(0, index + 1));
        const right = rdp(pts.slice(index));
        return left.slice(0, -1).concat(right);
      }

      return [pts[0], pts[pts.length - 1]];
    };

    // Close the polygon by appending the first point
    const closed = [...points, points[0]];
    const simplified = rdp(closed);
    
    // Remove the duplicated closing point
    simplified.pop();

    return simplified.length >= 3 ? simplified : points.slice();
  }

  /**
   * Remove vertices that are closer than tolerance (deduplication)
   */
  static deduplicateCloseVertices(poly: Point[], tol = 2): Point[] {
    if (!poly || poly.length < 2) return poly.slice();
    const out: Point[] = [];
    let last = poly[0];
    out.push(last);
    for (let i = 1; i < poly.length; i++) {
      const p = poly[i];
      const d = Math.hypot(p.x - last.x, p.y - last.y);
      if (d >= tol) {
        out.push(p);
        last = p;
      }
    }
    // final check: if first and last are close, remove last
    if (out.length > 1) {
      const d0 = Math.hypot(out[0].x - out[out.length - 1].x, out[0].y - out[out.length - 1].y);
      if (d0 < tol) out.pop();
    }
    return out;
  }

  /**
   * Merge near-collinear vertices where internal angle is extremely flat
   * angleThresholdDeg: vertices with angle > (180 - threshold) are removed
   */
  static mergeCollinearVertices(poly: Point[], angleThresholdDeg = 20): Point[] {
    if (!poly || poly.length < 3) return poly.slice();

    const internalAngle = (pts: Point[], i: number) => {
      const n = pts.length;
      const prev = pts[(i - 1 + n) % n];
      const cur = pts[i];
      const next = pts[(i + 1) % n];
      const v1x = prev.x - cur.x, v1y = prev.y - cur.y;
      const v2x = next.x - cur.x, v2y = next.y - cur.y;
      const dot = v1x * v2x + v1y * v2y;
      const m1 = Math.hypot(v1x, v1y) || 1;
      const m2 = Math.hypot(v2x, v2y) || 1;
      let cos = dot / (m1 * m2);
      cos = Math.max(-1, Math.min(1, cos));
      const ang = (Math.acos(cos) * 180) / Math.PI;
      return ang;
    };

    // Remove near-collinear vertices in single pass
    const out: Point[] = [];
    const n = poly.length;
    for (let i = 0; i < n; i++) {
      const ang = internalAngle(poly, i);
      // if angle is very flat (close to 180 deg) -> skip this vertex
      if (ang > (180 - angleThresholdDeg)) {
        continue;
      } else {
        out.push(poly[i]);
      }
    }

    // if removal collapsed too much, fallback to original
    if (out.length < 3) return poly.slice();

    // final tighten: if first and last are duplicates due to skipping, remove duplicate last
    const d0 = Math.hypot(out[0].x - out[out.length - 1].x, out[0].y - out[out.length - 1].y);
    if (d0 < 1) out.pop();

    return out;
  }
}
