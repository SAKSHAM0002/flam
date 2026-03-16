/**
 * Geometry utilities module
 * Core geometry calculations: area, perimeter, angles, region metrics
 */

import type { Point, RegionMetrics } from "./types.js";

export class GeometryUtils {
  /**
   * Compute region metrics: area, perimeter, bounding box, center, boundary points, aspect ratio
   */
  static computeRegionMetrics(regionPixels: Array<{ x: number; y: number }>, w: number, h: number): RegionMetrics {
    const set = new Set<string>();
    for (const p of regionPixels) set.add(`${p.x},${p.y}`);

    let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
    for (const p of regionPixels) {
      if (p.x < minX) minX = p.x;
      if (p.x > maxX) maxX = p.x;
      if (p.y < minY) minY = p.y;
      if (p.y > maxY) maxY = p.y;
    }
    const boundingBox = { x: minX, y: minY, width: Math.max(1, maxX - minX), height: Math.max(1, maxY - minY) };
    const center = { x: minX + boundingBox.width / 2, y: minY + boundingBox.height / 2 };

    const area = regionPixels.length;

    // Boundary detection: pixels adjacent to background
    const boundaryPoints: Point[] = [];
    for (const p of regionPixels) {
      let isBoundary = false;
      for (let dy = -1; dy <= 1 && !isBoundary; dy++) {
        for (let dx = -1; dx <= 1; dx++) {
          if (dx === 0 && dy === 0) continue;
          const nx = p.x + dx;
          const ny = p.y + dy;
          if (nx < 0 || nx >= w || ny < 0 || ny >= h || !set.has(`${nx},${ny}`)) {
            isBoundary = true;
            break;
          }
        }
      }
      if (isBoundary) boundaryPoints.push({ x: p.x, y: p.y });
    }

    // Perimeter: count boundary edges with correction for diagonals
    let perimeter = 0;
    const bset = new Set(boundaryPoints.map(p => `${p.x},${p.y}`));
    for (const p of boundaryPoints) {
      for (let ny = p.y - 1; ny <= p.y + 1; ny++) {
        for (let nx = p.x - 1; nx <= p.x + 1; nx++) {
          if (nx === p.x && ny === p.y) continue;
          if (nx < 0 || nx >= w || ny < 0 || ny >= h || !bset.has(`${nx},${ny}`)) {
            const dx = nx - p.x;
            const dy = ny - p.y;
            const d = Math.hypot(dx, dy);
            perimeter += d;
          }
        }
      }
    }
    perimeter = perimeter / 2; // empirical correction

    const aspectRatio = boundingBox.height > 0 ? boundingBox.width / boundingBox.height : 1;

    return { area, perimeter, boundingBox, center, boundaryPoints, aspectRatio };
  }

  /**
   * Calculate polygon area using shoelace formula
   */
  static polygonArea(poly: Point[]): number {
    if (!poly || poly.length < 3) return 0;
    let area = 0;
    for (let i = 0; i < poly.length; i++) {
      const a = poly[i];
      const b = poly[(i + 1) % poly.length];
      area += a.x * b.y - a.y * b.x;
    }
    return Math.abs(area) / 2;
  }

  /**
   * Calculate polygon perimeter: sum of edge lengths
   */
  static polygonPerimeter(poly: Point[]): number {
    if (!poly || poly.length < 2) return 0;
    let p = 0;
    for (let i = 0; i < poly.length; i++) {
      const a = poly[i];
      const b = poly[(i + 1) % poly.length];
      p += Math.hypot(b.x - a.x, b.y - a.y);
    }
    return p;
  }

  /**
   * Compute internal angles at each polygon vertex (degrees)
   */
  static computePolygonInternalAngles(poly: Point[]): number[] {
    if (!poly || poly.length < 3) return [];
    const angles: number[] = [];
    const n = poly.length;
    for (let i = 0; i < n; i++) {
      const prev = poly[(i - 1 + n) % n];
      const cur = poly[i];
      const next = poly[(i + 1) % n];
      const v1x = prev.x - cur.x;
      const v1y = prev.y - cur.y;
      const v2x = next.x - cur.x;
      const v2y = next.y - cur.y;
      const dot = v1x * v2x + v1y * v2y;
      const mag1 = Math.hypot(v1x, v1y) || 1;
      const mag2 = Math.hypot(v2x, v2y) || 1;
      let cos = dot / (mag1 * mag2);
      cos = Math.max(-1, Math.min(1, cos));
      const angleRad = Math.acos(cos);
      const angleDeg = (angleRad * 180) / Math.PI;
      angles.push(angleDeg);
    }
    return angles;
  }
}
