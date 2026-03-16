import "./style.css";
import { SelectionManager } from "./ui-utils.js";
import { EvaluationManager } from "./evaluation-manager.js";

export interface Point {
  x: number;
  y: number;
}

export interface DetectedShape {
  type: "circle" | "triangle" | "rectangle" | "pentagon" | "star";
  confidence: number;
  boundingBox: {
    x: number;
    y: number;
    width: number;
    height: number;
  };
  center: Point;
  area: number;
}

export interface DetectionResult {
  shapes: DetectedShape[];
  processingTime: number;
  imageWidth: number;
  imageHeight: number;
}

export class ShapeDetector {
  private canvas: HTMLCanvasElement;
  private ctx: CanvasRenderingContext2D;

  constructor(canvas: HTMLCanvasElement) {
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d")!;
  }

  /**
   * Improved shape detection pipeline using convex hull and topology
   * 
   * Pipeline:
   * 1. RGB to Grayscale
   * 2. Gaussian blur with edge clamping (fixes border pixel bug)
   * 3. Global threshold (128 default, tunable for antialiased images)
   * 4. Morphological closing with 8-connectivity (robust gap filling)
   * 5. Connected components with 8-connectivity (avoids diagonal fragmentation)
   * 6. Convex hull of boundary pixels (Monotone Chain algorithm)
   * 7. Solidity = area / convexHullArea (true definition)
   * 8. Classification by hull vertex count + solidity + circularity
   * 9. Confidence scoring
   */
  async detectShapes(imageData: ImageData): Promise<DetectionResult> {
    const startTime = performance.now();
    const detectedShapes: DetectedShape[] = [];

    // Step 1: grayscale
    const grayscale = this.convertToGrayscale(imageData);

    // Step 2: gaussian blur (with edge clamping to fix border pixel bug)
    const blurred = this.applyGaussianBlur(grayscale, imageData.width, imageData.height);

    // Step 3: global threshold (tunable constant for different image types)
    const THRESH = 128; // good default for anti-aliased black-on-white shapes
    const binary = this.applyGlobalThreshold(blurred, THRESH);

    // Step 3b: morphological closing with 8-connectivity
    const closed = this.applyMorphologicalClosing(binary, imageData.width, imageData.height);

    // Step 4: label connected components (8-connected for robustness)
    const components = this.labelConnectedComponents(closed, imageData.width, imageData.height);

    console.log(`[DEBUG] Found ${components.size} components`);

    // Process each component
    for (const [label, pixels] of components.entries()) {
      if (label === 0 || pixels.length < 80) continue; // skip background and tiny noise

      console.log(`[DEBUG] Processing component ${label} with ${pixels.length} pixels`);

      // compute region geometry and boundary
      const region = this.computeRegionMetrics(pixels, imageData.width, imageData.height);

      if (region.area < 50) {
        console.log(`[DEBUG]   Shape too small: area=${region.area}`);
        continue;
      }

      // compute convex hull of boundary pixels (unordered)
      const rawHull = this.convexHull(region.boundaryPoints);

      // use bounding-box-based epsilon (stable) — scale with shape size, not raw hull perimeter
      const maxDim = Math.max(region.boundingBox.width, region.boundingBox.height);
      const eps = Math.max(1, Math.round(maxDim * 0.015)); // 1.5% of max dimension, more conservative

      // simplify polygon to collapse edge points into true corners
      let hull = this.simplifyPolygon(rawHull, eps);

      // remove vertices that are extremely close to each other (noise)
      hull = this.deduplicateCloseVertices(hull, Math.max(1, Math.round(maxDim * 0.005)));

      // remove near-collinear vertices (internal angle ≈ 180°) using a conservative angle threshold
      hull = this.mergeCollinearVertices(hull, 12); // 12 degrees: more aggressive collinear removal for pentagon robustness

      // recompute measurements on the cleaned hull
      const hullArea = this.polygonArea(hull);
      const hullPerimeter = this.polygonPerimeter(hull);
      let hullVertexCount = hull.length;

      console.log(`[DEBUG]   Raw hull vertices: ${rawHull.length}, Simplified hull vertices: ${hullVertexCount}, eps:${eps}`);
      console.log(`[DEBUG] rawHull(${rawHull.length}):`, rawHull.map(p => `${Math.round(p.x)},${Math.round(p.y)}`).slice(0, 20));
      console.log(`[DEBUG] simplifiedHull(${hull.length}):`, hull.map(p => `${Math.round(p.x)},${Math.round(p.y)}`));

      // solidity: area / convexHullArea (true definition)
      const solidity = hullArea > 0 ? region.area / hullArea : 1;

      // circularity = 4πA / P^2 (use hull perimeter)
      const perimeter = hullPerimeter || region.perimeter || 1;
      const circularity = (4 * Math.PI * region.area) / (perimeter * perimeter);

      console.log(`[DEBUG]   Boundary: ${region.boundaryPoints.length} pixels, Hull: ${hullVertexCount} vertices`);
      console.log(`[DEBUG]   Area: ${region.area}, Circularity: ${circularity.toFixed(3)}, Solidity: ${solidity.toFixed(3)}`);

      // compute internal angles at hull vertices (simple validation)
      const angles = this.computePolygonInternalAngles(hull);
      console.log(`[DEBUG] angles:`, angles.map(a => a.toFixed(1)));

      // angle-based safety: if hull reports 4 vertices but has 2+ nearly-straight angles, treat as degenerate triangle
      if (hullVertexCount === 4 && angles.length === 4) {
        const largeAngles = angles.filter(a => a > 150).length;
        
        // if two+ angles are almost straight lines → degenerate quad from curved triangle
        if (largeAngles >= 2) {
          console.log(`[DEBUG] Degenerate quad → triangle (${largeAngles} angles > 150°)`);
          hullVertexCount = 3;
        }
      }

      console.log(`[DEBUG] FINAL VERTEX COUNT: ${hullVertexCount}`);
      console.log(
        "[DEBUG] Classification metrics - Vertices:", hullVertexCount,
        "Solidity:", solidity.toFixed(3),
        "Circularity:", circularity.toFixed(3)
      );

      let type: DetectedShape["type"] = "star";

      // ⭐ Detect concave shapes FIRST (stars, etc.) using solidity
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
        // Only classify as rectangle if 3+ angles are near 90° (true right angles)
        const rectAngles = angles.filter(a => Math.abs(a - 90) < 20).length;
        
        if (rectAngles >= 3) {
          type = "rectangle";
          console.log(`[DEBUG]   → RECTANGLE (${hullVertexCount} vertices, ${rectAngles} right angles)`);
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

      // confidence scoring (mix of metrics & vertex match)
      let confidence = 0.85;
      if (type === "circle") confidence = Math.min(0.97, 0.4 + circularity);
      if (type === "triangle" && hullVertexCount === 3) confidence = 0.92;
      if (type === "rectangle" && hullVertexCount === 4) confidence = 0.95;
      if (type === "pentagon" && hullVertexCount === 5) confidence = 0.92;
      if (type === "star") confidence = Math.max(0.65, 0.9 - solidity);

      confidence = Math.max(0.5, Math.min(0.97, confidence));

      detectedShapes.push({
        type,
        confidence,
        boundingBox: region.boundingBox,
        center: region.center,
        area: region.area,
      });
    }

    const processingTime = performance.now() - startTime;
    return {
      shapes: detectedShapes,
      processingTime,
      imageWidth: imageData.width,
      imageHeight: imageData.height,
    };
  }

  // ---------- Image processing helpers ----------

  private convertToGrayscale(imageData: ImageData): Uint8ClampedArray {
    const d = imageData.data;
    const out = new Uint8ClampedArray(imageData.width * imageData.height);
    for (let i = 0, j = 0; i < d.length; i += 4, j++) {
      out[j] = Math.round(0.299 * d[i] + 0.587 * d[i + 1] + 0.114 * d[i + 2]);
    }
    return out;
  }

  private applyGaussianBlur(gray: Uint8ClampedArray, w: number, h: number): Uint8ClampedArray {
    const out = new Uint8ClampedArray(gray.length);
    const kernel = [1, 2, 1, 2, 4, 2, 1, 2, 1];
    const ksum = 16;

    // convolution with clamped-edge behavior (replicate border pixels)
    // This fixes the border-zero bug in the original implementation
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        let sum = 0;
        let k = 0;
        for (let ky = -1; ky <= 1; ky++) {
          for (let kx = -1; kx <= 1; kx++) {
            const sx = Math.max(0, Math.min(w - 1, x + kx));
            const sy = Math.max(0, Math.min(h - 1, y + ky));
            sum += gray[sy * w + sx] * kernel[k++];
          }
        }
        out[y * w + x] = Math.round(sum / ksum);
      }
    }
    return out;
  }

  private applyGlobalThreshold(gray: Uint8ClampedArray, threshold = 128): Uint8ClampedArray {
    const out = new Uint8ClampedArray(gray.length);
    for (let i = 0; i < gray.length; i++) {
      out[i] = gray[i] < threshold ? 255 : 0;
    }
    return out;
  }

  private applyMorphologicalClosing(bin: Uint8ClampedArray, w: number, h: number): Uint8ClampedArray {
    // dilation then erosion using 8-connectivity for robustness
    const dilated = new Uint8ClampedArray(bin.length);
    
    // dilation: if any 8-neighbor is foreground, set foreground
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        const idx = y * w + x;
        if (bin[idx] > 0) {
          dilated[idx] = 255;
          continue;
        }
        let any = false;
        for (let ny = Math.max(0, y - 1); ny <= Math.min(h - 1, y + 1) && !any; ny++) {
          for (let nx = Math.max(0, x - 1); nx <= Math.min(w - 1, x + 1); nx++) {
            if (bin[ny * w + nx] > 0) {
              any = true;
              break;
            }
          }
        }
        dilated[idx] = any ? 255 : 0;
      }
    }

    // erosion: only keep foreground if all 8-neighbors are foreground
    const eroded = new Uint8ClampedArray(bin.length);
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        const idx = y * w + x;
        if (dilated[idx] === 0) {
          eroded[idx] = 0;
          continue;
        }
        let all = true;
        for (let ny = Math.max(0, y - 1); ny <= Math.min(h - 1, y + 1) && all; ny++) {
          for (let nx = Math.max(0, x - 1); nx <= Math.min(w - 1, x + 1); nx++) {
            if (dilated[ny * w + nx] === 0) {
              all = false;
              break;
            }
          }
        }
        eroded[idx] = all ? 255 : 0;
      }
    }
    return eroded;
  }

  private labelConnectedComponents(bin: Uint8ClampedArray, w: number, h: number): Map<number, Array<{ x: number; y: number }>> {
    const visited = new Uint8Array(bin.length);
    const components = new Map<number, Array<{ x: number; y: number }>>();
    let label = 1;

    for (let i = 0; i < bin.length; i++) {
      if (visited[i] || bin[i] === 0) continue;
      
      // BFS with 8-connectivity
      const q: number[] = [i];
      visited[i] = 1;
      const arr: Array<{ x: number; y: number }> = [];
      
      while (q.length) {
        const idx = q.shift()!;
        const x = idx % w;
        const y = Math.floor(idx / w);
        arr.push({ x, y });

        // 8 neighbors
        for (let ny = Math.max(0, y - 1); ny <= Math.min(h - 1, y + 1); ny++) {
          for (let nx = Math.max(0, x - 1); nx <= Math.min(w - 1, x + 1); nx++) {
            const nidx = ny * w + nx;
            if (!visited[nidx] && bin[nidx] > 0) {
              visited[nidx] = 1;
              q.push(nidx);
            }
          }
        }
      }
      components.set(label++, arr);
    }
    return components;
  }

  // ---------- Geometry & metrics ----------

  private computeRegionMetrics(regionPixels: Array<{ x: number; y: number }>, w: number, h: number) {
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

    // boundary detection (unordered list of boundary pixels)
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

    // perimeter: count boundary pixels with correction for diagonals
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

  // Monotone Chain convex hull algorithm
  private convexHull(points: Point[]): Point[] {
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

  private polygonArea(poly: Point[]): number {
    if (!poly || poly.length < 3) return 0;
    let area = 0;
    for (let i = 0; i < poly.length; i++) {
      const a = poly[i];
      const b = poly[(i + 1) % poly.length];
      area += a.x * b.y - a.y * b.x;
    }
    return Math.abs(area) / 2;
  }

  private polygonPerimeter(poly: Point[]): number {
    if (!poly || poly.length < 2) return 0;
    let p = 0;
    for (let i = 0; i < poly.length; i++) {
      const a = poly[i];
      const b = poly[(i + 1) % poly.length];
      p += Math.hypot(b.x - a.x, b.y - a.y);
    }
    return p;
  }

  // Ramer-Douglas-Peucker polygon simplification for CLOSED polygons
  private simplifyPolygon(points: Point[], epsilon = 4): Point[] {
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

  // remove vertices that are closer than `tol` pixels (keep first, drop subsequent close ones)
  private deduplicateCloseVertices(poly: Point[], tol = 2): Point[] {
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

  // merge/remove vertices where the internal angle is extremely flat (near 180°).
  // angleThresholdDeg: if internal angle > (180 - angleThresholdDeg) => treat as collinear and remove the vertex.
  private mergeCollinearVertices(poly: Point[], angleThresholdDeg = 20): Point[] {
    if (!poly || poly.length < 3) return poly.slice();

    // helper to compute internal angle at index i (degrees)
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

    // iterate and remove near-collinear vertices (single pass; repeat if you want stronger reduction)
    const out: Point[] = [];
    const n = poly.length;
    for (let i = 0; i < n; i++) {
      const ang = internalAngle(poly, i);
      // if angle is very flat (close to 180 deg) -> skip this vertex
      if (ang > (180 - angleThresholdDeg)) {
        // skip (merge prev & next across this point)
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

  // sample radial-signature and count peaks
  private computePolygonInternalAngles(poly: Point[]): number[] {
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

  loadImage(file: File): Promise<ImageData> {
    return new Promise((resolve, reject) => {
      const img = new Image();
      img.onload = () => {
        this.canvas.width = img.width;
        this.canvas.height = img.height;
        this.ctx.drawImage(img, 0, 0);
        const imageData = this.ctx.getImageData(0, 0, img.width, img.height);
        resolve(imageData);
      };
      img.onerror = reject;
      img.src = URL.createObjectURL(file);
    });
  }
}

class ShapeDetectionApp {
  private detector: ShapeDetector;
  private imageInput: HTMLInputElement;
  private testImagesDiv: HTMLDivElement;
  private evaluateButton: HTMLButtonElement;
  private selectionManager: SelectionManager;
  private evaluationManager: EvaluationManager;
  private resultsModal: HTMLElement;
  private modalContent: HTMLElement;
  private modalClose: HTMLElement;
  private currentImageData: ImageData | null = null;

  constructor() {
    const canvas = document.getElementById(
      "originalCanvas"
    ) as HTMLCanvasElement;
    this.detector = new ShapeDetector(canvas);

    this.imageInput = document.getElementById("imageInput") as HTMLInputElement;
    this.testImagesDiv = document.getElementById(
      "testImages"
    ) as HTMLDivElement;
    this.evaluateButton = document.getElementById(
      "evaluateButton"
    ) as HTMLButtonElement;
    this.resultsModal = document.getElementById("resultsModal") as HTMLElement;
    this.modalContent = document.getElementById("modalContent") as HTMLElement;
    this.modalClose = document.getElementById("modalClose") as HTMLElement;

    this.selectionManager = new SelectionManager();
    this.evaluationManager = new EvaluationManager(
      this.detector,
      this.evaluateButton,
      {
        open: (content: string) => this.openResultsModal(content),
        close: () => this.closeResultsModal(),
      }
    );

    this.setupEventListeners();
    this.setupModalControls();
    this.loadTestImages().catch(console.error);
    
    // Disable button on initial load (0 images selected)
    this.evaluateButton.disabled = true;
  }

  private setupModalControls(): void {
    this.modalClose.addEventListener("click", () => this.closeResultsModal());
    this.resultsModal.addEventListener("click", (e) => {
      if (e.target === this.resultsModal.querySelector(".modal-backdrop")) {
        this.closeResultsModal();
      }
    });
    document.addEventListener("keydown", (e) => {
      if (
        e.key === "Escape" &&
        this.resultsModal.classList.contains("open")
      ) {
        this.closeResultsModal();
      }
    });
  }

  private openResultsModal(content: string): void {
    this.modalContent.innerHTML = content;
    this.resultsModal.classList.add("open");
    document.body.style.overflow = "hidden";
  }

  private closeResultsModal(): void {
    this.resultsModal.classList.remove("open");
    document.body.style.overflow = "";
  }

  private setupEventListeners(): void {
    this.imageInput.addEventListener("change", async (event) => {
      const file = (event.target as HTMLInputElement).files?.[0];
      if (file) {
        await this.processImage(file);
      }
    });

    this.evaluateButton.addEventListener("click", async () => {
      const selectedImages = this.selectionManager.getSelectedImages();
      await this.evaluationManager.runSelectedEvaluation(selectedImages);
    });
  }

  private async processImage(file: File): Promise<void> {
    try {
      const overlay = document.getElementById("processingOverlay");
      if (overlay) overlay.classList.remove("hidden");

      const startTime = performance.now();
      const imageData = await this.detector.loadImage(file);
      this.currentImageData = imageData;
      const results = await this.detector.detectShapes(imageData);

      // Ensure minimum loading time for UX
      const elapsed = performance.now() - startTime;
      if (elapsed < 1000) {
        await new Promise((r) => setTimeout(r, 1000 - elapsed));
      }

      if (overlay) overlay.classList.add("hidden");
      this.displayResultsInModal(results, file.name || "Uploaded image");
    } catch (error) {
      const overlay = document.getElementById("processingOverlay");
      if (overlay) overlay.classList.add("hidden");
      alert(`Error: ${error}`);
    }
  }

  private displayResultsInModal(
    results: DetectionResult,
    fileName: string
  ): void {
    const { shapes, processingTime } = results;

    let html = `
      <div class="result-card">
        <div class="result-image-section">
          <h3>Image</h3>
          <canvas id="resultCanvas" style="width: 100%; border: 1px solid var(--border); border-radius: var(--radius-lg);"></canvas>
          <p style="text-align: center; color: var(--text-secondary); margin: 0.5rem 0 0 0; font-size: 0.9rem;">${fileName}</p>
        </div>

        <div class="result-metrics-section">
          <h3>Detection Results</h3>
          
          <div class="metrics-grid">
            <div class="metric-item">
              <div class="metric-label">Processing Time</div>
              <div class="metric-value">${processingTime.toFixed(0)}ms</div>
            </div>
            <div class="metric-item">
              <div class="metric-label">Shapes Found</div>
              <div class="metric-value">${shapes.length}</div>
            </div>
          </div>
    `;

    if (shapes.length > 0) {
      html += `
        <div class="shapes-list">
          <h4>Detected Shapes</h4>
      `;
      shapes.forEach((shape) => {
        html += `
          <div class="shape-item">
            <div class="shape-type">${shape.type.charAt(0).toUpperCase() + shape.type.slice(1)}</div>
            <div class="shape-confidence">Confidence: ${(shape.confidence * 100).toFixed(0)}% | Area: ${shape.area.toFixed(0)}px²</div>
          </div>
        `;
      });
      html += `
        </div>
      `;
    } else {
      html += `
        <div style="padding: 1.5rem; text-align: center; color: var(--text-secondary);">
          No shapes detected in this image
        </div>
      `;
    }

    html += `
        </div>
      </div>
    `;

    this.openResultsModal(html);

    // Draw canvas after modal is open
    setTimeout(() => {
      const canvas = document.getElementById("resultCanvas") as HTMLCanvasElement;
      if (canvas && this.currentImageData) {
        canvas.width = results.imageWidth;
        canvas.height = results.imageHeight;
        const ctx = canvas.getContext("2d");
        if (ctx) {
          ctx.putImageData(this.currentImageData, 0, 0);
        }
      }
    }, 0);
  }

  private async loadTestImages(): Promise<void> {
    try {
      const module = await import("./test-images-data.js");
      const testImages = module.testImages;
      const imageNames = module.getAllTestImageNames();

      // Helper function to categorize images
      const categorizeImage = (name: string): string => {
        const lowerName = name.toLowerCase();
        if (
          lowerName.includes("simple") ||
          lowerName.includes("basic") ||
          lowerName.includes("pentagon") ||
          lowerName.includes("rectangle") ||
          lowerName.includes("star") ||
          lowerName.includes("triangle") ||
          lowerName.includes("circle")
        ) {
          return "basic";
        } else if (
          lowerName.includes("complex") ||
          lowerName.includes("mixed")
        ) {
          return "complex";
        } else {
          return "edge";
        }
      };

      // Helper function to properly capitalize image names
      const capitalizeImageName = (name: string): string => {
        return name
          .replace(/[_-]/g, " ")
          .replace(/\.(svg|png)$/i, "")
          .split(" ")
          .map((word) => word.charAt(0).toUpperCase() + word.slice(1).toLowerCase())
          .join(" ");
      };

      const categories = {
        basic: [] as string[],
        complex: [] as string[],
        edge: [] as string[],
      };

      // Categorize all images
      imageNames.forEach((imageName) => {
        const category = categorizeImage(imageName);
        categories[category as keyof typeof categories].push(imageName);
      });

      let html = "";

      // Basic Shapes Section
      if (categories.basic.length > 0) {
        html += `<div style="grid-column: 1 / -1; padding-top: 0.2rem;"><h3 style="margin: 0 0 0.25rem 0; color: #374151; font-size: 1.8rem; font-weight: 700; letter-spacing: 0.3px; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;">Basic Shapes</h3></div>`;
        categories.basic.forEach((imageName) => {
          const dataUrl = testImages[imageName as keyof typeof testImages];
          const displayName = capitalizeImageName(imageName);
          html += `
            <div style="display: flex; flex-direction: column; align-items: center; gap: 0.75rem;">
              <div class="test-image-item" data-image="${imageName}" 
                   onclick="loadTestImage('${imageName}', '${dataUrl}')" 
                   oncontextmenu="toggleImageSelection(event, '${imageName}')">
                <img src="${dataUrl}" alt="${imageName}">
              </div>
              <span style="font-size: 1rem; color: #374151; font-weight: 500; text-align: center; max-width: 200px;">${displayName}</span>
            </div>
          `;
        });
      }

      // Complex Shapes Section
      if (categories.complex.length > 0) {
        html += `<div style="grid-column: 1 / -1; padding-top: 0.2rem;"><h3 style="margin: 0 0 0.25rem 0; color: #374151; font-size: 1.8rem; font-weight: 700; letter-spacing: 0.3px; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;">Complex Scenes</h3></div>`;
        categories.complex.forEach((imageName) => {
          const dataUrl = testImages[imageName as keyof typeof testImages];
          const displayName = capitalizeImageName(imageName);
          html += `
            <div style="display: flex; flex-direction: column; align-items: center; gap: 0.75rem;">
              <div class="test-image-item" data-image="${imageName}" 
                   onclick="loadTestImage('${imageName}', '${dataUrl}')" 
                   oncontextmenu="toggleImageSelection(event, '${imageName}')">
                <img src="${dataUrl}" alt="${imageName}">
              </div>
              <span style="font-size: 1rem; color: #374151; font-weight: 500; text-align: center; max-width: 200px;">${displayName}</span>
            </div>
          `;
        });
      }

      // Edge Cases Section
      if (categories.edge.length > 0) {
        html += `<div style="grid-column: 1 / -1; padding-top: 0.2rem;"><h3 style="margin: 0 0 0.25rem 0; color: #374151; font-size: 1.8rem; font-weight: 700; letter-spacing: 0.3px; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;">Edge Cases</h3></div>`;
        categories.edge.forEach((imageName) => {
          const dataUrl = testImages[imageName as keyof typeof testImages];
          const displayName = capitalizeImageName(imageName);
          html += `
            <div style="display: flex; flex-direction: column; align-items: center; gap: 0.75rem;">
              <div class="test-image-item" data-image="${imageName}" 
                   onclick="loadTestImage('${imageName}', '${dataUrl}')" 
                   oncontextmenu="toggleImageSelection(event, '${imageName}')">
                <img src="${dataUrl}" alt="${imageName}">
              </div>
              <span style="font-size: 1rem; color: #374151; font-weight: 500; text-align: center; max-width: 200px;">${displayName}</span>
            </div>
          `;
        });
      }

      // Upload Image at the end
      html += `<div style="grid-column: 1 / -1; padding-top: 0.2rem;"><h3 style="margin: 0 0 0.25rem 0; color: #374151; font-size: 1.8rem; font-weight: 700; letter-spacing: 0.3px; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;">Custom Image</h3></div>`;
      html += `
        <div class="test-image-item upload-item" onclick="triggerFileUpload()">
          <div class="upload-icon">📁</div>
          <div class="upload-text">Upload Image</div>
          <div class="upload-subtext">Click to select file</div>
        </div>
      `;

      this.testImagesDiv.innerHTML = html;
      this.selectionManager.setupSelectionControls();

      (window as any).loadTestImage = async (name: string, dataUrl: string) => {
        try {
          const overlay = document.getElementById("processingOverlay");
          if (overlay) overlay.classList.remove("hidden");

          const response = await fetch(dataUrl);
          const blob = await response.blob();
          const file = new File([blob], name, { type: "image/svg+xml" });

          const imageData = await this.detector.loadImage(file);
          this.currentImageData = imageData;
          const results = await this.detector.detectShapes(imageData);

          // Ensure minimum loading time
          await new Promise((r) => setTimeout(r, 500));
          if (overlay) overlay.classList.add("hidden");

          const displayName = name
            .replace(/[_-]/g, " ")
            .replace(/\.(svg|png)$/i, "");
          this.displayResultsInModal(results, displayName);

          console.log(`Loaded test image: ${name}`);
        } catch (error) {
          const overlay = document.getElementById("processingOverlay");
          if (overlay) overlay.classList.add("hidden");
          console.error("Error loading test image:", error);
          alert("Error loading image.");
        }
      };

      (window as any).toggleImageSelection = (
        event: MouseEvent,
        imageName: string
      ) => {
        event.preventDefault();
        this.selectionManager.toggleImageSelection(imageName);
        this.updateSelectionCount();
      };

      (window as any).triggerFileUpload = () => {
        this.imageInput.click();
      };
    } catch (error) {
      this.testImagesDiv.innerHTML = `
        <p>Test images not available. Run 'node convert-svg-to-png.js' to generate test image data.</p>
      `;
    }
  }

  private updateSelectionCount(): void {
    const count = this.selectionManager.getSelectedImages().length;
    const countEl = document.querySelector(".selection-count");
    if (countEl) {
      countEl.textContent = `${count} image${count !== 1 ? "s" : ""} selected`;
    }
    // Disable button if no images are selected
    this.evaluateButton.disabled = count === 0;
  }
}

document.addEventListener("DOMContentLoaded", () => {
  new ShapeDetectionApp();
});
