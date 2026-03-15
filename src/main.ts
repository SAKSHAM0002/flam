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
   * MAIN ALGORITHM TO IMPLEMENT
   * Method for detecting shapes in an image
   * @param imageData - ImageData from canvas
   * @returns Promise<DetectionResult> - Detection results
   */
  async detectShapes(imageData: ImageData): Promise<DetectionResult> {
    const startTime = performance.now();
    const shapes: DetectedShape[] = [];

    // Step 1: Preprocess image (convert to grayscale and apply threshold)
    const processed = this.preprocessImage(imageData);

    // Step 2: Find contours/blobs in the image
    const contours = this.findContours(processed);

    // Step 3: Analyze each contour and classify shape
    for (const contour of contours) {
      if (contour.length < 4) continue; // Need at least 4 points for a shape

      const boundingBox = this.getBoundingBox(contour);
      const area = this.calculateArea(contour);
      const center = this.getCenter(boundingBox);

      // Skip very small shapes (noise)
      if (area < 50) continue;

      // Classify the shape based on contour properties
      const shapeType = this.classifyShape(contour, area, boundingBox);

      shapes.push({
        type: shapeType,
        confidence: this.calculateConfidence(contour, shapeType),
        boundingBox,
        center,
        area,
      });
    }

    const processingTime = performance.now() - startTime;

    return {
      shapes,
      processingTime,
      imageWidth: imageData.width,
      imageHeight: imageData.height,
    };
  }

  private preprocessImage(imageData: ImageData): Uint8ClampedArray {
    const data = imageData.data;
    const processed = new Uint8ClampedArray(imageData.width * imageData.height);

    // Convert to grayscale using luminosity method
    for (let i = 0; i < data.length; i += 4) {
      const r = data[i];
      const g = data[i + 1];
      const b = data[i + 2];
      const gray = 0.299 * r + 0.587 * g + 0.114 * b;
      processed[i / 4] = gray;
    }

    // Apply binary threshold
    const threshold = 128;
    for (let i = 0; i < processed.length; i++) {
      processed[i] = processed[i] > threshold ? 255 : 0;
    }

    return processed;
  }

  private findContours(processed: Uint8ClampedArray): Point[][] {
    const visited = new Set<number>();
    const contours: Point[][] = [];
    const width = this.canvas.width;
    const height = this.canvas.height;

    for (let i = 0; i < processed.length; i++) {
      const x = i % width;
      const y = Math.floor(i / width);

      if (processed[i] > 128 && !visited.has(i)) {
        // Found a white pixel, start flood fill
        const contour = this.floodFill(processed, x, y, visited, width, height);
        if (contour.length > 3) {
          contours.push(contour);
        }
      }
    }

    return contours;
  }

  private floodFill(
    processed: Uint8ClampedArray,
    startX: number,
    startY: number,
    visited: Set<number>,
    width: number,
    height: number
  ): Point[] {
    const contour: Point[] = [];
    const queue = [[startX, startY]];
    const width_mod = width;

    while (queue.length > 0) {
      const [x, y] = queue.shift()!;
      const idx = y * width_mod + x;

      if (
        x < 0 ||
        x >= width_mod ||
        y < 0 ||
        y >= height ||
        visited.has(idx) ||
        processed[idx] < 128
      ) {
        continue;
      }

      visited.add(idx);
      contour.push({ x, y });

      // Add neighbors to queue (4-connectivity)
      queue.push([x + 1, y]);
      queue.push([x - 1, y]);
      queue.push([x, y + 1]);
      queue.push([x, y - 1]);
    }

    return contour;
  }

  private getBoundingBox(contour: Point[]): {
    x: number;
    y: number;
    width: number;
    height: number;
  } {
    let minX = Infinity,
      minY = Infinity,
      maxX = -Infinity,
      maxY = -Infinity;

    for (const point of contour) {
      minX = Math.min(minX, point.x);
      maxX = Math.max(maxX, point.x);
      minY = Math.min(minY, point.y);
      maxY = Math.max(maxY, point.y);
    }

    return {
      x: minX,
      y: minY,
      width: maxX - minX,
      height: maxY - minY,
    };
  }

  private getCenter(bbox: {
    x: number;
    y: number;
    width: number;
    height: number;
  }): Point {
    return {
      x: bbox.x + bbox.width / 2,
      y: bbox.y + bbox.height / 2,
    };
  }

  private calculateArea(contour: Point[]): number {
    // Shoelace formula for polygon area
    let area = 0;
    for (let i = 0; i < contour.length; i++) {
      const p1 = contour[i];
      const p2 = contour[(i + 1) % contour.length];
      area += p1.x * p2.y - p2.x * p1.y;
    }
    return Math.abs(area) / 2;
  }

  private classifyShape(
    contour: Point[],
    area: number,
    bbox: { x: number; y: number; width: number; height: number }
  ): "circle" | "triangle" | "rectangle" | "pentagon" | "star" {
    const aspectRatio = bbox.width / bbox.height;

    // Approximate the contour to simplify it
    const simplified = this.simplifyContour(contour);
    const vertices = simplified.length;

    // Check circularity using area and perimeter
    const perimeter = this.calculatePerimeter(contour);
    const circularity = (4 * Math.PI * area) / (perimeter * perimeter);

    // If very circular (circularity > 0.7), it's a circle
    if (circularity > 0.7) {
      return "circle";
    }

    // Count vertices to classify polygon
    if (vertices <= 4) {
      // Check if it's more square-like or rectangular
      if (Math.abs(aspectRatio - 1) < 0.3) {
        return "rectangle"; // More like square
      }
      return "rectangle";
    } else if (vertices === 5) {
      return "pentagon";
    } else if (vertices > 6 && vertices <= 8) {
      return "star";
    } else if (vertices === 3) {
      return "triangle";
    }

    // Default classification based on vertices
    if (vertices < 4) return "triangle";
    if (vertices < 6) return "pentagon";
    return "star";
  }

  private simplifyContour(contour: Point[], epsilon: number = 2): Point[] {
    // Simple contour simplification using distance threshold
    if (contour.length < 3) return contour;

    const simplified: Point[] = [contour[0]];

    for (let i = 1; i < contour.length - 1; i++) {
      const prev = simplified[simplified.length - 1];
      const curr = contour[i];
      const next = contour[i + 1];

      const dist = this.pointToLineDistance(curr, prev, next);
      if (dist > epsilon) {
        simplified.push(curr);
      }
    }

    simplified.push(contour[contour.length - 1]);
    return simplified;
  }

  private pointToLineDistance(
    point: Point,
    lineStart: Point,
    lineEnd: Point
  ): number {
    const numerator = Math.abs(
      (lineEnd.y - lineStart.y) * point.x -
        (lineEnd.x - lineStart.x) * point.y +
        lineEnd.x * lineStart.y -
        lineEnd.y * lineStart.x
    );
    const denominator = Math.sqrt(
      Math.pow(lineEnd.y - lineStart.y, 2) +
        Math.pow(lineEnd.x - lineStart.x, 2)
    );
    return numerator / denominator;
  }

  private calculatePerimeter(contour: Point[]): number {
    let perimeter = 0;
    for (let i = 0; i < contour.length; i++) {
      const p1 = contour[i];
      const p2 = contour[(i + 1) % contour.length];
      const dist = Math.sqrt(
        Math.pow(p2.x - p1.x, 2) + Math.pow(p2.y - p1.y, 2)
      );
      perimeter += dist;
    }
    return perimeter;
  }

  private calculateConfidence(
    contour: Point[],
    shapeType: string
  ): number {
    // Base confidence on shape regularity and area
    const bbox = this.getBoundingBox(contour);
    const area = this.calculateArea(contour);
    const perim = this.calculatePerimeter(contour);

    // Calculate bounding box area
    const bboxArea = bbox.width * bbox.height;

    // Confidence based on how much of bbox is filled
    const fillRatio = area / bboxArea;

    // Start with base confidence
    let confidence = Math.min(fillRatio, 1.0);

    // Boost for circles (high circularity)
    if (shapeType === "circle") {
      const circularity = (4 * Math.PI * area) / (perim * perim);
      confidence = Math.max(confidence, circularity);
    }

    // Ensure minimum confidence
    return Math.max(0.5, Math.min(0.95, confidence));
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
