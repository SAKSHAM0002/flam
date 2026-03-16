import "./style.css";
import { SelectionManager } from "./ui-utils.js";
import { EvaluationManager } from "./evaluation-manager.js";
import { ImageProcessor } from "./detector/ImageProcessor.js";
import { ComponentLabeler } from "./detector/ComponentLabeler.js";
import { GeometryUtils } from "./detector/GeometryUtils.js";
import { ConvexHullComputer } from "./detector/ConvexHullComputer.js";
import { PolygonSimplification } from "./detector/PolygonSimplification.js";
import { ShapeClassifier } from "./detector/ShapeClassifier.js";
import type { Point, DetectedShape, DetectionResult } from "./detector/types.js";

export type { Point, DetectedShape, DetectionResult };

export class ShapeDetector {
  private canvas: HTMLCanvasElement;
  private ctx: CanvasRenderingContext2D;

  constructor(canvas: HTMLCanvasElement) {
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d")!;
  }

  async detectShapes(imageData: ImageData): Promise<DetectionResult> {
    const startTime = performance.now();
    const detectedShapes: DetectedShape[] = [];

    // Step 1: Preprocess using modular ImageProcessor
    const grayscale = ImageProcessor.convertToGrayscale(imageData);
    const blurred = ImageProcessor.applyGaussianBlur(grayscale, imageData.width, imageData.height);
    const binary = ImageProcessor.applyGlobalThreshold(blurred, 128);
    const closed = ImageProcessor.applyMorphologicalClosing(binary, imageData.width, imageData.height);

    // Step 2: Label components using modular ComponentLabeler
    const components = ComponentLabeler.labelConnectedComponents(closed, imageData.width, imageData.height);
    console.log(`[DEBUG] Found ${components.size} components`);

    // Process each component
    for (const [label, pixels] of components.entries()) {
      if (label === 0 || pixels.length < 80) continue;

      console.log(`[DEBUG] Processing component ${label} with ${pixels.length} pixels`);

      // Compute region metrics using modular GeometryUtils
      const region = GeometryUtils.computeRegionMetrics(pixels, imageData.width, imageData.height);

      if (region.area < 50) {
        console.log(`[DEBUG]   Shape too small: area=${region.area}`);
        continue;
      }

      // Compute convex hull using modular ConvexHullComputer
      const rawHull = ConvexHullComputer.computeHull(region.boundaryPoints);

      const maxDim = Math.max(region.boundingBox.width, region.boundingBox.height);
      const eps = Math.max(1, Math.round(maxDim * 0.015));

      // Simplify polygon using modular PolygonSimplification
      let hull = PolygonSimplification.simplifyPolygon(rawHull, eps);
      hull = PolygonSimplification.deduplicateCloseVertices(hull, Math.max(1, Math.round(maxDim * 0.005)));
      hull = PolygonSimplification.mergeCollinearVertices(hull, 12);

      // Recompute measurements using modular GeometryUtils
      const hullArea = GeometryUtils.polygonArea(hull);
      const hullPerimeter = GeometryUtils.polygonPerimeter(hull);
      let hullVertexCount = hull.length;

      console.log(`[DEBUG]   Raw hull vertices: ${rawHull.length}, Simplified hull vertices: ${hullVertexCount}, eps:${eps}`);
      console.log(`[DEBUG] rawHull(${rawHull.length}):`, rawHull.map(p => `${Math.round(p.x)},${Math.round(p.y)}`).slice(0, 20));
      console.log(`[DEBUG] simplifiedHull(${hull.length}):`, hull.map(p => `${Math.round(p.x)},${Math.round(p.y)}`));

      const solidity = hullArea > 0 ? region.area / hullArea : 1;
      const perimeter = hullPerimeter || region.perimeter || 1;
      const circularity = (4 * Math.PI * region.area) / (perimeter * perimeter);

      console.log(`[DEBUG]   Boundary: ${region.boundaryPoints.length} pixels, Hull: ${hullVertexCount} vertices`);
      console.log(`[DEBUG]   Area: ${region.area}, Circularity: ${circularity.toFixed(3)}, Solidity: ${solidity.toFixed(3)}`);

      // Compute angles using modular GeometryUtils
      const angles = GeometryUtils.computePolygonInternalAngles(hull);
      console.log(`[DEBUG] angles:`, angles.map(a => a.toFixed(1)));

      // Correct degenerate quads using modular ShapeClassifier
      hullVertexCount = ShapeClassifier.correctDegenerateQuad(hullVertexCount, angles);

      console.log(`[DEBUG] FINAL VERTEX COUNT: ${hullVertexCount}`);
      console.log(
        "[DEBUG] Classification metrics - Vertices:", hullVertexCount,
        "Solidity:", solidity.toFixed(3),
        "Circularity:", circularity.toFixed(3)
      );

      // Classify shape using modular ShapeClassifier
      const { type, confidence } = ShapeClassifier.classifyShape(
        hullVertexCount,
        angles,
        solidity,
        circularity
      );

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
          <div class="upload-icon">[Upload]</div>
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
