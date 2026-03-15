

import type { ShapeDetector } from "./main.js";

interface IResultsModal {
  open(content: string): void;
  close(): void;
}

export class EvaluationManager {
  private detector: ShapeDetector;
  private evaluateButton: HTMLButtonElement;
  private resultsModal: IResultsModal;

  constructor(
    detector: ShapeDetector,
    evaluateButton: HTMLButtonElement,
    resultsModal: IResultsModal
  ) {
    this.detector = detector;
    this.evaluateButton = evaluateButton;
    this.resultsModal = resultsModal;
  }

  async runSelectedEvaluation(selectedImages: string[]): Promise<void> {
    if (selectedImages.length === 0) {
      alert(
        "Please select at least one image for evaluation (right-click to select)"
      );
      return;
    }

    const overlay = document.getElementById("processingOverlay");
    try {
      this.evaluateButton.disabled = true;
      if (overlay) {
        const textEl = document.getElementById("processingOverlayText");
        if (textEl) textEl.textContent = "Running evaluation...";
        overlay.classList.remove("hidden");
      }

      const evaluationModule = await import("./evaluation.js");
      const results = await evaluationModule.runSelectedEvaluation(
        this.detector,
        selectedImages
      );

      // Build and display evaluation results
      const resultsHTML = this.buildEvaluationHTML(results);
      this.resultsModal.open(resultsHTML);

      console.log("Selected Evaluation Results:", results);
    } catch (error) {
      alert(`Error during evaluation: ${error}`);
      console.error("Evaluation error:", error);
    } finally {
      this.evaluateButton.disabled = false;
      if (overlay) {
        overlay.classList.add("hidden");
      }
    }
  }

  private buildEvaluationHTML(results: any): string {
    return `
      <h2 style="margin: 0 0 2rem 0; color: var(--text-primary);">Evaluation Results</h2>
      <div class="evaluation-results-grid">
        ${results.testResults
          .map(
            (result: any) => {
              const passed = result.passed;
              const bgColor = passed ? "#d1fae5" : "#fee2e2";
              const borderColor = passed ? "#10b981" : "#ef4444";
              const textColor = passed ? "#059669" : "#dc2626";
              return `
          <div class="evaluation-card" style="border: 3px solid ${borderColor} !important; background-color: ${bgColor} !important;">
            <h4 style="color: ${textColor} !important;">${result.imageName} ${passed ? "✓ PASS" : "✗ FAIL"}</h4>
            <p><strong>Shapes Found:</strong> <span class="metric">${result.detectionResult.shapes.length}</span></p>
            <p><strong>Processing Time:</strong> <span class="metric">${result.detectionResult.processingTime.toFixed(0)}ms</span></p>
            <p><strong>F1 Score:</strong> <span class="metric">${result.evaluation.f1_score.toFixed(3)}</span></p>
            <p><strong>Precision:</strong> <span class="metric">${result.evaluation.precision.toFixed(3)}</span></p>
            <p><strong>Recall:</strong> <span class="metric">${result.evaluation.recall.toFixed(3)}</span></p>
          </div>
        `;
            }
          )
          .join("")}
      </div>
    `;
  }

  async runFullEvaluation(): Promise<void> {
    try {
      this.evaluateButton.disabled = true;

      const evaluationModule = await import("./evaluation.js");
      const results = await evaluationModule.runEvaluation(this.detector);

      const resultsHTML = this.buildEvaluationHTML(results);
      this.resultsModal.open(resultsHTML);

      console.log("Full Evaluation Results:", results);
    } catch (error) {
      alert(`Error during evaluation: ${error}`);
      console.error("Evaluation error:", error);
    } finally {
      this.evaluateButton.disabled = false;
    }
  }
}
