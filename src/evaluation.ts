import type { DetectionResult } from "./main.js";
import { ShapeDetector } from "./main.js";
import { evaluateDetection } from "./evaluation-utils.js";
import type { EvaluationMetrics } from "./evaluation-utils.js";
import { testImages, getAllTestImageNames } from "./test-images-data.js";

let groundTruthData: any = null;

async function loadGroundTruth() {
  if (!groundTruthData) {
    try {
      const response = await fetch("/ground_truth.json");
      groundTruthData = await response.json();
    } catch (error) {
      console.error("Failed to load ground truth data:", error);
      throw error;
    }
  }
  return groundTruthData;
}

export interface TestResult {
  imageName: string;
  detectionResult: DetectionResult;
  evaluation: EvaluationMetrics;
  passed: boolean;
  feedback: string[];
}

export interface OverallResults {
  totalScore: number;
  maxScore: number;
  percentage: number;
  grade: string;
  testResults: TestResult[];
  summary: {
    averagePrecision: number;
    averageRecall: number;
    averageF1: number;
    averageIoU: number;
    totalProcessingTime: number;
  };
}

export async function runSelectedEvaluation(
  detector: ShapeDetector,
  selectedImageNames: string[]
): Promise<OverallResults> {
  const groundTruth = await loadGroundTruth();
  const testResults: TestResult[] = [];

  let totalPrecision = 0;
  let totalRecall = 0;
  let totalF1 = 0;
  let totalIoU = 0;
  let totalProcessingTime = 0;
  let totalScore = 0;

  for (const imageName of selectedImageNames) {
    try {
      const dataUrl = testImages[imageName as keyof typeof testImages];
      const response = await fetch(dataUrl);
      const blob = await response.blob();
      const file = new File([blob], imageName, { type: "image/svg+xml" });

      const canvas = document.createElement("canvas");
      const tempDetector = new ShapeDetector(canvas);

      const imageData = await tempDetector.loadImage(file);
      const detectionResult = await detector.detectShapes(imageData);

      const gtShapes = groundTruth.images[imageName]?.shapes || [];

      // Evaluate results
      const evaluation = evaluateDetection(
        detectionResult.shapes,
        gtShapes,
        imageName
      );
      evaluation.processing_time = detectionResult.processingTime;

      const { passed, feedback, score } = calculateScore(
        evaluation,
        detectionResult
      );

      totalPrecision += evaluation.precision;
      totalRecall += evaluation.recall;
      totalF1 += evaluation.f1_score;
      totalIoU += evaluation.average_iou;
      totalProcessingTime += evaluation.processing_time;
      totalScore += score;

      testResults.push({
        imageName,
        detectionResult,
        evaluation,
        passed,
        feedback,
      });
    } catch (error) {
      console.error(`Error testing ${imageName}:`, error);
      testResults.push({
        imageName,
        detectionResult: {
          shapes: [],
          processingTime: 0,
          imageWidth: 0,
          imageHeight: 0,
        },
        evaluation: {
          precision: 0,
          recall: 0,
          f1_score: 0,
          average_iou: 0,
          center_point_accuracy: 0,
          area_accuracy: 0,
          confidence_calibration: 0,
          processing_time: 0,
        },
        passed: false,
        feedback: [`Error during testing: ${error}`],
      });
    }
  }

  const numTests = testResults.length;
  const maxScore = numTests * 100;
  const percentage = (totalScore / maxScore) * 100;

  const results: OverallResults = {
    totalScore: Math.round(totalScore),
    maxScore,
    percentage: Math.round(percentage * 100) / 100,
    grade: calculateGrade(percentage),
    testResults,
    summary: {
      averagePrecision: totalPrecision / numTests,
      averageRecall: totalRecall / numTests,
      averageF1: totalF1 / numTests,
      averageIoU: totalIoU / numTests,
      totalProcessingTime,
    },
  };

  return results;
}

export async function runEvaluation(
  detector: ShapeDetector
): Promise<OverallResults> {
  const groundTruth = await loadGroundTruth();
  const testResults: TestResult[] = [];
  const imageNames = getAllTestImageNames();

  let totalPrecision = 0;
  let totalRecall = 0;
  let totalF1 = 0;
  let totalIoU = 0;
  let totalProcessingTime = 0;
  let totalScore = 0;

  for (const imageName of imageNames) {
    try {
      const dataUrl = testImages[imageName as keyof typeof testImages];
      const response = await fetch(dataUrl);
      const blob = await response.blob();
      const file = new File([blob], imageName, { type: "image/svg+xml" });

      const canvas = document.createElement("canvas");
      const tempDetector = new ShapeDetector(canvas);

      const imageData = await tempDetector.loadImage(file);
      const detectionResult = await detector.detectShapes(imageData);

      const gtShapes = groundTruth.images[imageName]?.shapes || [];

      // Evaluate results
      const evaluation = evaluateDetection(
        detectionResult.shapes,
        gtShapes,
        imageName
      );
      evaluation.processing_time = detectionResult.processingTime;

      const { passed, feedback, score } = calculateScore(
        evaluation,
        detectionResult
      );

      totalPrecision += evaluation.precision;
      totalRecall += evaluation.recall;
      totalF1 += evaluation.f1_score;
      totalIoU += evaluation.average_iou;
      totalProcessingTime += evaluation.processing_time;
      totalScore += score;

      testResults.push({
        imageName,
        detectionResult,
        evaluation,
        passed,
        feedback,
      });
    } catch (error) {
      console.error(`Error testing ${imageName}:`, error);
      testResults.push({
        imageName,
        detectionResult: {
          shapes: [],
          processingTime: 0,
          imageWidth: 0,
          imageHeight: 0,
        },
        evaluation: {
          precision: 0,
          recall: 0,
          f1_score: 0,
          average_iou: 0,
          center_point_accuracy: 0,
          area_accuracy: 0,
          confidence_calibration: 0,
          processing_time: 0,
        },
        passed: false,
        feedback: [`Error during testing: ${error}`],
      });
    }
  }

  const numTests = testResults.length;
  const maxScore = numTests * 100;
  const percentage = (totalScore / maxScore) * 100;

  const results: OverallResults = {
    totalScore: Math.round(totalScore),
    maxScore,
    percentage: Math.round(percentage * 100) / 100,
    grade: calculateGrade(percentage),
    testResults,
    summary: {
      averagePrecision: totalPrecision / numTests,
      averageRecall: totalRecall / numTests,
      averageF1: totalF1 / numTests,
      averageIoU: totalIoU / numTests,
      totalProcessingTime,
    },
  };

  return results;
}

function calculateScore(
  evaluation: EvaluationMetrics,
  detection: DetectionResult
): {
  passed: boolean;
  feedback: string[];
  score: number;
} {
  const feedback: string[] = [];
  let score = 0;

  const f1 = evaluation.f1_score;
  if (f1 >= 0.9) {
    score += 40;
    feedback.push(`detection accuracy (F1: ${f1.toFixed(3)})`);
  } else if (f1 >= 0.7) {
    score += 30;
    feedback.push(`detection accuracy (F1: ${f1.toFixed(3)})`);
  } else if (f1 >= 0.5) {
    score += 20;
    feedback.push(`detection accuracy (F1: ${f1.toFixed(3)})`);
  } else {
    feedback.push(`detection accuracy (F1: ${f1.toFixed(3)})`);
  }

  const avgIoU = evaluation.average_iou;
  if (avgIoU >= 0.8) {
    score += 25;
    feedback.push(`✓ Excellent localization (IoU: ${avgIoU.toFixed(3)})`);
  } else if (avgIoU >= 0.6) {
    score += 20;
    feedback.push(`✓ Good localization (IoU: ${avgIoU.toFixed(3)})`);
  } else if (avgIoU >= 0.4) {
    score += 10;
    feedback.push(`△ Fair localization (IoU: ${avgIoU.toFixed(3)})`);
  } else {
    feedback.push(`✗ Poor localization (IoU: ${avgIoU.toFixed(3)})`);
  }

  const centerAcc = evaluation.center_point_accuracy;
  if (centerAcc <= 5) {
    score += 15;
    feedback.push(`center accuracy (${centerAcc.toFixed(1)}px error)`);
  } else if (centerAcc <= 10) {
    score += 12;
    feedback.push(`center accuracy (${centerAcc.toFixed(1)}px error)`);
  } else if (centerAcc <= 20) {
    score += 8;
    feedback.push(`center accuracy (${centerAcc.toFixed(1)}px error)`);
  } else {
    feedback.push(`center accuracy (${centerAcc.toFixed(1)}px error)`);
  }

  const areaAcc = evaluation.area_accuracy;
  if (areaAcc >= 0.9) {
    score += 10;
    feedback.push(`area calculation (${(areaAcc * 100).toFixed(1)}% accuracy)`);
  } else if (areaAcc >= 0.8) {
    score += 8;
    feedback.push(`area calculation (${(areaAcc * 100).toFixed(1)}% accuracy)`);
  } else if (areaAcc >= 0.7) {
    score += 5;
    feedback.push(`area calculation (${(areaAcc * 100).toFixed(1)}% accuracy)`);
  } else {
    feedback.push(`area calculation (${(areaAcc * 100).toFixed(1)}% accuracy)`);
  }

  const processingTime = detection.processingTime;
  if (processingTime <= 500) {
    score += 10;
    feedback.push(`performance (${processingTime.toFixed(0)}ms)`);
  } else if (processingTime <= 1000) {
    score += 8;
    feedback.push(`performance (${processingTime.toFixed(0)}ms)`);
  } else if (processingTime <= 2000) {
    score += 5;
    feedback.push(`performance (${processingTime.toFixed(0)}ms)`);
  } else {
    feedback.push(`performance (${processingTime.toFixed(0)}ms)`);
  }

  const passed = score >= 60;
  return { passed, feedback, score };
}

function calculateGrade(percentage: number): string {
  if (percentage >= 90) return "A";
  if (percentage >= 80) return "B";
  if (percentage >= 70) return "C";
  if (percentage >= 60) return "D";
  return "F";
}


