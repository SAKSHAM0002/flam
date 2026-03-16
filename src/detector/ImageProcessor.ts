/**
 * Image preprocessing module
 * Handles grayscale conversion, blur, threshold, and morphological operations
 */

export class ImageProcessor {
  /**
   * Convert RGB image to grayscale using standard luminosity formula
   */
  static convertToGrayscale(imageData: ImageData): Uint8ClampedArray {
    const d = imageData.data;
    const out = new Uint8ClampedArray(imageData.width * imageData.height);
    for (let i = 0, j = 0; i < d.length; i += 4, j++) {
      out[j] = Math.round(0.299 * d[i] + 0.587 * d[i + 1] + 0.114 * d[i + 2]);
    }
    return out;
  }

  /**
   * Apply Gaussian blur with edge clamping (replicate border pixels)
   * Uses 3x3 kernel to fix border-zero bug
   */
  static applyGaussianBlur(gray: Uint8ClampedArray, w: number, h: number): Uint8ClampedArray {
    const out = new Uint8ClampedArray(gray.length);
    const kernel = [1, 2, 1, 2, 4, 2, 1, 2, 1];
    const ksum = 16;

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

  /**
   * Apply global threshold to convert grayscale to binary
   */
  static applyGlobalThreshold(gray: Uint8ClampedArray, threshold = 128): Uint8ClampedArray {
    const out = new Uint8ClampedArray(gray.length);
    for (let i = 0; i < gray.length; i++) {
      out[i] = gray[i] < threshold ? 255 : 0;
    }
    return out;
  }

  /**
   * Morphological closing: dilation followed by erosion
   * Uses 8-connectivity for robustness
   */
  static applyMorphologicalClosing(bin: Uint8ClampedArray, w: number, h: number): Uint8ClampedArray {
    const dilated = new Uint8ClampedArray(bin.length);
    
    // Dilation: if any 8-neighbor is foreground, set foreground
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

    // Erosion: only keep foreground if all 8-neighbors are foreground
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
}
