/**
 * Connected components labeling module
 * Identifies and extracts individual shape regions from binary image
 */

export class ComponentLabeler {
  /**
   * Label connected components using BFS with 8-connectivity
   * Returns map of label -> array of pixels in that component
   */
  static labelConnectedComponents(
    bin: Uint8ClampedArray,
    w: number,
    h: number
  ): Map<number, Array<{ x: number; y: number }>> {
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
}
