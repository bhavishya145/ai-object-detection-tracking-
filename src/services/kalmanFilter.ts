/**
 * AI-Powered Real-Time Object Detection & Tracking System
 * 8-State Kalman Filter for Bounding Box Trajectory Estimation
 */

export class KalmanBoxTracker {
  // State: [cx, cy, aspect_ratio, h, vx, vy, va, vh]
  private mean: number[];
  private covariance: number[][];
  private count: number = 0;

  constructor(bbox: [number, number, number, number]) {
    // bbox: [x1, y1, x2, y2]
    const w = bbox[2] - bbox[0];
    const h = bbox[3] - bbox[1];
    const cx = bbox[0] + w / 2;
    const cy = bbox[1] + h / 2;
    const a = w / Math.max(1, h);

    this.mean = [cx, cy, a, h, 0, 0, 0, 0];

    // Initial covariance
    this.covariance = [];
    const stdPos = [
      (2 * h) / 20,
      (2 * h) / 20,
      1e-2,
      (2 * h) / 20,
      (10 * h) / 160,
      (10 * h) / 160,
      1e-5,
      (10 * h) / 160,
    ];

    for (let i = 0; i < 8; i++) {
      this.covariance[i] = new Array(8).fill(0);
      this.covariance[i][i] = stdPos[i] * stdPos[i];
    }
  }

  predict(): [number, number, number, number] {
    // State transition matrix F (dt = 1)
    // x_k = x_{k-1} + v_{k-1}
    for (let i = 0; i < 4; i++) {
      this.mean[i] += this.mean[4 + i];
    }

    // Process noise inflation
    const h = Math.max(1, this.mean[3]);
    const qPos = (h / 20) ** 2;
    const qVel = (h / 160) ** 2;

    for (let i = 0; i < 4; i++) {
      this.covariance[i][i] += qPos;
      this.covariance[4 + i][4 + i] += qVel;
    }

    return this.getBbox();
  }

  update(bbox: [number, number, number, number]): void {
    const w = bbox[2] - bbox[0];
    const h = bbox[3] - bbox[1];
    const cx = bbox[0] + w / 2;
    const cy = bbox[1] + h / 2;
    const a = w / Math.max(1, h);

    const measurement = [cx, cy, a, h];

    // Kalman gain approximation for measurement update
    const alpha = 0.65; // adaptive weight
    for (let i = 0; i < 4; i++) {
      const diff = measurement[i] - this.mean[i];
      this.mean[i] += alpha * diff;
      this.mean[4 + i] += (alpha * 0.4) * diff; // velocity update
    }

    this.count++;
  }

  getBbox(): [number, number, number, number] {
    const cx = this.mean[0];
    const cy = this.mean[1];
    const a = Math.max(0.1, this.mean[2]);
    const h = Math.max(1, this.mean[3]);
    const w = a * h;

    const x1 = cx - w / 2;
    const y1 = cy - h / 2;
    const x2 = cx + w / 2;
    const y2 = cy + h / 2;

    return [x1, y1, x2, y2];
  }

  getVelocity(): [number, number] {
    return [this.mean[4], this.mean[5]];
  }
}
