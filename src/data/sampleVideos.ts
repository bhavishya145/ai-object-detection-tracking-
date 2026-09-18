/**
 * Realistic Synthetic Video Engine & Presets
 * Generates dynamic, realistic movement of cars, pedestrians, trucks, and bikes
 * on a canvas with realistic kinematics, occlusion, and road background.
 */

export interface SyntheticObject {
  id: string;
  className: 'car' | 'truck' | 'bus' | 'person' | 'bicycle';
  x: number;
  y: number;
  width: number;
  height: number;
  vx: number;
  vy: number;
  color: string;
  seed: number;
}

export class VideoScenarioSimulator {
  private type: 'traffic' | 'pedestrian';
  private width: number;
  private height: number;
  private objects: SyntheticObject[] = [];
  private frameCount: number = 0;

  constructor(type: 'traffic' | 'pedestrian', width: number = 854, height: number = 480) {
    this.type = type;
    this.width = width;
    this.height = height;
    this.initScenario();
  }

  private initScenario() {
    this.objects = [];
    if (this.type === 'traffic') {
      // Highway traffic: multiple lanes
      const classes: Array<'car' | 'truck' | 'bus'> = ['car', 'car', 'car', 'truck', 'bus'];
      const colors = ['#e2e8f0', '#ef4444', '#3b82f6', '#10b981', '#f59e0b', '#8b5cf6'];

      // Northbound & Southbound lanes
      for (let i = 0; i < 7; i++) {
        const c = classes[i % classes.length];
        const isSouth = i % 2 === 0;
        const w = c === 'car' ? 52 : c === 'truck' ? 78 : 70;
        const h = c === 'car' ? 34 : c === 'truck' ? 44 : 40;
        const laneX = isSouth ? 160 + (i % 2) * 90 : 480 + (i % 2) * 90;

        this.objects.push({
          id: `veh_${i}`,
          className: c,
          x: laneX,
          y: Math.random() * (this.height - 100),
          width: w,
          height: h,
          vx: (Math.random() - 0.5) * 0.4,
          vy: isSouth ? 1.8 + Math.random() * 1.2 : -1.8 - Math.random() * 1.2,
          color: colors[i % colors.length],
          seed: Math.random(),
        });
      }
    } else {
      // Urban pedestrian plaza
      const colors = ['#38bdf8', '#34d399', '#f43f5e', '#a855f7', '#fbbf24'];
      for (let i = 0; i < 8; i++) {
        const isBike = i === 3;
        this.objects.push({
          id: `ped_${i}`,
          className: isBike ? 'bicycle' : 'person',
          x: Math.random() * (this.width - 150) + 50,
          y: Math.random() * (this.height - 120) + 40,
          width: isBike ? 36 : 24,
          height: isBike ? 42 : 54,
          vx: (Math.random() - 0.5) * 1.8,
          vy: (Math.random() - 0.5) * 1.8,
          color: colors[i % colors.length],
          seed: Math.random(),
        });
      }
    }
  }

  public render(ctx: CanvasRenderingContext2D, w: number, h: number): Array<{ bbox: [number, number, number, number]; className: string; score: number }> {
    this.width = w;
    this.height = h;
    this.frameCount++;

    // 1. Draw realistic backdrop
    ctx.save();
    if (this.type === 'traffic') {
      // Asphalt
      ctx.fillStyle = '#0f172a';
      ctx.fillRect(0, 0, w, h);

      // Road curbs & grass
      ctx.fillStyle = '#064e3b';
      ctx.fillRect(0, 0, 70, h);
      ctx.fillRect(w - 70, 0, 70, h);

      // Road surface
      ctx.fillStyle = '#1e293b';
      ctx.fillRect(70, 0, w - 140, h);

      // Road Median divider
      ctx.strokeStyle = '#facc15';
      ctx.lineWidth = 4;
      ctx.beginPath();
      ctx.moveTo(w / 2 - 4, 0);
      ctx.lineTo(w / 2 - 4, h);
      ctx.stroke();

      ctx.beginPath();
      ctx.moveTo(w / 2 + 4, 0);
      ctx.lineTo(w / 2 + 4, h);
      ctx.stroke();

      // Lane dash lines
      ctx.strokeStyle = '#64748b';
      ctx.lineWidth = 2;
      ctx.setLineDash([20, 20]);
      [w / 2 - 130, w / 2 + 130].forEach((lx) => {
        ctx.beginPath();
        ctx.moveTo(lx, (this.frameCount * 3) % 40);
        ctx.lineTo(lx, h);
        ctx.stroke();
      });
      ctx.setLineDash([]);
    } else {
      // Urban plaza pavers
      ctx.fillStyle = '#1e1b2e';
      ctx.fillRect(0, 0, w, h);

      // Walkway grid
      ctx.strokeStyle = '#2d2a45';
      ctx.lineWidth = 1;
      for (let x = 0; x < w; x += 40) {
        ctx.beginPath();
        ctx.moveTo(x, 0);
        ctx.lineTo(x, h);
        ctx.stroke();
      }
      for (let y = 0; y < h; y += 40) {
        ctx.beginPath();
        ctx.moveTo(0, y);
        ctx.lineTo(w, y);
        ctx.stroke();
      }

      // Plaza benches/planters
      ctx.fillStyle = '#065f46';
      ctx.fillRect(50, 40, 60, 30);
      ctx.fillRect(w - 120, h - 80, 70, 30);
    }
    ctx.restore();

    // 2. Update and draw simulated objects
    const detected: Array<{ bbox: [number, number, number, number]; className: string; score: number }> = [];

    for (const obj of this.objects) {
      obj.x += obj.vx;
      obj.y += obj.vy;

      // Boundary wraps or bounces
      if (this.type === 'traffic') {
        if (obj.vy > 0 && obj.y > h + 50) obj.y = -60;
        if (obj.vy < 0 && obj.y < -60) obj.y = h + 50;
      } else {
        if (obj.x < 30 || obj.x > w - 60) obj.vx *= -1;
        if (obj.y < 30 || obj.y > h - 70) obj.vy *= -1;
      }

      // Draw synthetic body
      ctx.save();
      ctx.fillStyle = obj.color;
      ctx.shadowColor = 'rgba(0,0,0,0.5)';
      ctx.shadowBlur = 8;
      ctx.shadowOffsetY = 4;

      if (obj.className === 'person') {
        // Head
        ctx.beginPath();
        ctx.arc(obj.x + obj.width / 2, obj.y + 10, 7, 0, Math.PI * 2);
        ctx.fill();
        // Torso & legs
        ctx.fillRect(obj.x + 4, obj.y + 18, obj.width - 8, obj.height - 18);
      } else if (obj.className === 'bicycle') {
        // Bike frame
        ctx.fillRect(obj.x, obj.y + 15, obj.width, 10);
        ctx.beginPath();
        ctx.arc(obj.x + 6, obj.y + 30, 8, 0, Math.PI * 2);
        ctx.arc(obj.x + obj.width - 6, obj.y + 30, 8, 0, Math.PI * 2);
        ctx.fill();
      } else {
        // Vehicle chassis
        ctx.beginPath();
        ctx.roundRect(obj.x, obj.y, obj.width, obj.height, 6);
        ctx.fill();
        // Windshield
        ctx.fillStyle = 'rgba(15, 23, 42, 0.7)';
        ctx.fillRect(obj.x + 6, obj.y + 4, obj.width - 12, 10);
      }
      ctx.restore();

      // Detection with slight sensor noise & confidence jitter
      const conf = 0.82 + Math.sin(this.frameCount * 0.1 + obj.seed * 10) * 0.12;
      detected.push({
        bbox: [obj.x, obj.y, obj.x + obj.width, obj.y + obj.height],
        className: obj.className,
        score: Math.min(0.98, Math.max(0.65, conf)),
      });
    }

    return detected;
  }
}
