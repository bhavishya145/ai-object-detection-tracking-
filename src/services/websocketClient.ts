/**
 * AI Vision — Real-Time Object Detection & Tracking
 * WebSocket Client Bridge for Python FastAPI / Ultralytics YOLOv8 backend
 */

export interface WebSocketVisionPayload {
  type: 'result';
  fps: number;
  latencyMs: number;
  activeCount: number;
  uniqueCount: number;
  countIn: number;
  countOut: number;
  classCounts: Record<string, number>;
  events: Array<{
    id: string;
    timestamp: string;
    trackId: number;
    className: string;
    confidence: number;
    action: string;
  }>;
  tracks: Array<{
    trackId: number;
    bbox: [number, number, number, number];
    score: number;
    className: string;
    classId: number;
    trail: Array<[number, number]>;
    state: string;
    color: string;
    velocity: [number, number];
  }>;
}

export type WSConnectionStatus = 'disconnected' | 'connecting' | 'connected' | 'error';

export class VisionWebSocketClient {
  private ws: WebSocket | null = null;
  private url: string;
  private status: WSConnectionStatus = 'disconnected';
  private onStatusChange?: (status: WSConnectionStatus) => void;
  private onResult?: (data: WebSocketVisionPayload) => void;
  private reconnectTimer: any = null;
  private isManuallyClosed = false;

  constructor(
    url: string = 'ws://localhost:8000/ws',
    onResult?: (data: WebSocketVisionPayload) => void,
    onStatusChange?: (status: WSConnectionStatus) => void
  ) {
    this.url = url;
    this.onResult = onResult;
    this.onStatusChange = onStatusChange;
  }

  public connect(): void {
    if (this.ws && (this.ws.readyState === WebSocket.OPEN || this.ws.readyState === WebSocket.CONNECTING)) {
      return;
    }

    this.isManuallyClosed = false;
    this.setStatus('connecting');

    try {
      this.ws = new WebSocket(this.url);

      this.ws.onopen = () => {
        this.setStatus('connected');
        console.log(`[VisionWS] Connected to AI Vision Python Backend at ${this.url}`);
      };

      this.ws.onmessage = (event) => {
        try {
          const parsed = JSON.parse(event.data);
          if (parsed.type === 'result' && this.onResult) {
            this.onResult(parsed);
          }
        } catch (e) {
          console.warn('[VisionWS] JSON parse error:', e);
        }
      };

      this.ws.onerror = (err) => {
        this.setStatus('error');
        console.warn('[VisionWS] WebSocket error (Backend offline or unreachable):', err);
      };

      this.ws.onclose = () => {
        this.setStatus('disconnected');
        this.ws = null;
        if (!this.isManuallyClosed) {
          // Schedule reconnect attempt
          clearTimeout(this.reconnectTimer);
          this.reconnectTimer = setTimeout(() => this.connect(), 4000);
        }
      };
    } catch (err) {
      this.setStatus('error');
      console.warn('[VisionWS] Failed to instantiate WebSocket:', err);
    }
  }

  public sendFrame(
    canvas: HTMLCanvasElement,
    conf: number = 0.4,
    iou: number = 0.35,
    tripwireRatio: number = 0.58
  ): boolean {
    if (!this.ws || this.ws.readyState !== WebSocket.OPEN) {
      return false;
    }

    try {
      const dataUrl = canvas.toDataURL('image/jpeg', 0.7);
      this.ws.send(
        JSON.stringify({
          type: 'frame',
          image: dataUrl,
          conf,
          iou,
          tripwireRatio,
        })
      );
      return true;
    } catch (e) {
      console.error('[VisionWS] Error sending frame to WebSocket:', e);
      return false;
    }
  }

  public reset(): void {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify({ type: 'reset' }));
    }
  }

  public disconnect(): void {
    this.isManuallyClosed = true;
    clearTimeout(this.reconnectTimer);
    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }
    this.setStatus('disconnected');
  }

  public getStatus(): WSConnectionStatus {
    return this.status;
  }

  private setStatus(status: WSConnectionStatus): void {
    this.status = status;
    if (this.onStatusChange) {
      this.onStatusChange(status);
    }
  }
}
