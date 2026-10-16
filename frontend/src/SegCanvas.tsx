import { useEffect, useRef, useState, type MouseEvent, type PointerEvent } from 'react';
import type { Click } from './api';
import { useZoomPan, ZoomControls } from './zoom';

interface Props {
  imageUrl: string;
  overlayUrl: string;
  width: number;   // image size the backend's click coordinates refer to
  height: number;
  clicks: Click[];
  alpha: number;
  clickRadius: number;
  busy: boolean;
  onClick: (x: number, y: number, positive: boolean) => void;
}

const DRAG_THRESHOLD = 5;   // screen px before a press counts as a drag

function loadImage(src: string): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = () => reject(new Error(`Failed to load ${src.slice(0, 40)}`));
    img.src = src;
  });
}

/**
 * Draws image + mask overlay + clicks at native resolution; CSS fits the
 * canvas to the stage and zooms it (see zoom.tsx). Left click = positive,
 * right click = negative; when zoomed in, left-drag pans (middle-drag pans
 * at any zoom). Space is left to the page's "finish object" shortcut.
 */
export default function SegCanvas(props: Props) {
  const { imageUrl, overlayUrl, width, height, clicks, alpha, clickRadius, busy, onClick } = props;
  const stageRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const zoom = useZoomPan(stageRef, width, height);
  const [image, setImage] = useState<HTMLImageElement | null>(null);
  const [overlay, setOverlay] = useState<HTMLImageElement | null>(null);
  const press = useRef<{ x: number; y: number; button: number; moved: boolean } | null>(null);
  const [panning, setPanning] = useState(false);

  useEffect(() => {
    let live = true;
    loadImage(imageUrl).then((img) => live && setImage(img)).catch(console.error);
    return () => { live = false; };
  }, [imageUrl]);

  useEffect(() => {
    let live = true;
    loadImage(overlayUrl).then((img) => live && setOverlay(img)).catch(console.error);
    return () => { live = false; };
  }, [overlayUrl]);

  useEffect(() => {
    const ctx = canvasRef.current?.getContext('2d');
    if (!ctx) return;
    ctx.clearRect(0, 0, width, height);
    if (image) ctx.drawImage(image, 0, 0, width, height);
    if (overlay) {
      ctx.globalAlpha = alpha;
      ctx.drawImage(overlay, 0, 0, width, height);
      ctx.globalAlpha = 1;
    }
    // Keep click markers a constant on-screen size regardless of zoom.
    const rect = canvasRef.current!.getBoundingClientRect();
    const scale = rect.width > 0 ? width / rect.width : 1;
    const r = clickRadius * scale;
    for (const c of clicks) {
      ctx.beginPath();
      ctx.arc(c.x, c.y, r, 0, Math.PI * 2);
      ctx.fillStyle = c.positive ? '#22c55e' : '#ef4444';
      ctx.fill();
      ctx.lineWidth = Math.max(1, r / 3);
      ctx.strokeStyle = '#ffffff';
      ctx.stroke();
    }
  }, [image, overlay, width, height, clicks, alpha, clickRadius, zoom.view, zoom.fit]);

  const toImage = (e: { clientX: number; clientY: number }): [number, number] | null => {
    const rect = canvasRef.current!.getBoundingClientRect();
    const x = Math.floor(((e.clientX - rect.left) / rect.width) * width);
    const y = Math.floor(((e.clientY - rect.top) / rect.height) * height);
    return x < 0 || y < 0 || x >= width || y >= height ? null : [x, y];
  };

  const onPointerDown = (e: PointerEvent<HTMLCanvasElement>) => {
    if (e.button === 2) return;   // right button: onContextMenu
    e.currentTarget.setPointerCapture(e.pointerId);
    press.current = { x: e.clientX, y: e.clientY, button: e.button, moved: false };
  };

  const onPointerMove = (e: PointerEvent<HTMLCanvasElement>) => {
    const p = press.current;
    if (!p) return;
    if (!p.moved && Math.hypot(e.clientX - p.x, e.clientY - p.y) > DRAG_THRESHOLD) {
      p.moved = true;
      setPanning(true);
    }
    if (p.moved && (p.button === 1 || zoom.view.zoom > 1)) zoom.panBy(e.movementX, e.movementY);
  };

  const onPointerUp = (e: PointerEvent<HTMLCanvasElement>) => {
    const p = press.current;
    press.current = null;
    setPanning(false);
    if (!p || p.moved || p.button !== 0 || busy) return;
    const pt = toImage(e);
    if (pt) onClick(pt[0], pt[1], true);
  };

  const onContextMenu = (e: MouseEvent<HTMLCanvasElement>) => {
    e.preventDefault();
    if (busy) return;
    const pt = toImage(e);
    if (pt) onClick(pt[0], pt[1], false);
  };

  const cursor = panning ? 'grabbing' : busy ? 'progress' : 'crosshair';
  return (
    <div className="annot-stage" ref={stageRef} onWheel={zoom.onWheel}>
      <canvas
        ref={canvasRef}
        width={width}
        height={height}
        className="seg-canvas"
        style={{ ...zoom.canvasStyle, cursor }}
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={onPointerUp}
        onContextMenu={onContextMenu}
        onAuxClick={(e) => e.preventDefault()}
      />
      <ZoomControls zoom={zoom.view.zoom} onZoomIn={zoom.zoomIn} onZoomOut={zoom.zoomOut}
                    onFit={zoom.reset} />
    </div>
  );
}
