import { useEffect, useRef, useState, type MouseEvent } from 'react';
import type { Click } from './api';

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

function loadImage(src: string): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = () => reject(new Error(`Failed to load ${src.slice(0, 40)}`));
    img.src = src;
  });
}

/**
 * Where the image is drawn inside the canvas element. The CSS uses
 * ``object-fit: contain``, so the element box can be larger than the
 * picture, which is centred in it.
 */
function contentRect(canvas: HTMLCanvasElement, width: number, height: number) {
  const rect = canvas.getBoundingClientRect();
  const scale = Math.min(rect.width / width, rect.height / height);
  const w = width * scale, h = height * scale;
  return { left: rect.left + (rect.width - w) / 2, top: rect.top + (rect.height - h) / 2,
           width: w, height: h };
}

/**
 * Draws image + mask overlay + clicks at native resolution; CSS scales the
 * canvas to fit. Left click = positive, right click = negative.
 */
export default function SegCanvas(props: Props) {
  const { imageUrl, overlayUrl, width, height, clicks, alpha, clickRadius, busy, onClick } = props;
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [image, setImage] = useState<HTMLImageElement | null>(null);
  const [overlay, setOverlay] = useState<HTMLImageElement | null>(null);

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
    const rect = contentRect(canvasRef.current!, width, height);
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
  }, [image, overlay, width, height, clicks, alpha, clickRadius]);

  const handle = (e: MouseEvent<HTMLCanvasElement>, positive: boolean) => {
    e.preventDefault();
    if (busy) return;
    const rect = contentRect(e.currentTarget, width, height);
    const x = Math.floor(((e.clientX - rect.left) / rect.width) * width);
    const y = Math.floor(((e.clientY - rect.top) / rect.height) * height);
    if (x < 0 || y < 0 || x >= width || y >= height) return;
    onClick(x, y, positive);
  };

  return (
    <canvas
      ref={canvasRef}
      width={width}
      height={height}
      className={busy ? 'seg-canvas busy' : 'seg-canvas'}
      onClick={(e) => handle(e, true)}
      onContextMenu={(e) => handle(e, false)}
    />
  );
}
