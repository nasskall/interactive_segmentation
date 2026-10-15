import { useCallback, useEffect, useMemo, useRef, useState, type MouseEvent, type PointerEvent,
         type WheelEvent } from 'react';
import type { Annotation, LabelClass, SessionState } from './api';
import { maskCanvas } from './lib';

export type Tool = 'assist' | 'box' | 'polygon' | 'select';

interface Props {
  width: number;            // item's original size; all coordinates use it
  height: number;
  image: HTMLImageElement | null;
  // Change detection: the other image of the pair, shown left of the divider.
  swipeImage?: HTMLImageElement | null;
  swipe?: number;           // 0..1 divider position
  annotations: Annotation[];
  classes: Map<number, LabelClass>;
  hidden: Set<number>;      // annotation ids not drawn
  selectedId: number | null;
  opacity: number;
  session: SessionState | null;
  sessionOverlay: HTMLImageElement | null;
  tool: Tool;
  drawColor: string;
  busy: boolean;
  onAssistClick: (x: number, y: number, positive: boolean) => void;
  onAssistBox: (x0: number, y0: number, x1: number, y1: number) => void;
  onBox: (bbox: [number, number, number, number]) => void;
  onPolygon: (points: [number, number][]) => void;
  onSelect: (id: number | null) => void;
}

const DRAG_THRESHOLD = 5;   // screen px before a press counts as a drag
const MIN_ZOOM = 1;
const MAX_ZOOM = 32;

/**
 * The labelling surface. Draws at the image's native resolution and lets
 * CSS fit it to the stage; wheel zooms around the cursor, middle-drag or
 * Space+drag pans.
 */
export default function AnnotCanvas(p: Props) {
  const stageRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [fit, setFit] = useState(1);
  const [view, setView] = useState({ zoom: 1, tx: 0, ty: 0 });
  const [drag, setDrag] = useState<null | { x0: number; y0: number; x1: number; y1: number;
                                           sx: number; sy: number; moved: boolean;
                                           pan: boolean }>(null);
  const [draft, setDraft] = useState<[number, number][]>([]);
  const [hover, setHover] = useState<[number, number] | null>(null);
  const spaceDown = useRef(false);

  // Fit the image into the stage; re-fit on resize.
  useEffect(() => {
    const el = stageRef.current;
    if (!el) return;
    const update = () => setFit(Math.min(el.clientWidth / p.width, el.clientHeight / p.height));
    update();
    const ro = new ResizeObserver(update);
    ro.observe(el);
    return () => ro.disconnect();
  }, [p.width, p.height]);

  // New image: reset zoom and any half-drawn shape.
  useEffect(() => {
    setView({ zoom: 1, tx: 0, ty: 0 });
    setDraft([]);
  }, [p.image, p.width, p.height]);
  useEffect(() => { setDraft([]); }, [p.tool]);

  const masks = useMemo(() => {
    const out = new Map<number, HTMLCanvasElement>();
    for (const a of p.annotations) {
      if (a.geometry.type === 'mask') {
        const cls = p.classes.get(a.class_id);
        out.set(a.id, maskCanvas(a.geometry.rle, cls?.color ?? '#888888'));
      }
    }
    return out;
  }, [p.annotations, p.classes]);

  // Image pixels per screen pixel, for constant on-screen line widths.
  const unit = useCallback(() => {
    const rect = canvasRef.current?.getBoundingClientRect();
    return rect && rect.width > 0 ? p.width / rect.width : 1;
  }, [p.width]);

  // ── drawing ─────────────────────────────────────────────────────────────
  useEffect(() => {
    const ctx = canvasRef.current?.getContext('2d');
    if (!ctx) return;
    const { width: W, height: H } = p;
    const u = unit();
    ctx.clearRect(0, 0, W, H);

    if (p.image) ctx.drawImage(p.image, 0, 0, W, H);
    if (p.swipeImage && p.swipe !== undefined) {
      const sx = p.swipe * W;
      ctx.save();
      ctx.beginPath();
      ctx.rect(0, 0, sx, H);
      ctx.clip();
      ctx.drawImage(p.swipeImage, 0, 0, W, H);
      ctx.restore();
      ctx.fillStyle = '#ffffff';
      ctx.fillRect(sx - u, 0, 2 * u, H);
    }

    for (const a of p.annotations) {
      if (p.hidden.has(a.id)) continue;
      const color = p.classes.get(a.class_id)?.color ?? '#888888';
      const selected = a.id === p.selectedId;
      ctx.lineWidth = (selected ? 3 : 2) * u;
      ctx.strokeStyle = color;
      const g = a.geometry;
      if (g.type === 'mask') {
        ctx.globalAlpha = selected ? Math.min(1, p.opacity + 0.25) : p.opacity;
        ctx.drawImage(masks.get(a.id)!, 0, 0, W, H);
        ctx.globalAlpha = 1;
      } else if (g.type === 'box') {
        const [x, y, w, h] = g.bbox;
        ctx.strokeRect(x, y, w, h);
        if (selected) {
          ctx.globalAlpha = p.opacity * 0.5;
          ctx.fillStyle = color;
          ctx.fillRect(x, y, w, h);
          ctx.globalAlpha = 1;
        }
      } else if (g.type === 'polygon') {
        ctx.beginPath();
        g.points.forEach(([x, y], i) => (i ? ctx.lineTo(x, y) : ctx.moveTo(x, y)));
        ctx.closePath();
        ctx.globalAlpha = p.opacity * (selected ? 0.9 : 0.6);
        ctx.fillStyle = color;
        ctx.fill();
        ctx.globalAlpha = 1;
        ctx.stroke();
      }
      if (selected && g.type !== 'tag') {
        ctx.setLineDash([6 * u, 4 * u]);
        ctx.strokeStyle = '#ffffff';
        ctx.lineWidth = u;
        if (g.type === 'box') ctx.strokeRect(...g.bbox);
        ctx.setLineDash([]);
      }
    }

    // Live model prediction (session coordinates may be downscaled).
    if (p.session && p.sessionOverlay) {
      const s = W / p.session.width;
      ctx.globalAlpha = Math.max(0.35, p.opacity);
      ctx.drawImage(p.sessionOverlay, 0, 0, W, H);
      ctx.globalAlpha = 1;
      if (p.session.box) {
        const [x0, y0, x1, y1] = p.session.box.map((v) => v * s);
        ctx.setLineDash([6 * u, 4 * u]);
        ctx.strokeStyle = '#facc15';
        ctx.lineWidth = 2 * u;
        ctx.strokeRect(x0, y0, x1 - x0, y1 - y0);
        ctx.setLineDash([]);
      }
      for (const c of p.session.clicks) {
        ctx.beginPath();
        ctx.arc(c.x * s, c.y * s, 5 * u, 0, Math.PI * 2);
        ctx.fillStyle = c.positive ? '#22c55e' : '#ef4444';
        ctx.fill();
        ctx.lineWidth = 1.5 * u;
        ctx.strokeStyle = '#ffffff';
        ctx.stroke();
      }
    }

    // Shape being dragged.
    if (drag && drag.moved && !drag.pan) {
      ctx.setLineDash([6 * u, 4 * u]);
      ctx.strokeStyle = p.tool === 'assist' ? '#facc15' : p.drawColor;
      ctx.lineWidth = 2 * u;
      ctx.strokeRect(Math.min(drag.x0, drag.x1), Math.min(drag.y0, drag.y1),
                     Math.abs(drag.x1 - drag.x0), Math.abs(drag.y1 - drag.y0));
      ctx.setLineDash([]);
    }

    // Polygon in progress.
    if (draft.length) {
      ctx.strokeStyle = p.drawColor;
      ctx.lineWidth = 2 * u;
      ctx.beginPath();
      draft.forEach(([x, y], i) => (i ? ctx.lineTo(x, y) : ctx.moveTo(x, y)));
      if (hover) ctx.lineTo(hover[0], hover[1]);
      ctx.stroke();
      draft.forEach(([x, y], i) => {
        ctx.beginPath();
        ctx.arc(x, y, (i === 0 ? 5 : 3) * u, 0, Math.PI * 2);
        ctx.fillStyle = i === 0 ? '#ffffff' : p.drawColor;
        ctx.fill();
      });
    }
  }, [p, masks, drag, draft, hover, unit, view, fit]);

  // ── coordinates ─────────────────────────────────────────────────────────
  const toImage = (e: { clientX: number; clientY: number }): [number, number] => {
    const rect = canvasRef.current!.getBoundingClientRect();
    return [((e.clientX - rect.left) / rect.width) * p.width,
            ((e.clientY - rect.top) / rect.height) * p.height];
  };
  const inside = ([x, y]: [number, number]) => x >= 0 && y >= 0 && x < p.width && y < p.height;
  const clampPt = ([x, y]: [number, number]): [number, number] =>
    [Math.min(Math.max(x, 0), p.width), Math.min(Math.max(y, 0), p.height)];

  // ── hit testing (select tool) ───────────────────────────────────────────
  const hitTest = (x: number, y: number): number | null => {
    const ctx = canvasRef.current!.getContext('2d')!;
    for (let i = p.annotations.length - 1; i >= 0; i--) {
      const a = p.annotations[i];
      if (p.hidden.has(a.id)) continue;
      const g = a.geometry;
      if (g.type === 'box') {
        const [bx, by, bw, bh] = g.bbox;
        if (x >= bx && x <= bx + bw && y >= by && y <= by + bh) return a.id;
      } else if (g.type === 'polygon') {
        const path = new Path2D();
        g.points.forEach(([px, py], k) => (k ? path.lineTo(px, py) : path.moveTo(px, py)));
        path.closePath();
        if (ctx.isPointInPath(path, x, y)) return a.id;
      } else if (g.type === 'mask') {
        const m = masks.get(a.id)!;
        const px = m.getContext('2d')!.getImageData(Math.floor(x), Math.floor(y), 1, 1).data;
        if (px[3] > 0) return a.id;
      }
    }
    return null;
  };

  // ── pointer handling ────────────────────────────────────────────────────
  const onPointerDown = (e: PointerEvent<HTMLCanvasElement>) => {
    if (e.button === 2) return;   // right button: handled by onContextMenu
    const pan = e.button === 1 || spaceDown.current;
    const [x, y] = toImage(e);
    e.currentTarget.setPointerCapture(e.pointerId);
    setDrag({ x0: x, y0: y, x1: x, y1: y, sx: e.clientX, sy: e.clientY, moved: false, pan });
  };

  const onPointerMove = (e: PointerEvent<HTMLCanvasElement>) => {
    const pt = toImage(e);
    if (p.tool === 'polygon') setHover(clampPt(pt));
    if (!drag) return;
    const moved = drag.moved || Math.hypot(e.clientX - drag.sx, e.clientY - drag.sy) > DRAG_THRESHOLD;
    if (drag.pan) {
      setView((v) => ({ ...v, tx: v.tx + e.movementX, ty: v.ty + e.movementY }));
      setDrag({ ...drag, moved });
      return;
    }
    const [x1, y1] = clampPt(pt);
    setDrag({ ...drag, x1, y1, moved });
  };

  const onPointerUp = (e: PointerEvent<HTMLCanvasElement>) => {
    const d = drag;
    setDrag(null);
    if (!d || d.pan || p.busy) return;
    const [x, y] = toImage(e);
    const x0 = Math.min(d.x0, d.x1), x1 = Math.max(d.x0, d.x1);
    const y0 = Math.min(d.y0, d.y1), y1 = Math.max(d.y0, d.y1);
    const isBox = d.moved && x1 - x0 >= 2 && y1 - y0 >= 2;

    if (p.tool === 'assist') {
      if (isBox) p.onAssistBox(x0, y0, x1, y1);
      else if (!d.moved && inside([x, y])) p.onAssistClick(x, y, true);
    } else if (p.tool === 'box') {
      if (isBox) p.onBox([x0, y0, x1 - x0, y1 - y0]);
    } else if (p.tool === 'polygon' && !d.moved && inside([x, y])) {
      const first = draft[0];
      const closeDist = 8 * unit();
      if (draft.length >= 3 && first && Math.hypot(x - first[0], y - first[1]) < closeDist) {
        p.onPolygon(draft);
        setDraft([]);
      } else {
        setDraft([...draft, [x, y]]);
      }
    } else if (p.tool === 'select' && !d.moved) {
      p.onSelect(hitTest(x, y));
    }
  };

  const onContextMenu = (e: MouseEvent<HTMLCanvasElement>) => {
    e.preventDefault();
    if (p.busy || p.tool !== 'assist') return;
    const pt = toImage(e);
    if (inside(pt)) p.onAssistClick(pt[0], pt[1], false);
  };

  const onWheel = (e: WheelEvent<HTMLDivElement>) => {
    const stage = stageRef.current!.getBoundingClientRect();
    const px = e.clientX - stage.left, py = e.clientY - stage.top;
    setView((v) => {
      const zoom = Math.min(MAX_ZOOM, Math.max(MIN_ZOOM, v.zoom * (e.deltaY < 0 ? 1.2 : 1 / 1.2)));
      if (zoom === MIN_ZOOM) return { zoom, tx: 0, ty: 0 };
      // Keep the point under the cursor fixed.
      const ox = stage.width / 2 - (p.width * fit) / 2, oy = stage.height / 2 - (p.height * fit) / 2;
      const ux = (px - ox - v.tx) / v.zoom, uy = (py - oy - v.ty) / v.zoom;
      return { zoom, tx: px - ox - ux * zoom, ty: py - oy - uy * zoom };
    });
  };

  // Keys for the polygon tool and Space-to-pan.
  useEffect(() => {
    const typing = (e: KeyboardEvent) =>
      ['INPUT', 'SELECT', 'TEXTAREA'].includes((e.target as HTMLElement).tagName);
    const down = (e: KeyboardEvent) => {
      if (typing(e)) return;
      if (e.key === ' ') spaceDown.current = true;
      if (p.tool !== 'polygon' || !draft.length) return;
      if (e.key === 'Enter' && draft.length >= 3) {
        e.preventDefault();
        e.stopImmediatePropagation();
        p.onPolygon(draft);
        setDraft([]);
      } else if (e.key === 'Backspace') {
        e.preventDefault();
        setDraft(draft.slice(0, -1));
      } else if (e.key === 'Escape') {
        e.stopImmediatePropagation();
        setDraft([]);
      }
    };
    const up = (e: KeyboardEvent) => { if (e.key === ' ') spaceDown.current = false; };
    // Capture phase so Enter / Esc here take priority over the workspace keys.
    window.addEventListener('keydown', down, true);
    window.addEventListener('keyup', up);
    return () => {
      window.removeEventListener('keydown', down, true);
      window.removeEventListener('keyup', up);
    };
  }, [p, draft]);

  const cursor = drag?.pan ? 'grabbing' : p.busy ? 'progress'
    : p.tool === 'select' ? 'pointer' : 'crosshair';

  return (
    <div className="annot-stage" ref={stageRef} onWheel={onWheel}>
      <canvas
        ref={canvasRef}
        width={p.width}
        height={p.height}
        style={{
          width: p.width * fit,
          height: p.height * fit,
          transform: `translate(${view.tx}px, ${view.ty}px) scale(${view.zoom})`,
          cursor,
        }}
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={onPointerUp}
        onPointerLeave={() => setHover(null)}
        onContextMenu={onContextMenu}
        onAuxClick={(e) => e.preventDefault()}
      />
      {view.zoom > 1 && (
        <button className="fit-button" onClick={() => setView({ zoom: 1, tx: 0, ty: 0 })}>
          Fit ({Math.round(view.zoom * 100)}%)
        </button>
      )}
    </div>
  );
}
