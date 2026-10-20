// Zoom and pan for an image canvas inside a stage element, shared by the
// labelling canvas and Quick segment.
//
// The canvas is laid out at the "fit" size (whole image visible, centred by
// the stage's flexbox) and zoomed / panned with a CSS transform, so
// getBoundingClientRect() on the canvas still maps screen points to image
// pixels exactly.
//
// Mouse wheel zooms around the cursor; the controls and the + / - / 0 keys
// zoom around the centre of the view / fit the image again.

import { useCallback, useEffect, useState, type CSSProperties, type RefObject,
         type WheelEvent } from 'react';

export const MIN_ZOOM = 1;
export const MAX_ZOOM = 32;
const STEP = 1.25;

export interface View { zoom: number; tx: number; ty: number }
const FIT: View = { zoom: 1, tx: 0, ty: 0 };

export function useZoomPan(stageRef: RefObject<HTMLDivElement | null>, width: number,
                           height: number) {
  const [fit, setFit] = useState(1);
  const [view, setView] = useState<View>(FIT);

  // Fit the image into the stage; re-fit when the stage resizes.
  useEffect(() => {
    const el = stageRef.current;
    if (!el) return;
    const update = () => setFit(Math.min(el.clientWidth / width, el.clientHeight / height));
    update();
    const ro = new ResizeObserver(update);
    ro.observe(el);
    return () => ro.disconnect();
  }, [stageRef, width, height]);

  // A new image starts fitted.
  useEffect(() => { setView(FIT); }, [width, height]);

  /** Zoom by ``factor`` keeping stage point (px, py) still (default: centre). */
  const zoomAt = useCallback((factor: number, px?: number, py?: number) => {
    const stage = stageRef.current?.getBoundingClientRect();
    if (!stage) return;
    const x = px ?? stage.width / 2, y = py ?? stage.height / 2;
    setView((v) => {
      const zoom = Math.min(MAX_ZOOM, Math.max(MIN_ZOOM, v.zoom * factor));
      if (zoom === MIN_ZOOM) return FIT;
      // Where the fitted canvas sits in the stage before the transform.
      const ox = stage.width / 2 - (width * fit) / 2, oy = stage.height / 2 - (height * fit) / 2;
      const ux = (x - ox - v.tx) / v.zoom, uy = (y - oy - v.ty) / v.zoom;
      return { zoom, tx: x - ox - ux * zoom, ty: y - oy - uy * zoom };
    });
  }, [stageRef, width, height, fit]);

  const onWheel = (e: WheelEvent<HTMLDivElement>) => {
    const stage = stageRef.current!.getBoundingClientRect();
    zoomAt(e.deltaY < 0 ? 1.2 : 1 / 1.2, e.clientX - stage.left, e.clientY - stage.top);
  };

  const panBy = useCallback((dx: number, dy: number) =>
    setView((v) => (v.zoom === MIN_ZOOM ? v : { ...v, tx: v.tx + dx, ty: v.ty + dy })), []);
  const reset = useCallback(() => setView(FIT), []);
  const zoomIn = useCallback(() => zoomAt(STEP), [zoomAt]);
  const zoomOut = useCallback(() => zoomAt(1 / STEP), [zoomAt]);

  // + / - / 0, unless the user is typing.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const tag = (e.target as HTMLElement).tagName;
      if (['INPUT', 'SELECT', 'TEXTAREA'].includes(tag) || e.ctrlKey || e.metaKey || e.altKey) return;
      if (e.key === '+' || e.key === '=') zoomIn();
      else if (e.key === '-' || e.key === '_') zoomOut();
      else if (e.key === '0') reset();
      else return;
      e.preventDefault();
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [zoomIn, zoomOut, reset]);

  const canvasStyle: CSSProperties = {
    width: width * fit,
    height: height * fit,
    transform: `translate(${view.tx}px, ${view.ty}px) scale(${view.zoom})`,
    transformOrigin: '0 0',
  };
  return { fit, view, setView, zoomAt, zoomIn, zoomOut, reset, onWheel, panBy, canvasStyle };
}

export function ZoomControls({ zoom, onZoomIn, onZoomOut, onFit }: {
  zoom: number; onZoomIn: () => void; onZoomOut: () => void; onFit: () => void;
}) {
  // Keep clicks on the controls from reaching the canvas underneath.
  const stop = (e: { stopPropagation: () => void }) => e.stopPropagation();
  return (
    <div className="zoom-controls" onPointerDown={stop} onWheel={stop}>
      <button onClick={onZoomOut} disabled={zoom <= MIN_ZOOM} title="Zoom out (−)"
              aria-label="Zoom out">−</button>
      <button onClick={onFit} disabled={zoom <= MIN_ZOOM} title="Fit the image (0)"
              className="zoom-level">{zoom <= MIN_ZOOM ? 'Fit' : `${Math.round(zoom * 100)}%`}</button>
      <button onClick={onZoomIn} disabled={zoom >= MAX_ZOOM} title="Zoom in (+)"
              aria-label="Zoom in">+</button>
    </div>
  );
}
