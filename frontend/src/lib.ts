// Small shared helpers: hash routing, RLE masks, authenticated images.

import { useEffect, useState } from 'react';
import { fetchImageUrl, type Rle } from './api';

// ── routing ────────────────────────────────────────────────────────────────
// Routes live in the URL hash so the backend's static file server needs no
// rewrites: #/ projects · #/projects/3[/classes|team|models|export|settings] · #/items/12
// workspace · #/users · #/audit · #/quick

export type ProjectTab = 'items' | 'classes' | 'team' | 'models' | 'export' | 'settings';

export type Route =
  | { page: 'projects' }
  | { page: 'project'; id: number; tab: ProjectTab }
  | { page: 'item'; id: number }
  | { page: 'users' }
  | { page: 'audit' }
  | { page: 'quick' };

const TABS: ProjectTab[] = ['items', 'classes', 'team', 'models', 'export', 'settings'];

export function parseRoute(hash: string): Route {
  const parts = hash.replace(/^#\/?/, '').split('/').filter(Boolean);
  if (parts[0] === 'projects' && parts[1]) {
    const tab = TABS.find((t) => t === parts[2]) ?? 'items';
    return { page: 'project', id: Number(parts[1]), tab };
  }
  if (parts[0] === 'users') return { page: 'users' };
  if (parts[0] === 'audit') return { page: 'audit' };
  if (parts[0] === 'items' && parts[1]) return { page: 'item', id: Number(parts[1]) };
  if (parts[0] === 'quick') return { page: 'quick' };
  return { page: 'projects' };
}

export const href = {
  projects: () => '#/',
  project: (id: number, tab: ProjectTab = 'items') =>
    tab === 'items' ? `#/projects/${id}` : `#/projects/${id}/${tab}`,
  users: () => '#/users',
  audit: () => '#/audit',
  item: (id: number) => `#/items/${id}`,
  quick: () => '#/quick',
};

export function useRoute(): Route {
  const [route, setRoute] = useState(() => parseRoute(window.location.hash));
  useEffect(() => {
    const onHash = () => setRoute(parseRoute(window.location.hash));
    window.addEventListener('hashchange', onHash);
    return () => window.removeEventListener('hashchange', onHash);
  }, []);
  return route;
}

export const go = (to: string) => { window.location.hash = to; };

// ── masks ──────────────────────────────────────────────────────────────────

/** COCO uncompressed RLE (column-major) → row-major 0/1 bytes. */
export function decodeRle(rle: Rle): Uint8Array {
  const [h, w] = rle.size;
  const out = new Uint8Array(h * w);
  let pos = 0;
  rle.counts.forEach((n, i) => {
    if (i % 2 === 1) {
      for (let k = pos; k < pos + n; k++) {
        const col = Math.floor(k / h);
        out[(k - col * h) * w + col] = 1;
      }
    }
    pos += n;
  });
  return out;
}

export function hexToRgb(hex: string): [number, number, number] {
  const n = parseInt(hex.slice(1), 16);
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
}

/** An offscreen canvas with the mask painted in ``color`` (opaque). */
export function maskCanvas(rle: Rle, color: string): HTMLCanvasElement {
  const [h, w] = rle.size;
  const bits = decodeRle(rle);
  const canvas = document.createElement('canvas');
  canvas.width = w;
  canvas.height = h;
  const ctx = canvas.getContext('2d')!;
  const img = ctx.createImageData(w, h);
  const [r, g, b] = hexToRgb(color);
  for (let i = 0; i < bits.length; i++) {
    if (bits[i]) {
      img.data[i * 4] = r;
      img.data[i * 4 + 1] = g;
      img.data[i * 4 + 2] = b;
      img.data[i * 4 + 3] = 255;
    }
  }
  ctx.putImageData(img, 0, 0);
  return canvas;
}

// ── images ─────────────────────────────────────────────────────────────────

/** Saves a blob through a temporary link (exports, masks). */
export function saveBlob(blob: Blob, filename: string) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

/** Loads an image that needs the auth header; null while loading. */
export function useAuthedImage(url: string | null): HTMLImageElement | null {
  const [image, setImage] = useState<HTMLImageElement | null>(null);
  useEffect(() => {
    setImage(null);
    if (!url) return;
    let live = true;
    let objectUrl: string | null = null;
    fetchImageUrl(url)
      .then((u) => {
        objectUrl = u;
        const img = new Image();
        img.onload = () => { if (live) setImage(img); };
        img.src = u;
      })
      .catch(console.error);
    return () => {
      live = false;
      if (objectUrl) URL.revokeObjectURL(objectUrl);
    };
  }, [url]);
  return image;
}

/** Loads a plain (data: or public) image URL. */
export function useImage(url: string | null): HTMLImageElement | null {
  const [image, setImage] = useState<HTMLImageElement | null>(null);
  useEffect(() => {
    if (!url) { setImage(null); return; }
    let live = true;
    const img = new Image();
    img.onload = () => { if (live) setImage(img); };
    img.src = url;
    return () => { live = false; };
  }, [url]);
  return image;
}
