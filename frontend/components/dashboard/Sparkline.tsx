/** Tiny data-driven sparkline (values are min–max normalised; no smoothing of the data itself). */
export function Sparkline({ values, stroke, label }: { values: number[]; stroke: string; label: string }) {
  if (!values || values.length < 2) return <span className="w-16 h-6 shrink-0" aria-hidden="true" />;
  const w = 65;
  const h = 22;
  const pad = 2;
  const min = Math.min(...values);
  const max = Math.max(...values);
  const span = max - min || 1;
  const pts = values.map((v, i) => [pad + (i / (values.length - 1)) * (w - 2 * pad), h - pad - ((v - min) / span) * (h - 2 * pad)]);
  const d = pts.reduce((acc, [x, y], i) => {
    if (i === 0) return `M ${x} ${y}`;
    const [px, py] = pts[i - 1];
    const cx = (px + x) / 2;
    return `${acc} C ${cx} ${py}, ${cx} ${y}, ${x} ${y}`;
  }, "");
  return (
    <svg className="w-16 h-6 shrink-0" viewBox={`0 0 ${w} ${h}`} role="img" aria-label={label}>
      <title>{label}</title>
      <path d={d} fill="none" stroke={stroke} strokeWidth="2.4" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}
