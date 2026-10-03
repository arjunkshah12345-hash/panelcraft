import { useMemo } from "react";

/**
 * Identity against variety, as an inline SVG scatter.
 *
 * Deliberately hand-rolled rather than a charting dependency: one scatter, one dashed
 * path, two marker classes. A library would be more bytes than the chart.
 */
export default function TradeoffChart({ tradeoff, recommended, trustworthy }) {
  const W = 620;
  const H = 420;
  const pad = { top: 24, right: 28, bottom: 58, left: 66 };

  const geometry = useMemo(() => {
    if (!tradeoff?.length) return null;

    const xs = tradeoff.map((p) => p.variety);
    const ys = tradeoff.map((p) => p.identity_mean);
    const xMin = Math.min(...xs);
    const xMax = Math.max(...xs);
    const yMin = Math.min(...ys);
    const yMax = Math.max(...ys);

    // Degenerate ranges would collapse to a point; give them a little room instead.
    const padX = (xMax - xMin) * 0.12 || 0.05;
    const padY = (yMax - yMin) * 0.12 || 0.02;
    const lo = { x: xMin - padX, y: yMin - padY };
    const hi = { x: xMax + padX, y: yMax + padY };

    const sx = (v) => pad.left + ((v - lo.x) / (hi.x - lo.x)) * (W - pad.left - pad.right);
    const sy = (v) => H - pad.bottom - ((v - lo.y) / (hi.y - lo.y)) * (H - pad.top - pad.bottom);

    return {
      sx,
      sy,
      lo,
      hi,
      path: tradeoff.map((p, i) => `${i ? "L" : "M"}${sx(p.variety)},${sy(p.identity_mean)}`).join(" "),
    };
  }, [tradeoff]);

  if (!geometry) return null;

  const ticks = (loV, hiV, fmt) => {
    const steps = [0, 0.25, 0.5, 0.75, 1];
    return steps
      .map((f) => loV + f * (hiV - loV))
      .map((v) => ({ v, text: fmt(v) }));
  };

  return (
    <svg
      className="chart"
      viewBox={`0 0 ${W} ${H}`}
      role="img"
      aria-label="Identity consistency plotted against panel variety for each identity strength"
    >
      {ticks(geometry.lo.y, geometry.hi.y, (v) => v.toFixed(2)).map(({ v, text }) => (
        <g key={`y${text}`}>
          <line
            x1={pad.left}
            x2={W - pad.right}
            y1={geometry.sy(v)}
            y2={geometry.sy(v)}
            stroke="#e6e2d9"
            strokeWidth="1"
          />
          <text
            x={pad.left - 10}
            y={geometry.sy(v) + 3.5}
            textAnchor="end"
            fontSize="10"
            fontFamily="ui-monospace, monospace"
            fill="#b9b2a6"
          >
            {text}
          </text>
        </g>
      ))}

      {ticks(geometry.lo.x, geometry.hi.x, (v) => v.toFixed(2)).map(({ v, text }) => (
        <text
          key={`x${text}`}
          x={geometry.sx(v)}
          y={H - pad.bottom + 20}
          textAnchor="middle"
          fontSize="10"
          fontFamily="ui-monospace, monospace"
          fill="#b9b2a6"
        >
          {text}
        </text>
      ))}

      <path d={geometry.path} fill="none" stroke="#b9b2a6" strokeWidth="1.5" strokeDasharray="4 4" />

      {tradeoff.map((point) => {
        const isRecommended = recommended != null && point.identity_strength === recommended;
        return (
          <g key={point.identity_strength}>
            <circle
              cx={geometry.sx(point.variety)}
              cy={geometry.sy(point.identity_mean)}
              r={isRecommended ? 9 : point.dominated ? 6 : 7}
              fill={point.dominated ? "#efece5" : "#0566ff"}
              stroke={isRecommended ? "#14110f" : point.dominated ? "#b9b2a6" : "#14110f"}
              strokeWidth={isRecommended ? 2 : 1.1}
            />
            <text
              x={geometry.sx(point.variety) + 11}
              y={geometry.sy(point.identity_mean) - 8}
              fontSize="10.5"
              fontFamily="ui-monospace, monospace"
              fontWeight="600"
              fill="#14110f"
            >
              {point.identity_strength}
            </text>
          </g>
        );
      })}

      <text
        x={W / 2}
        y={H - 12}
        textAnchor="middle"
        fontSize="11"
        fontFamily="ui-monospace, monospace"
        fill="#14110f"
      >
        panel variety (face excluded) →
      </text>
      <text
        x={16}
        y={H / 2}
        textAnchor="middle"
        fontSize="11"
        fontFamily="ui-monospace, monospace"
        fill={trustworthy ? "#14110f" : "#b4472e"}
        transform={`rotate(-90 16 ${H / 2})`}
      >
        {trustworthy ? "identity consistency →" : "identity (untrustworthy here) →"}
      </text>

      <g transform={`translate(${pad.left} ${pad.top - 12})`}>
        <circle cx="0" cy="-3" r="5" fill="#0566ff" stroke="#14110f" strokeWidth="1" />
        <text x="11" y="0" fontSize="10" fontFamily="ui-monospace, monospace" fill="#6b635a">
          non-dominated
        </text>
        <circle cx="88" cy="-3" r="5" fill="#efece5" stroke="#b9b2a6" strokeWidth="1" />
        <text x="99" y="0" fontSize="10" fontFamily="ui-monospace, monospace" fill="#6b635a">
          dominated
        </text>
      </g>
    </svg>
  );
}