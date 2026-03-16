import React from 'react';
import { AbsoluteFill, useCurrentFrame, interpolate, spring, useVideoConfig } from 'remotion';
import { renderBackground } from '../lib/backgrounds';
import { BrandTheme } from '../lib/types';

interface Props {
  vizType: 'bracket' | 'dot_matrix_number' | 'dot_grid' | 'bar_chart';
  data: Record<string, any>;
  background?: string;
  brand: BrandTheme;
}

/* ── 5x7 dot-matrix font for digits ── */
const DOT_MATRIX: Record<string, number[]> = {
  '0': [0x0E, 0x11, 0x13, 0x15, 0x19, 0x11, 0x0E],
  '1': [0x04, 0x0C, 0x04, 0x04, 0x04, 0x04, 0x0E],
  '2': [0x0E, 0x11, 0x01, 0x02, 0x04, 0x08, 0x1F],
  '3': [0x0E, 0x11, 0x01, 0x06, 0x01, 0x11, 0x0E],
  '4': [0x02, 0x06, 0x0A, 0x12, 0x1F, 0x02, 0x02],
  '5': [0x1F, 0x10, 0x1E, 0x01, 0x01, 0x11, 0x0E],
  '6': [0x06, 0x08, 0x10, 0x1E, 0x11, 0x11, 0x0E],
  '7': [0x1F, 0x01, 0x02, 0x04, 0x08, 0x08, 0x08],
  '8': [0x0E, 0x11, 0x11, 0x0E, 0x11, 0x11, 0x0E],
  '9': [0x0E, 0x11, 0x11, 0x0F, 0x01, 0x02, 0x0C],
};

/* ── Bracket visualization ── */
const BracketViz: React.FC<{ data: Record<string, any>; brand: BrandTheme }> = ({ data, brand }) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const rounds = data.rounds ?? 3;
  const centerLabel = data.centerLabel ?? '';

  const lineColor = brand.accentColor;
  const width = 800;
  const height = 500;
  const halfW = width / 2;

  // Animate stroke drawing
  const drawProgress = interpolate(frame, [5, 50], [0, 1], {
    extrapolateLeft: 'clamp',
    extrapolateRight: 'clamp',
  });

  // Center label spring
  const labelProgress = spring({
    frame: frame - 45,
    fps,
    config: { damping: 14, stiffness: 100, mass: 0.7 },
  });
  const labelScale = interpolate(labelProgress, [0, 1], [0.3, 1]);
  const labelOpacity = interpolate(labelProgress, [0, 1], [0, 1]);

  // Generate bracket lines
  const lines: Array<{ x1: number; y1: number; x2: number; y2: number }> = [];

  for (let r = 0; r < rounds; r++) {
    const count = Math.pow(2, rounds - r - 1);
    const spacing = height / count;
    const xStart = (r / rounds) * halfW;
    const xEnd = ((r + 1) / rounds) * halfW;

    for (let i = 0; i < count; i++) {
      const y = spacing * (i + 0.5);
      // Left side horizontal
      lines.push({ x1: xStart, y1: y, x2: xEnd, y2: y });
      // Right side (mirror)
      lines.push({ x1: width - xStart, y1: y, x2: width - xEnd, y2: y });
    }

    // Vertical connectors
    if (r < rounds - 1) {
      const nextCount = Math.pow(2, rounds - r - 2);
      const nextSpacing = height / nextCount;
      for (let i = 0; i < nextCount; i++) {
        const y1 = spacing * (i * 2 + 0.5);
        const y2 = spacing * (i * 2 + 1.5);
        // Left vertical
        lines.push({ x1: xEnd, y1, x2: xEnd, y2 });
        // Right vertical
        lines.push({ x1: width - xEnd, y1, x2: width - xEnd, y2 });
      }
    }
  }

  return (
    <div style={{ position: 'relative', width, height }}>
      <svg width={width} height={height}>
        {lines.map((line, i) => {
          const lineLen = Math.sqrt(
            Math.pow(line.x2 - line.x1, 2) + Math.pow(line.y2 - line.y1, 2),
          );
          const dashOffset = lineLen * (1 - drawProgress);
          return (
            <line
              key={i}
              x1={line.x1}
              y1={line.y1}
              x2={line.x2}
              y2={line.y2}
              stroke={lineColor}
              strokeWidth={2}
              strokeDasharray={lineLen}
              strokeDashoffset={dashOffset}
              opacity={0.8}
            />
          );
        })}
      </svg>
      {centerLabel && (
        <div
          style={{
            position: 'absolute',
            top: '50%',
            left: '50%',
            transform: `translate(-50%, -50%) scale(${labelScale})`,
            opacity: labelOpacity,
            fontSize: 32,
            fontWeight: 800,
            color: brand.accentColor,
            textAlign: 'center',
            backgroundColor: brand.backgroundColor,
            padding: '12px 24px',
            borderRadius: 12,
          }}
        >
          {centerLabel}
        </div>
      )}
    </div>
  );
};

/* ── Dot matrix number visualization ── */
const DotMatrixViz: React.FC<{ data: Record<string, any>; brand: BrandTheme }> = ({
  data,
  brand,
}) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const text = String(data.text ?? '0');
  const dotSize = 10;
  const gap = 4;
  const charGap = 8;

  // Build dot positions for all characters
  const dots: Array<{ x: number; y: number; index: number }> = [];
  let offsetX = 0;
  let dotIndex = 0;

  for (const char of text) {
    const matrix = DOT_MATRIX[char];
    if (!matrix) {
      offsetX += 3 * (dotSize + gap) + charGap;
      continue;
    }
    for (let row = 0; row < 7; row++) {
      for (let col = 0; col < 5; col++) {
        if (matrix[row] & (1 << (4 - col))) {
          dots.push({
            x: offsetX + col * (dotSize + gap),
            y: row * (dotSize + gap),
            index: dotIndex++,
          });
        }
      }
    }
    offsetX += 5 * (dotSize + gap) + charGap;
  }

  const totalWidth = offsetX - charGap;
  const totalHeight = 7 * (dotSize + gap) - gap;

  return (
    <div
      style={{
        position: 'relative',
        width: totalWidth,
        height: totalHeight,
      }}
    >
      {dots.map((dot) => {
        const dotDelay = 5 + dot.index * 1.5;
        const progress = spring({
          frame: frame - dotDelay,
          fps,
          config: { damping: 14, stiffness: 120, mass: 0.5 },
        });
        const opacity = interpolate(progress, [0, 1], [0, 1]);
        const scale = interpolate(progress, [0, 1], [0, 1]);

        return (
          <div
            key={dot.index}
            style={{
              position: 'absolute',
              left: dot.x,
              top: dot.y,
              width: dotSize,
              height: dotSize,
              borderRadius: '50%',
              backgroundColor: brand.accentColor,
              opacity,
              transform: `scale(${scale})`,
            }}
          />
        );
      })}
    </div>
  );
};

/* ── Bar chart visualization ── */
const BarChartViz: React.FC<{ data: Record<string, any>; brand: BrandTheme }> = ({
  data,
  brand,
}) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const bars: Array<{ label: string; value: number; accent?: boolean }> = data.bars ?? [];
  const maxValue = Math.max(...bars.map((b) => b.value), 1);
  const barHeight = 40;
  const barGap = 16;

  return (
    <div style={{ width: 700, display: 'flex', flexDirection: 'column', gap: barGap }}>
      {bars.map((bar, i) => {
        const barDelay = 8 + i * 10;
        const widthProgress = interpolate(frame, [barDelay, barDelay + 25], [0, 1], {
          extrapolateLeft: 'clamp',
          extrapolateRight: 'clamp',
        });
        const labelProgress = spring({
          frame: frame - barDelay,
          fps,
          config: { damping: 15, stiffness: 100, mass: 0.8 },
        });
        const labelOpacity = interpolate(labelProgress, [0, 1], [0, 1]);

        const barWidthPct = (bar.value / maxValue) * 100 * widthProgress;
        const barColor = bar.accent ? brand.accentColor : brand.primaryColor;

        return (
          <div key={i} style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
            <div
              style={{
                fontSize: 16,
                fontWeight: 600,
                color: brand.textColor,
                opacity: labelOpacity,
              }}
            >
              {bar.label}
            </div>
            <div
              style={{
                height: barHeight,
                borderRadius: barHeight / 2,
                backgroundColor: barColor,
                width: `${barWidthPct}%`,
                minWidth: barWidthPct > 0 ? 20 : 0,
              }}
            />
          </div>
        );
      })}
    </div>
  );
};

/* ── Dot grid visualization ── */
const DotGridViz: React.FC<{ data: Record<string, any>; brand: BrandTheme }> = ({
  data,
  brand,
}) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const count = data.count ?? 50;
  const dotSize = 8;

  // Generate deterministic positions using golden angle
  const dots = React.useMemo(() => {
    const result: Array<{ x: number; y: number }> = [];
    for (let i = 0; i < count; i++) {
      const angle = i * 137.508;
      const r = Math.sqrt(i / count) * 350;
      result.push({
        x: 400 + r * Math.cos((angle * Math.PI) / 180),
        y: 300 + r * Math.sin((angle * Math.PI) / 180),
      });
    }
    return result;
  }, [count]);

  return (
    <div style={{ position: 'relative', width: 800, height: 600 }}>
      {dots.map((dot, i) => {
        const dotDelay = 3 + i * 0.8;
        const progress = spring({
          frame: frame - dotDelay,
          fps,
          config: { damping: 14, stiffness: 100, mass: 0.7 },
        });
        const opacity = interpolate(progress, [0, 1], [0, 1]);
        const scale = interpolate(progress, [0, 1], [0, 1]);

        return (
          <div
            key={i}
            style={{
              position: 'absolute',
              left: dot.x,
              top: dot.y,
              width: dotSize,
              height: dotSize,
              borderRadius: '50%',
              backgroundColor: brand.accentColor,
              opacity,
              transform: `scale(${scale})`,
            }}
          />
        );
      })}
    </div>
  );
};

/* ── Main DataVizScene ── */
export const DataVizScene: React.FC<Props> = ({ vizType, data, background, brand }) => {
  return (
    <AbsoluteFill>
      {renderBackground(background, brand)}
      <AbsoluteFill
        style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          fontFamily: brand.fontFamily,
        }}
      >
        {vizType === 'bracket' && <BracketViz data={data} brand={brand} />}
        {vizType === 'dot_matrix_number' && <DotMatrixViz data={data} brand={brand} />}
        {vizType === 'bar_chart' && <BarChartViz data={data} brand={brand} />}
        {vizType === 'dot_grid' && <DotGridViz data={data} brand={brand} />}
      </AbsoluteFill>
    </AbsoluteFill>
  );
};
