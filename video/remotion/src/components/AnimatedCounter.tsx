import React from 'react';
import { useCurrentFrame, interpolate, useVideoConfig } from 'remotion';

interface Props {
  value: number;
  prefix?: string;
  suffix?: string;
  fontSize?: number;
  color?: string;
  delay?: number;
  /** Fraction of the scene duration to count over (0-1). Default 0.6 */
  duration?: number;
  accentFontFamily?: string;
  suffixStyle?: 'normal' | 'handwritten';
}

function formatWithCommas(n: number): string {
  return Math.round(n).toLocaleString('en-US');
}

export const AnimatedCounter: React.FC<Props> = ({
  value,
  prefix = '',
  suffix = '',
  fontSize = 120,
  color = '#ffffff',
  delay = 0,
  duration = 0.6,
  accentFontFamily,
  suffixStyle = 'normal',
}) => {
  const frame = useCurrentFrame();
  const { durationInFrames } = useVideoConfig();

  const countEndFrame = delay + Math.round(durationInFrames * duration);

  const currentValue = interpolate(
    frame,
    [delay, countEndFrame],
    [0, value],
    { extrapolateLeft: 'clamp', extrapolateRight: 'clamp' },
  );

  const opacity = interpolate(
    frame,
    [delay, delay + 6],
    [0, 1],
    { extrapolateLeft: 'clamp', extrapolateRight: 'clamp' },
  );

  return (
    <div style={{ opacity, display: 'flex', alignItems: 'baseline', gap: 8 }}>
      {prefix && (
        <span style={{ fontSize: fontSize * 0.6, fontWeight: 700, color }}>
          {prefix}
        </span>
      )}
      <span style={{ fontSize, fontWeight: 900, color, lineHeight: 1 }}>
        {formatWithCommas(currentValue)}
      </span>
      {suffix && (
        <span
          style={{
            fontSize: suffixStyle === 'handwritten' ? fontSize * 0.5 : fontSize * 0.4,
            fontWeight: suffixStyle === 'handwritten' ? 400 : 600,
            color,
            fontFamily: suffixStyle === 'handwritten' && accentFontFamily
              ? accentFontFamily
              : undefined,
            marginLeft: 4,
          }}
        >
          {suffix}
        </span>
      )}
    </div>
  );
};
