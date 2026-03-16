import React from 'react';
import { AbsoluteFill } from 'remotion';

interface Props {
  backgroundColor?: string;
  showBorder?: boolean;
}

export const CleanBg: React.FC<Props> = ({
  backgroundColor = '#f5f5f5',
  showBorder = true,
}) => {
  // Detect if bg is dark to skip the border
  const clean = backgroundColor.replace('#', '');
  const r = parseInt(clean.substring(0, 2), 16) || 0;
  const g = parseInt(clean.substring(2, 4), 16) || 0;
  const b = parseInt(clean.substring(4, 6), 16) || 0;
  const isDark = (0.299 * r + 0.587 * g + 0.114 * b) / 255 < 0.5;

  return (
    <AbsoluteFill>
      {/* Base off-white background */}
      <div
        style={{
          position: 'absolute',
          inset: 0,
          backgroundColor,
        }}
      />
      {/* Optional thin border line at edges — only visible on light backgrounds */}
      {showBorder && !isDark && (
        <div
          style={{
            position: 'absolute',
            inset: 24,
            border: '1px solid rgba(0, 0, 0, 0.06)',
            borderRadius: 4,
            pointerEvents: 'none',
          }}
        />
      )}
    </AbsoluteFill>
  );
};
