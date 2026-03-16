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
      {/* Optional thin border line at edges */}
      {showBorder && (
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
