import React from 'react';
import { AbsoluteFill } from 'remotion';

interface Props {
  backgroundColor?: string;
  dotColor?: string;
  dotSize?: number;
  gridSpacing?: number;
}

export const DotsBg: React.FC<Props> = ({
  backgroundColor = '#f5f5f5',
  dotColor = 'rgba(208, 208, 208, 0.3)',
  dotSize = 4,
  gridSpacing = 24,
}) => {
  const halfDot = dotSize / 2;

  return (
    <AbsoluteFill>
      {/* Base background */}
      <div
        style={{
          position: 'absolute',
          inset: 0,
          backgroundColor,
        }}
      />
      {/* Repeating dot pattern via CSS radial-gradient */}
      <div
        style={{
          position: 'absolute',
          inset: 0,
          backgroundImage: `radial-gradient(circle, ${dotColor} ${halfDot}px, transparent ${halfDot}px)`,
          backgroundSize: `${gridSpacing}px ${gridSpacing}px`,
          backgroundPosition: `${gridSpacing / 2}px ${gridSpacing / 2}px`,
        }}
      />
    </AbsoluteFill>
  );
};
