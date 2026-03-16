import React from 'react';
import { useCurrentFrame, interpolate, spring, useVideoConfig } from 'remotion';

interface Props {
  text: string;
  color: string;
  delay?: number;
}

export const BrandButton: React.FC<Props> = ({ text, color, delay = 0 }) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  const progress = spring({
    frame: frame - delay,
    fps,
    config: { damping: 12, stiffness: 100, mass: 0.8 },
  });

  const opacity = interpolate(progress, [0, 1], [0, 1]);
  const scale = interpolate(progress, [0, 1], [0.9, 1]);

  return (
    <div
      style={{
        display: 'inline-flex',
        alignItems: 'center',
        justifyContent: 'center',
        padding: '16px 40px',
        borderRadius: 50,
        background: color,
        color: '#000000',
        fontSize: 22,
        fontWeight: 700,
        opacity,
        transform: `scale(${scale})`,
        letterSpacing: 0.5,
      }}
    >
      {text}
    </div>
  );
};
