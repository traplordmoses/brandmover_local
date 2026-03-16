import React from 'react';
import { useCurrentFrame, interpolate, spring, useVideoConfig } from 'remotion';

interface Props {
  text: string;
  fontSize?: number;
  color?: string;
  fontWeight?: number;
  delay?: number;
  style?: React.CSSProperties;
}

export const AnimatedText: React.FC<Props> = ({
  text,
  fontSize = 64,
  color = '#ffffff',
  fontWeight = 700,
  delay = 0,
  style = {},
}) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  const progress = spring({
    frame: frame - delay,
    fps,
    config: { damping: 15, stiffness: 100, mass: 0.8 },
  });

  const opacity = interpolate(progress, [0, 1], [0, 1]);
  const translateY = interpolate(progress, [0, 1], [30, 0]);

  return (
    <div
      style={{
        fontSize,
        fontWeight,
        color,
        opacity,
        transform: `translateY(${translateY}px)`,
        lineHeight: 1.2,
        ...style,
      }}
    >
      {text}
    </div>
  );
};
