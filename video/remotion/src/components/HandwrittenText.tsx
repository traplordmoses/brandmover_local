import React from 'react';
import { useCurrentFrame, interpolate, spring, useVideoConfig } from 'remotion';

interface Props {
  text: string;
  fontSize?: number;
  color?: string;
  delay?: number;
  fontFamily: string;
  style?: React.CSSProperties;
}

export const HandwrittenText: React.FC<Props> = ({
  text,
  fontSize = 48,
  color = '#ffffff',
  delay = 0,
  fontFamily,
  style = {},
}) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  const progress = spring({
    frame: frame - delay,
    fps,
    config: { damping: 14, stiffness: 100, mass: 0.7 },
  });

  const opacity = interpolate(progress, [0, 1], [0, 1]);
  const translateY = interpolate(progress, [0, 1], [24, 0]);

  return (
    <div
      style={{
        fontSize,
        fontFamily,
        color,
        fontWeight: 400,
        opacity,
        transform: `translateY(${translateY}px)`,
        lineHeight: 1.3,
        ...style,
      }}
    >
      {text}
    </div>
  );
};
