import React from 'react';
import { useCurrentFrame, interpolate, spring, useVideoConfig } from 'remotion';

interface Props {
  text: string;
  fontSize?: number;
  color?: string;
  fontWeight?: number;
  delay?: number;
  /** Frames between each word appearing */
  wordStagger?: number;
  style?: React.CSSProperties;
  /** Animation style — 'rise' slides up, 'drop' slides down, 'scale' pops in */
  entrance?: 'rise' | 'drop' | 'scale';
}

/**
 * Word-by-word kinetic text reveal. Each word animates in independently
 * with spring physics, creating dramatic staggered reveals.
 */
export const KineticText: React.FC<Props> = ({
  text,
  fontSize = 64,
  color = '#ffffff',
  fontWeight = 700,
  delay = 0,
  wordStagger = 3,
  style = {},
  entrance = 'rise',
}) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  const words = text.split(/\s+/).filter(Boolean);

  return (
    <div
      style={{
        display: 'flex',
        flexWrap: 'wrap',
        justifyContent: 'center',
        alignItems: 'baseline',
        gap: `0px ${Math.max(fontSize * 0.22, 8)}px`,
        lineHeight: 1.15,
        ...style,
      }}
    >
      {words.map((word, i) => {
        const wordDelay = delay + i * wordStagger;

        const progress = spring({
          frame: frame - wordDelay,
          fps,
          config: { damping: 18, stiffness: 140, mass: 0.6 },
        });

        const opacity = interpolate(progress, [0, 1], [0, 1]);

        let transform: string;
        if (entrance === 'drop') {
          const y = interpolate(progress, [0, 1], [-20, 0]);
          transform = `translateY(${y}px)`;
        } else if (entrance === 'scale') {
          const s = interpolate(progress, [0, 1], [0.7, 1]);
          const y = interpolate(progress, [0, 1], [8, 0]);
          transform = `scale(${s}) translateY(${y}px)`;
        } else {
          // 'rise' — default
          const y = interpolate(progress, [0, 1], [24, 0]);
          transform = `translateY(${y}px)`;
        }

        return (
          <span
            key={i}
            style={{
              fontSize,
              fontWeight,
              color,
              opacity,
              transform,
              display: 'inline-block',
              willChange: 'transform, opacity',
            }}
          >
            {word}
          </span>
        );
      })}
    </div>
  );
};
