import React from 'react';
import { AbsoluteFill, useCurrentFrame, interpolate, spring, useVideoConfig } from 'remotion';
import { GradientBg } from '../components/GradientBg';
import { BrandTheme } from '../lib/types';

interface Props {
  count: number;
  subtitle: string;
  brand: BrandTheme;
}

export const FeatureCountScene: React.FC<Props> = ({ count, subtitle, brand }) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  const numProgress = spring({ frame, fps, config: { damping: 15, stiffness: 80, mass: 1 } });
  const textProgress = spring({ frame: frame - 15, fps, config: { damping: 15, stiffness: 100, mass: 0.8 } });

  const numOpacity = interpolate(numProgress, [0, 1], [0, 1]);
  const numScale = interpolate(numProgress, [0, 1], [0.5, 1]);
  const textOpacity = interpolate(textProgress, [0, 1], [0, 1]);
  const textY = interpolate(textProgress, [0, 1], [20, 0]);

  return (
    <AbsoluteFill>
      <GradientBg backgroundColor={brand.backgroundColor} accentColor={brand.accentColor} />
      <AbsoluteFill style={{
        display: 'flex',
        flexDirection: 'row',
        alignItems: 'center',
        justifyContent: 'center',
        fontFamily: brand.fontFamily,
        gap: 16,
      }}>
        <span style={{
          fontSize: 140,
          fontWeight: 900,
          color: brand.accentColor,
          opacity: numOpacity,
          transform: `scale(${numScale})`,
          lineHeight: 1,
        }}>
          {count}
        </span>
        <span style={{
          fontSize: 42,
          fontWeight: 500,
          color: '#ffffff',
          opacity: textOpacity,
          transform: `translateY(${textY}px)`,
          maxWidth: 400,
          lineHeight: 1.3,
        }}>
          {subtitle}
        </span>
      </AbsoluteFill>
    </AbsoluteFill>
  );
};
