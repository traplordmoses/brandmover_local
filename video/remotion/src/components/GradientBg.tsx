import React from 'react';
import { AbsoluteFill } from 'remotion';

interface Props {
  backgroundColor: string;
  accentColor: string;
}

export const GradientBg: React.FC<Props> = ({ backgroundColor, accentColor }) => {
  // Parse accent color and create a low-opacity version
  const glowColor = accentColor + '15'; // ~8% opacity
  const glowColor2 = accentColor + '0a'; // ~4% opacity

  return (
    <AbsoluteFill>
      {/* Base dark background */}
      <div style={{
        position: 'absolute',
        inset: 0,
        backgroundColor,
      }} />
      {/* Aurora glow top-right */}
      <div style={{
        position: 'absolute',
        top: '-20%',
        right: '-20%',
        width: '80%',
        height: '80%',
        borderRadius: '50%',
        background: `radial-gradient(circle, ${glowColor} 0%, ${glowColor2} 40%, transparent 70%)`,
        filter: 'blur(60px)',
      }} />
      {/* Subtle bottom-left glow */}
      <div style={{
        position: 'absolute',
        bottom: '-30%',
        left: '-20%',
        width: '70%',
        height: '70%',
        borderRadius: '50%',
        background: `radial-gradient(circle, ${glowColor2} 0%, transparent 60%)`,
        filter: 'blur(80px)',
      }} />
    </AbsoluteFill>
  );
};
