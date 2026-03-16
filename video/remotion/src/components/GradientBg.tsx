import React from 'react';
import { AbsoluteFill } from 'remotion';

interface Props {
  backgroundColor: string;
  accentColor: string;
}

export const GradientBg: React.FC<Props> = ({ backgroundColor, accentColor }) => {
  // Create glow layers — stronger than before for cinematic depth
  const glowStrong = accentColor + '30'; // ~19% opacity
  const glowMedium = accentColor + '1a'; // ~10% opacity
  const glowSoft = accentColor + '0d';   // ~5% opacity

  // Brighten the base bg slightly for a subtle inner glow effect
  const bgLighter = backgroundColor + 'cc'; // Used in radial center

  return (
    <AbsoluteFill>
      {/* Base background */}
      <div style={{
        position: 'absolute',
        inset: 0,
        backgroundColor,
      }} />
      {/* Center radial glow — gives depth like underwater light */}
      <div style={{
        position: 'absolute',
        top: '10%',
        left: '10%',
        width: '80%',
        height: '80%',
        borderRadius: '50%',
        background: `radial-gradient(ellipse, ${glowSoft} 0%, transparent 70%)`,
        filter: 'blur(40px)',
      }} />
      {/* Aurora glow top-right — accent color */}
      <div style={{
        position: 'absolute',
        top: '-20%',
        right: '-15%',
        width: '75%',
        height: '75%',
        borderRadius: '50%',
        background: `radial-gradient(circle, ${glowStrong} 0%, ${glowMedium} 35%, transparent 70%)`,
        filter: 'blur(60px)',
      }} />
      {/* Bottom-left glow — secondary warmth */}
      <div style={{
        position: 'absolute',
        bottom: '-25%',
        left: '-15%',
        width: '65%',
        height: '65%',
        borderRadius: '50%',
        background: `radial-gradient(circle, ${glowMedium} 0%, ${glowSoft} 30%, transparent 60%)`,
        filter: 'blur(80px)',
      }} />
    </AbsoluteFill>
  );
};
