import React from 'react';
import { AbsoluteFill } from 'remotion';
import { AnimatedText } from '../components/AnimatedText';
import { GradientBg } from '../components/GradientBg';
import { BrandTheme } from '../lib/types';

interface Props {
  supertext?: string;
  lines: Array<{ text: string; accent?: boolean; style?: string }>;
  brand: BrandTheme;
}

export const TaglineScene: React.FC<Props> = ({ supertext, lines, brand }) => {
  return (
    <AbsoluteFill>
      <GradientBg backgroundColor={brand.backgroundColor} accentColor={brand.accentColor} />
      <AbsoluteFill style={{
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        fontFamily: brand.fontFamily,
        gap: 12,
      }}>
        {supertext && (
          <AnimatedText
            text={supertext}
            fontSize={18}
            color="rgba(255,255,255,0.4)"
            fontWeight={500}
            delay={0}
            style={{ letterSpacing: 4, textTransform: 'uppercase' }}
          />
        )}
        {lines.map((line, i) => (
          <AnimatedText
            key={i}
            text={line.text}
            fontSize={52}
            color={line.accent ? brand.accentColor : (brand.textColor || '#ffffff')}
            fontWeight={line.style === 'bold' ? 900 : 700}
            delay={(supertext ? 5 : 0) + i * 5}
            style={line.style === 'handwritten' ? { fontStyle: 'italic' } : undefined}
          />
        ))}
      </AbsoluteFill>
    </AbsoluteFill>
  );
};
