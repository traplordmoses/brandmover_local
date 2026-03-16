import React from 'react';
import { AbsoluteFill } from 'remotion';
import { AnimatedText } from '../components/AnimatedText';
import { KineticText } from '../components/KineticText';
import { GradientBg } from '../components/GradientBg';
import { BrandTheme } from '../lib/types';

interface Props {
  supertext?: string;
  lines: Array<{ text: string; accent?: boolean; style?: string }>;
  brand: BrandTheme;
}

export const TaglineScene: React.FC<Props> = ({ supertext, lines, brand }) => {
  // Calculate cumulative word count for stagger offset
  let wordOffset = 0;

  return (
    <AbsoluteFill>
      <GradientBg backgroundColor={brand.backgroundColor} accentColor={brand.accentColor} />
      <AbsoluteFill style={{
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        fontFamily: brand.fontFamily,
        gap: 8,
      }}>
        {supertext && (
          <AnimatedText
            text={supertext}
            fontSize={18}
            color="rgba(255,255,255,0.4)"
            fontWeight={500}
            delay={0}
            style={{ letterSpacing: 4, textTransform: 'uppercase', marginBottom: 8 }}
          />
        )}
        {lines.map((line, i) => {
          const lineDelay = (supertext ? 5 : 0) + wordOffset * 3;
          const wordCount = line.text.split(/\s+/).filter(Boolean).length;
          wordOffset += wordCount;

          return (
            <KineticText
              key={i}
              text={line.text}
              fontSize={52}
              color={line.accent ? brand.accentColor : (brand.textColor || '#ffffff')}
              fontWeight={line.style === 'bold' ? 900 : 700}
              delay={lineDelay}
              wordStagger={3}
              entrance={line.accent ? 'scale' : 'rise'}
            />
          );
        })}
      </AbsoluteFill>
    </AbsoluteFill>
  );
};
