import React from 'react';
import { AbsoluteFill } from 'remotion';
import { AnimatedText } from '../components/AnimatedText';
import { GradientBg } from '../components/GradientBg';
import { BrandTheme } from '../lib/types';

interface Props {
  supertext: string;
  line1: string;
  line2: string;
  accentLine: 1 | 2;
  brand: BrandTheme;
}

export const TaglineScene: React.FC<Props> = ({ supertext, line1, line2, accentLine, brand }) => {
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
        <AnimatedText
          text={supertext}
          fontSize={18}
          color="rgba(255,255,255,0.4)"
          fontWeight={500}
          delay={0}
          style={{ letterSpacing: 4, textTransform: 'uppercase' }}
        />
        <AnimatedText
          text={line1}
          fontSize={52}
          color={accentLine === 1 ? brand.accentColor : '#ffffff'}
          fontWeight={700}
          delay={10}
        />
        <AnimatedText
          text={line2}
          fontSize={52}
          color={accentLine === 2 ? brand.accentColor : '#ffffff'}
          fontWeight={700}
          delay={18}
        />
      </AbsoluteFill>
    </AbsoluteFill>
  );
};
