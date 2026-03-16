import React from 'react';
import { AbsoluteFill } from 'remotion';
import { AnimatedText } from '../components/AnimatedText';
import { GradientBg } from '../components/GradientBg';
import { BrandTheme } from '../lib/types';

interface Props {
  label: string;
  headline: string;
  brand: BrandTheme;
}

export const TitleScene: React.FC<Props> = ({ label, headline, brand }) => {
  return (
    <AbsoluteFill>
      <GradientBg backgroundColor={brand.backgroundColor} accentColor={brand.accentColor} />
      <AbsoluteFill style={{
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        fontFamily: brand.fontFamily,
        gap: 16,
      }}>
        <AnimatedText
          text={label}
          fontSize={28}
          color="rgba(255,255,255,0.5)"
          fontWeight={500}
          delay={0}
          style={{ letterSpacing: 6, textTransform: 'uppercase' }}
        />
        <AnimatedText
          text={headline}
          fontSize={80}
          color={brand.primaryColor}
          fontWeight={800}
          delay={8}
        />
      </AbsoluteFill>
    </AbsoluteFill>
  );
};
