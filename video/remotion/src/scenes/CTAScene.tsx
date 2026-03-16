import React from 'react';
import { AbsoluteFill } from 'remotion';
import { AnimatedText } from '../components/AnimatedText';
import { BrandButton } from '../components/BrandButton';
import { GradientBg } from '../components/GradientBg';
import { BrandTheme } from '../lib/types';

interface Props {
  line1: string;
  line2: string;
  accentLine: 1 | 2;
  url: string;
  buttonText: string;
  brand: BrandTheme;
}

export const CTAScene: React.FC<Props> = ({ line1, line2, accentLine, url, buttonText, brand }) => {
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
          text={line1}
          fontSize={48}
          color={accentLine === 1 ? brand.accentColor : '#ffffff'}
          fontWeight={700}
          delay={0}
        />
        <AnimatedText
          text={line2}
          fontSize={48}
          color={accentLine === 2 ? brand.accentColor : '#ffffff'}
          fontWeight={700}
          delay={8}
        />
        <div style={{ marginTop: 32 }}>
          <BrandButton text={buttonText} color={brand.accentColor} delay={20} />
        </div>
        <AnimatedText
          text={url}
          fontSize={18}
          color="rgba(255,255,255,0.4)"
          fontWeight={400}
          delay={28}
          style={{ marginTop: 16 }}
        />
      </AbsoluteFill>
    </AbsoluteFill>
  );
};
