import React from 'react';
import { AbsoluteFill } from 'remotion';
import { AnimatedText } from '../components/AnimatedText';
import { BrandButton } from '../components/BrandButton';
import { GradientBg } from '../components/GradientBg';
import { BrandTheme } from '../lib/types';

interface Props {
  lines: Array<{ text: string; accent?: boolean }>;
  url?: string;
  buttonText?: string;
  brand: BrandTheme;
}

export const CTAScene: React.FC<Props> = ({ lines, url, buttonText, brand }) => {
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
        {lines.map((line, i) => (
          <AnimatedText
            key={i}
            text={line.text}
            fontSize={48}
            color={line.accent ? brand.accentColor : (brand.textColor || '#ffffff')}
            fontWeight={700}
            delay={i * 8}
          />
        ))}
        {buttonText && (
          <div style={{ marginTop: 32 }}>
            <BrandButton text={buttonText} color={brand.accentColor} delay={lines.length * 8 + 12} />
          </div>
        )}
        {url && (
          <AnimatedText
            text={url}
            fontSize={18}
            color="rgba(255,255,255,0.4)"
            fontWeight={400}
            delay={lines.length * 8 + 20}
            style={{ marginTop: 16 }}
          />
        )}
      </AbsoluteFill>
    </AbsoluteFill>
  );
};
