import React from 'react';
import { AbsoluteFill } from 'remotion';
import { AnimatedText } from '../components/AnimatedText';
import { GradientBg } from '../components/GradientBg';
import { BrandTheme } from '../lib/types';

interface Props {
  label?: string;
  headline: string;
  subheadline?: string;
  disclaimer?: string;
  brand: BrandTheme;
}

export const TitleScene: React.FC<Props> = ({ label, headline, subheadline, disclaimer, brand }) => {
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
        {label && (
          <AnimatedText
            text={label}
            fontSize={28}
            color="rgba(255,255,255,0.5)"
            fontWeight={500}
            delay={0}
            style={{ letterSpacing: 6, textTransform: 'uppercase' }}
          />
        )}
        <AnimatedText
          text={headline}
          fontSize={80}
          color={brand.textColor || '#ffffff'}
          fontWeight={800}
          delay={label ? 8 : 0}
        />
        {subheadline && (
          <AnimatedText
            text={subheadline}
            fontSize={32}
            color="rgba(255,255,255,0.6)"
            fontWeight={400}
            delay={label ? 16 : 8}
          />
        )}
        {disclaimer && (
          <div style={{
            position: 'absolute',
            bottom: 40,
            left: 0,
            right: 0,
            textAlign: 'center',
          }}>
            <AnimatedText
              text={disclaimer}
              fontSize={14}
              color="rgba(255,255,255,0.3)"
              fontWeight={400}
              delay={label ? 24 : 16}
            />
          </div>
        )}
      </AbsoluteFill>
    </AbsoluteFill>
  );
};
