import React from 'react';
import { AbsoluteFill } from 'remotion';
import { AnimatedText } from '../components/AnimatedText';
import { KineticText } from '../components/KineticText';
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
  // Calculate total words for button delay
  const totalWords = lines.reduce((sum, l) => sum + l.text.split(/\s+/).filter(Boolean).length, 0);

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
        {lines.map((line, i) => {
          const wordsBeforeThis = lines.slice(0, i).reduce(
            (sum, l) => sum + l.text.split(/\s+/).filter(Boolean).length, 0
          );
          return (
            <KineticText
              key={i}
              text={line.text}
              fontSize={56}
              color={line.accent ? brand.accentColor : (brand.textColor || '#ffffff')}
              fontWeight={700}
              delay={wordsBeforeThis * 3}
              wordStagger={3}
              entrance={line.accent ? 'scale' : 'rise'}
            />
          );
        })}
        {buttonText && (
          <div style={{ marginTop: 32 }}>
            <BrandButton text={buttonText} color={brand.accentColor} delay={totalWords * 3 + 8} />
          </div>
        )}
        {url && (
          <AnimatedText
            text={url}
            fontSize={18}
            color="rgba(255,255,255,0.4)"
            fontWeight={400}
            delay={totalWords * 3 + 16}
            style={{ marginTop: 16 }}
          />
        )}
      </AbsoluteFill>
    </AbsoluteFill>
  );
};
