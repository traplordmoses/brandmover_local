import React from 'react';
import { AbsoluteFill, useCurrentFrame, interpolate, spring, useVideoConfig } from 'remotion';
import { AnimatedText } from '../components/AnimatedText';
import { AnimatedCounter } from '../components/AnimatedCounter';
import { BrandTheme } from '../lib/types';
import { renderBackground } from '../lib/backgrounds';

interface Props {
  prefix?: string;
  value: string;
  suffix?: string;
  suffixStyle?: 'normal' | 'handwritten';
  rawNumber?: string;
  animate?: 'countUp' | 'fadeIn';
  background?: string;
  brand: BrandTheme;
}

function isNumeric(s: string): boolean {
  return !isNaN(parseFloat(s.replace(/[,$%]/g, '')));
}

export const StatScene: React.FC<Props> = ({
  prefix,
  value,
  suffix,
  suffixStyle = 'normal',
  rawNumber,
  animate = 'fadeIn',
  background,
  brand,
}) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  const numericValue = parseFloat(value.replace(/[,$%]/g, ''));
  const shouldCountUp = animate === 'countUp' && isNumeric(value);

  // Raw number animation
  const rawProgress = spring({
    frame,
    fps,
    config: { damping: 15, stiffness: 100, mass: 0.8 },
  });
  const rawOpacity = interpolate(rawProgress, [0, 1], [0, 1]);
  const rawY = interpolate(rawProgress, [0, 1], [20, 0]);

  // Suffix animation (delayed)
  const suffixProgress = spring({
    frame: frame - 8,
    fps,
    config: { damping: 14, stiffness: 100, mass: 0.7 },
  });
  const suffixOpacity = interpolate(suffixProgress, [0, 1], [0, 1]);
  const suffixY = interpolate(suffixProgress, [0, 1], [16, 0]);

  return (
    <AbsoluteFill>
      {renderBackground(background, brand)}
      <AbsoluteFill
        style={{
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          justifyContent: 'center',
          fontFamily: brand.fontFamily,
          gap: 12,
        }}
      >
        {/* Raw number (small, muted) */}
        {rawNumber && (
          <div
            style={{
              fontSize: 22,
              fontWeight: 500,
              color: brand.textColor,
              opacity: rawOpacity * 0.5,
              transform: `translateY(${rawY}px)`,
              letterSpacing: 2,
            }}
          >
            {rawNumber}
          </div>
        )}

        {/* Main value */}
        {shouldCountUp ? (
          <AnimatedCounter
            value={numericValue}
            prefix={prefix}
            fontSize={120}
            color={brand.accentColor}
            delay={5}
            accentFontFamily={brand.accentFontFamily}
          />
        ) : (
          <div style={{ display: 'flex', alignItems: 'baseline', gap: 8 }}>
            {prefix && (
              <AnimatedText
                text={prefix}
                fontSize={72}
                color={brand.accentColor}
                fontWeight={700}
                delay={0}
              />
            )}
            <AnimatedText
              text={value}
              fontSize={120}
              color={brand.accentColor}
              fontWeight={900}
              delay={5}
            />
          </div>
        )}

        {/* Suffix */}
        {suffix && (
          <div
            style={{
              fontSize: suffixStyle === 'handwritten' ? 40 : 36,
              fontWeight: suffixStyle === 'handwritten' ? 400 : 600,
              color: brand.textColor,
              opacity: suffixOpacity,
              transform: `translateY(${suffixY}px)`,
              fontFamily: suffixStyle === 'handwritten' && brand.accentFontFamily
                ? brand.accentFontFamily
                : brand.fontFamily,
            }}
          >
            {suffix}
          </div>
        )}
      </AbsoluteFill>
    </AbsoluteFill>
  );
};
