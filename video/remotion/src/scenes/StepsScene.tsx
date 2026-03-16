import React from 'react';
import { AbsoluteFill, useCurrentFrame, spring, interpolate, useVideoConfig } from 'remotion';
import { AnimatedText } from '../components/AnimatedText';
import { GradientBg } from '../components/GradientBg';
import { BrandTheme } from '../lib/types';

interface Step {
  number: string;
  heading: string;
  detail: string;
}

interface Props {
  title?: string;
  steps: Step[];
  brand: BrandTheme;
}

export const StepsScene: React.FC<Props> = ({ title, steps, brand }) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  return (
    <AbsoluteFill>
      <GradientBg backgroundColor={brand.backgroundColor} accentColor={brand.accentColor} />
      <AbsoluteFill style={{
        display: 'flex',
        flexDirection: 'column',
        padding: '60px 50px',
        fontFamily: brand.fontFamily,
        gap: 32,
      }}>
        {title && (
          <AnimatedText
            text={title}
            fontSize={28}
            color="rgba(255,255,255,0.5)"
            fontWeight={600}
            delay={0}
            style={{ letterSpacing: 4, textTransform: 'uppercase' }}
          />
        )}
        <div style={{ display: 'flex', flexDirection: 'column', gap: 24 }}>
          {steps.map((step, i) => {
            const delay = i * 18 + 12;
            const progress = spring({
              frame: frame - delay,
              fps,
              config: { damping: 14, stiffness: 100, mass: 0.8 },
            });
            const opacity = interpolate(progress, [0, 1], [0, 1]);
            const translateX = interpolate(progress, [0, 1], [30, 0]);

            return (
              <div
                key={i}
                style={{
                  display: 'flex',
                  alignItems: 'flex-start',
                  gap: 20,
                  opacity,
                  transform: `translateX(${translateX}px)`,
                }}
              >
                <span style={{
                  fontSize: 48,
                  fontWeight: 900,
                  color: brand.accentColor,
                  lineHeight: 1,
                  minWidth: 60,
                }}>
                  {step.number}
                </span>
                <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
                  <span style={{ fontSize: 28, fontWeight: 700, color: '#ffffff' }}>
                    {step.heading}
                  </span>
                  <span style={{ fontSize: 20, color: 'rgba(255,255,255,0.5)', lineHeight: 1.4 }}>
                    {step.detail}
                  </span>
                </div>
              </div>
            );
          })}
        </div>
      </AbsoluteFill>
    </AbsoluteFill>
  );
};
