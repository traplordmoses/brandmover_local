import React from 'react';
import { AbsoluteFill, useCurrentFrame, interpolate, spring, useVideoConfig } from 'remotion';
import { AnimatedText } from '../components/AnimatedText';
import { BrandTheme } from '../lib/types';
import { renderBackground } from '../lib/backgrounds';

interface Props {
  title?: string;
  items: Array<{ text: string; accent?: boolean }>;
  layout: 'centered-stack' | 'left-aligned';
  background?: string;
  brand: BrandTheme;
}

const STAGGER_FRAMES = 8;

export const FeatureListScene: React.FC<Props> = ({
  title,
  items,
  layout,
  background,
  brand,
}) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  const isCentered = layout === 'centered-stack';
  const titleDelay = 0;
  const firstItemDelay = title ? 6 : 0;

  return (
    <AbsoluteFill>
      {renderBackground(background, brand)}
      <AbsoluteFill
        style={{
          display: 'flex',
          flexDirection: 'column',
          alignItems: isCentered ? 'center' : 'flex-start',
          justifyContent: 'center',
          fontFamily: brand.fontFamily,
          padding: isCentered ? 60 : '60px 100px',
          gap: 20,
        }}
      >
        {/* Title */}
        {title && (
          <AnimatedText
            text={title}
            fontSize={24}
            color={brand.textColor}
            fontWeight={500}
            delay={titleDelay}
            style={{
              opacity: 0.5,
              letterSpacing: 3,
              textTransform: 'uppercase',
              marginBottom: 16,
            }}
          />
        )}

        {/* Items */}
        {items.map((item, i) => {
          const itemDelay = firstItemDelay + i * STAGGER_FRAMES;
          const progress = spring({
            frame: frame - itemDelay,
            fps,
            config: { damping: 15, stiffness: 100, mass: 0.8 },
          });
          const opacity = interpolate(progress, [0, 1], [0, 1]);
          const translateY = interpolate(progress, [0, 1], [24, 0]);

          const displayText = isCentered ? item.text : `— ${item.text}`;

          return (
            <div
              key={i}
              style={{
                fontSize: items.length > 4 ? 28 : 36,
                fontWeight: item.accent ? 700 : 500,
                color: item.accent ? brand.accentColor : brand.textColor,
                opacity,
                transform: `translateY(${translateY}px)`,
                lineHeight: 1.4,
                textAlign: isCentered ? 'center' : 'left',
              }}
            >
              {displayText}
            </div>
          );
        })}
      </AbsoluteFill>
    </AbsoluteFill>
  );
};
