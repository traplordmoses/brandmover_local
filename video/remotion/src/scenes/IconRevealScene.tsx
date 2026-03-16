import React from 'react';
import { AbsoluteFill, useCurrentFrame, interpolate, spring, useVideoConfig } from 'remotion';
import { renderBackground } from '../lib/backgrounds';
import { BrandTheme } from '../lib/types';

interface Props {
  icons: Array<{ name: string; label?: string }>;
  caption?: string;
  layout: 'single_centered' | 'row';
  background?: string;
  brand: BrandTheme;
}

const ICON_EMOJI_MAP: Record<string, string> = {
  atom: '\u269B\uFE0F',
  globe: '\u{1F30D}',
  'file-text': '\u{1F4C4}',
  zap: '\u26A1',
  shield: '\u{1F6E1}\uFE0F',
  rocket: '\u{1F680}',
  wallet: '\u{1F45B}',
  lock: '\u{1F512}',
  users: '\u{1F465}',
  'chart-bar': '\u{1F4CA}',
  trophy: '\u{1F3C6}',
  star: '\u2B50',
  coin: '\u{1FA99}',
  layers: '\u{1F4DA}',
  'credit-card': '\u{1F4B3}',
  heart: '\u2764\uFE0F',
  fire: '\u{1F525}',
  lightning: '\u26A1',
  diamond: '\u{1F48E}',
  check: '\u2705',
};

function resolveIcon(name: string): string {
  if (/\p{Emoji}/u.test(name) && name.length <= 4) {
    return name;
  }
  return ICON_EMOJI_MAP[name.toLowerCase()] ?? '\u2B50';
}

export const IconRevealScene: React.FC<Props> = ({
  icons,
  caption,
  layout,
  background,
  brand,
}) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  const isSingle = layout === 'single_centered';
  const iconSize = isSingle ? 96 : 64;
  const labelSize = isSingle ? 32 : 22;
  const staggerDelay = 10;

  // Caption animation
  const captionDelay = 5 + icons.length * staggerDelay + 10;
  const captionProgress = spring({
    frame: frame - captionDelay,
    fps,
    config: { damping: 15, stiffness: 100, mass: 0.8 },
  });
  const captionOpacity = interpolate(captionProgress, [0, 1], [0, 1]);
  const captionY = interpolate(captionProgress, [0, 1], [20, 0]);

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
          gap: 32,
        }}
      >
        {/* Icons */}
        <div
          style={{
            display: 'flex',
            flexDirection: isSingle ? 'column' : 'row',
            alignItems: 'center',
            justifyContent: 'center',
            gap: isSingle ? 16 : 48,
          }}
        >
          {icons.map((icon, i) => {
            const iconDelay = 5 + i * staggerDelay;
            const progress = spring({
              frame: frame - iconDelay,
              fps,
              config: { damping: 14, stiffness: 100, mass: 0.7 },
            });
            const opacity = interpolate(progress, [0, 1], [0, 1]);
            const scale = interpolate(progress, [0, 1], [0.5, 1]);
            const translateY = interpolate(progress, [0, 1], [20, 0]);

            return (
              <div
                key={i}
                style={{
                  display: 'flex',
                  flexDirection: 'column',
                  alignItems: 'center',
                  gap: 8,
                  opacity,
                  transform: `scale(${scale}) translateY(${translateY}px)`,
                }}
              >
                <span style={{ fontSize: iconSize, lineHeight: 1 }}>
                  {resolveIcon(icon.name)}
                </span>
                {icon.label && (
                  <span
                    style={{
                      fontSize: labelSize,
                      fontWeight: 600,
                      color: brand.textColor,
                      textAlign: 'center',
                      maxWidth: 200,
                    }}
                  >
                    {icon.label}
                  </span>
                )}
              </div>
            );
          })}
        </div>

        {/* Caption */}
        {caption && (
          <div
            style={{
              fontSize: 28,
              fontWeight: 500,
              color: brand.textColor,
              opacity: captionOpacity * 0.7,
              transform: `translateY(${captionY}px)`,
              textAlign: 'center',
              maxWidth: 700,
              lineHeight: 1.4,
            }}
          >
            {caption}
          </div>
        )}
      </AbsoluteFill>
    </AbsoluteFill>
  );
};
