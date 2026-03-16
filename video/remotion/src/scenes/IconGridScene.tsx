import React from 'react';
import { AbsoluteFill } from 'remotion';
import { StaggeredGrid } from '../components/StaggeredGrid';
import { BrandTheme } from '../lib/types';
import { renderBackground } from '../lib/backgrounds';

interface Props {
  icon: string;
  rows: number;
  cols: number;
  revealPattern: 'staggered-ltr' | 'random' | 'all-at-once';
  showCheckmarks?: boolean;
  background?: string;
  brand: BrandTheme;
}

const ICON_EMOJI_MAP: Record<string, string> = {
  coin: '\u{1FA99}',
  star: '\u2B50',
  trophy: '\u{1F3C6}',
  heart: '\u2764\uFE0F',
  fire: '\u{1F525}',
  rocket: '\u{1F680}',
  check: '\u2705',
  lightning: '\u26A1',
  diamond: '\u{1F48E}',
  globe: '\u{1F30D}',
  lock: '\u{1F512}',
  shield: '\u{1F6E1}\uFE0F',
  users: '\u{1F465}',
};

function resolveIcon(icon: string): string | undefined {
  // If the icon is already an emoji, return it directly
  if (/\p{Emoji}/u.test(icon) && icon.length <= 4) {
    return icon;
  }
  return ICON_EMOJI_MAP[icon.toLowerCase()];
}

export const IconGridScene: React.FC<Props> = ({
  icon,
  rows,
  cols,
  revealPattern,
  showCheckmarks = false,
  background,
  brand,
}) => {
  const resolvedIcon = resolveIcon(icon);

  return (
    <AbsoluteFill>
      {renderBackground(background, brand)}
      <AbsoluteFill
        style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          fontFamily: brand.fontFamily,
        }}
      >
        <StaggeredGrid
          rows={rows}
          cols={cols}
          icon={resolvedIcon}
          accentColor={brand.accentColor}
          showCheckmarks={showCheckmarks}
          revealPattern={revealPattern}
          delay={5}
        />
      </AbsoluteFill>
    </AbsoluteFill>
  );
};
