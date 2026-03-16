import React from 'react';
import { AbsoluteFill } from 'remotion';
import { KineticText } from '../components/KineticText';
import { BrandTheme } from '../lib/types';
import { renderBackground } from '../lib/backgrounds';

interface Props {
  text: string;
  size?: 'medium' | 'large' | 'xlarge';
  background?: string;
  brand: BrandTheme;
}

const SIZE_MAP = {
  medium: 42,
  large: 56,
  xlarge: 72,
};

export const TextOnlyScene: React.FC<Props> = ({
  text,
  size = 'large',
  background,
  brand,
}) => {
  const fontSize = SIZE_MAP[size];

  return (
    <AbsoluteFill>
      {renderBackground(background, brand)}
      <AbsoluteFill
        style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          fontFamily: brand.fontFamily,
          padding: 60,
        }}
      >
        <KineticText
          text={text}
          fontSize={fontSize}
          color={brand.textColor}
          fontWeight={700}
          delay={0}
          wordStagger={3}
          entrance="rise"
          style={{ textAlign: 'center', maxWidth: 900 }}
        />
      </AbsoluteFill>
    </AbsoluteFill>
  );
};
