import React from 'react';
import { GradientBg } from '../components/GradientBg';
import { CleanBg } from '../components/CleanBg';
import { DotsBg } from '../components/DotsBg';
import { BrandTheme } from './types';

/**
 * Determine if a color is "dark" by parsing hex and checking luminance.
 */
export function isDarkColor(hex: string): boolean {
  const clean = hex.replace('#', '');
  const r = parseInt(clean.substring(0, 2), 16);
  const g = parseInt(clean.substring(2, 4), 16);
  const b = parseInt(clean.substring(4, 6), 16);
  // Relative luminance approximation
  const luminance = (0.299 * r + 0.587 * g + 0.114 * b) / 255;
  return luminance < 0.5;
}

/**
 * Alias for isDarkColor using the brand's backgroundColor.
 */
export function isDarkTheme(brand: BrandTheme): boolean {
  return isDarkColor(brand.backgroundColor);
}

/**
 * Choose which background type to use based on the explicit background prop
 * and the brand's backgroundColor brightness.
 *
 * Returns: 'gradient' | 'clean' | 'dots'
 */
export function chooseBgType(
  background: string | undefined,
  brand: BrandTheme,
): 'gradient' | 'clean' | 'dots' {
  if (background === 'gradient') return 'gradient';
  if (background === 'clean') return 'clean';
  if (background === 'dots') return 'dots';
  // Auto-detect from theme
  return isDarkColor(brand.backgroundColor) ? 'gradient' : 'clean';
}

/**
 * Return a React element for the appropriate background component.
 */
export function renderBackground(
  bg: string | undefined,
  brand: BrandTheme
): React.ReactElement {
  const bgType = chooseBgType(bg, brand);
  if (bgType === 'dots') {
    return React.createElement(DotsBg, { backgroundColor: brand.backgroundColor });
  }
  if (bgType === 'clean') {
    return React.createElement(CleanBg, { backgroundColor: brand.backgroundColor });
  }
  return React.createElement(GradientBg, { backgroundColor: brand.backgroundColor, accentColor: brand.accentColor });
}
