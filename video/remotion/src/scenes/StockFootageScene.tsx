import React from 'react';
import { AbsoluteFill, useCurrentFrame, interpolate, spring, useVideoConfig, Img, OffthreadVideo } from 'remotion';
import { renderBackground } from '../lib/backgrounds';
import { BrandTheme } from '../lib/types';

interface Props {
  assetPath?: string;
  display: 'full_bleed' | 'inset_centered';
  overlayText?: string;
  filter?: 'none' | 'grayscale' | 'desaturated';
  background?: string;
  brand: BrandTheme;
}

const VIDEO_EXTENSIONS = ['.mp4', '.webm', '.mov', '.avi', '.mkv'];

function isVideoFile(path: string): boolean {
  const lower = path.toLowerCase();
  return VIDEO_EXTENSIONS.some((ext) => lower.endsWith(ext));
}

function getCssFilter(filter?: string): string {
  if (filter === 'grayscale') return 'grayscale(100%)';
  if (filter === 'desaturated') return 'grayscale(50%)';
  return 'none';
}

export const StockFootageScene: React.FC<Props> = ({
  assetPath,
  display,
  overlayText,
  filter = 'none',
  background,
  brand,
}) => {
  const frame = useCurrentFrame();
  const { fps, durationInFrames } = useVideoConfig();

  // Ken Burns: slow zoom from 1.0 to 1.05
  const zoomScale = interpolate(frame, [0, durationInFrames], [1.0, 1.05], {
    extrapolateLeft: 'clamp',
    extrapolateRight: 'clamp',
  });

  // Overlay text animation
  const textProgress = spring({
    frame: frame - 10,
    fps,
    config: { damping: 15, stiffness: 100, mass: 0.8 },
  });
  const textOpacity = interpolate(textProgress, [0, 1], [0, 1]);
  const textY = interpolate(textProgress, [0, 1], [30, 0]);

  const isInset = display === 'inset_centered';
  const cssFilter = getCssFilter(filter);

  const mediaContainerStyle: React.CSSProperties = isInset
    ? {
        width: '80%',
        height: '80%',
        borderRadius: 16,
        overflow: 'hidden',
        position: 'relative',
      }
    : {
        width: '100%',
        height: '100%',
        position: 'relative',
        overflow: 'hidden',
      };

  const mediaStyle: React.CSSProperties = {
    width: '100%',
    height: '100%',
    objectFit: 'cover',
    filter: cssFilter,
    transform: `scale(${zoomScale})`,
  };

  const renderMedia = () => {
    if (!assetPath) {
      // Gradient placeholder
      return (
        <div
          style={{
            ...mediaContainerStyle,
            background: `linear-gradient(135deg, ${brand.primaryColor}44, ${brand.accentColor}44)`,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
          }}
        >
          {overlayText && (
            <div
              style={{
                fontSize: 48,
                fontWeight: 700,
                color: brand.textColor,
                textAlign: 'center',
                padding: 40,
                opacity: textOpacity,
                transform: `translateY(${textY}px)`,
                maxWidth: '80%',
              }}
            >
              {overlayText}
            </div>
          )}
        </div>
      );
    }

    const isVideo = isVideoFile(assetPath);

    return (
      <div style={mediaContainerStyle}>
        {isVideo ? (
          <OffthreadVideo src={assetPath} style={mediaStyle} />
        ) : (
          <Img src={assetPath} style={mediaStyle} />
        )}

        {/* Dark overlay for text readability */}
        {overlayText && (
          <>
            <div
              style={{
                position: 'absolute',
                inset: 0,
                backgroundColor: 'rgba(0, 0, 0, 0.4)',
                pointerEvents: 'none',
              }}
            />
            <div
              style={{
                position: 'absolute',
                inset: 0,
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                padding: 40,
              }}
            >
              <div
                style={{
                  fontSize: 48,
                  fontWeight: 700,
                  color: '#ffffff',
                  textAlign: 'center',
                  opacity: textOpacity,
                  transform: `translateY(${textY}px)`,
                  maxWidth: '80%',
                  fontFamily: brand.fontFamily,
                  lineHeight: 1.2,
                }}
              >
                {overlayText}
              </div>
            </div>
          </>
        )}
      </div>
    );
  };

  return (
    <AbsoluteFill>
      {isInset && renderBackground(background, brand)}
      <AbsoluteFill
        style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          fontFamily: brand.fontFamily,
        }}
      >
        {renderMedia()}
      </AbsoluteFill>
    </AbsoluteFill>
  );
};
