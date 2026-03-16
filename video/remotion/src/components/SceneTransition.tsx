import React from 'react';
import { useCurrentFrame, interpolate, AbsoluteFill } from 'remotion';

interface Props {
  type: 'cut' | 'crossfade' | 'fade-to-black';
  children: React.ReactNode;
  durationFrames: number;
}

const FADE_FRAMES = 8;

export const SceneTransition: React.FC<Props> = ({
  type,
  children,
  durationFrames,
}) => {
  const frame = useCurrentFrame();

  if (type === 'cut') {
    return <>{children}</>;
  }

  if (type === 'crossfade') {
    const fadeIn = interpolate(frame, [0, FADE_FRAMES], [0, 1], {
      extrapolateLeft: 'clamp',
      extrapolateRight: 'clamp',
    });
    const fadeOut = interpolate(
      frame,
      [durationFrames - FADE_FRAMES, durationFrames],
      [1, 0],
      { extrapolateLeft: 'clamp', extrapolateRight: 'clamp' },
    );
    const opacity = Math.min(fadeIn, fadeOut);

    return <AbsoluteFill style={{ opacity }}>{children}</AbsoluteFill>;
  }

  // fade-to-black
  const fadeIn = interpolate(frame, [0, FADE_FRAMES], [1, 0], {
    extrapolateLeft: 'clamp',
    extrapolateRight: 'clamp',
  });
  const fadeOut = interpolate(
    frame,
    [durationFrames - FADE_FRAMES, durationFrames],
    [0, 1],
    { extrapolateLeft: 'clamp', extrapolateRight: 'clamp' },
  );
  const blackOpacity = Math.max(fadeIn, fadeOut);

  return (
    <AbsoluteFill>
      {children}
      <AbsoluteFill
        style={{
          backgroundColor: '#000000',
          opacity: blackOpacity,
          pointerEvents: 'none',
        }}
      />
    </AbsoluteFill>
  );
};
