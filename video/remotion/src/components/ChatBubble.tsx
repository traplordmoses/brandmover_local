import React from 'react';
import { useCurrentFrame, interpolate, spring, useVideoConfig } from 'remotion';

interface Props {
  text: string;
  isUser: boolean;
  label?: string;
  accentColor: string;
  delay: number;
}

export const ChatBubble: React.FC<Props> = ({
  text,
  isUser,
  label,
  accentColor,
  delay,
}) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  const progress = spring({
    frame: frame - delay,
    fps,
    config: { damping: 14, stiffness: 120, mass: 0.7 },
  });

  const opacity = interpolate(progress, [0, 1], [0, 1]);
  const translateY = interpolate(progress, [0, 1], [40, 0]);
  const scale = interpolate(progress, [0, 1], [0.95, 1]);

  const bubbleStyle: React.CSSProperties = {
    maxWidth: '75%',
    padding: '16px 20px',
    borderRadius: 16,
    fontSize: 24,
    lineHeight: 1.5,
    opacity,
    transform: `translateY(${translateY}px) scale(${scale})`,
    whiteSpace: 'pre-line',
    alignSelf: isUser ? 'flex-end' : 'flex-start',
  };

  if (isUser) {
    // User bubble — darker, accent-tinted glass
    Object.assign(bubbleStyle, {
      background: `${accentColor}25`,
      border: `1px solid ${accentColor}40`,
      color: '#ffffff',
      borderBottomRightRadius: 4,
    });
  } else {
    // Bot bubble — frosted glass
    Object.assign(bubbleStyle, {
      background: 'rgba(255, 255, 255, 0.08)',
      border: '1px solid rgba(255, 255, 255, 0.12)',
      color: '#ffffff',
      backdropFilter: 'blur(12px)',
      borderBottomLeftRadius: 4,
    });
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', alignItems: isUser ? 'flex-end' : 'flex-start', gap: 4 }}>
      {label && !isUser && (
        <span style={{
          fontSize: 13,
          fontWeight: 600,
          color: accentColor,
          letterSpacing: 1,
          textTransform: 'uppercase',
          opacity,
          marginLeft: 4,
        }}>
          {label}
        </span>
      )}
      <div style={bubbleStyle}>{text}</div>
    </div>
  );
};
