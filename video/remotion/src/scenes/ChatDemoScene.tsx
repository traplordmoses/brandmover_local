import React from 'react';
import { AbsoluteFill } from 'remotion';
import { ChatBubble } from '../components/ChatBubble';
import { GradientBg } from '../components/GradientBg';
import { BrandTheme } from '../lib/types';

interface Message {
  text: string;
  isUser: boolean;
  label?: string;
}

interface Props {
  messages: Message[];
  brand: BrandTheme;
}

export const ChatDemoScene: React.FC<Props> = ({ messages, brand }) => {
  const staggerDelay = 20; // frames between each message appearing

  return (
    <AbsoluteFill>
      <GradientBg backgroundColor={brand.backgroundColor} accentColor={brand.accentColor} />
      <AbsoluteFill style={{
        display: 'flex',
        flexDirection: 'column',
        justifyContent: 'center',
        padding: '60px 50px',
        gap: 16,
        fontFamily: brand.fontFamily,
      }}>
        {messages.map((msg, i) => (
          <ChatBubble
            key={i}
            text={msg.text}
            isUser={msg.isUser}
            label={msg.label}
            accentColor={brand.accentColor}
            delay={i * staggerDelay + 10}
          />
        ))}
      </AbsoluteFill>
    </AbsoluteFill>
  );
};
