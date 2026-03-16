import React from 'react';
import { Composition } from 'remotion';
import { PromoVideo } from './PromoVideo';
import { SceneData } from './lib/types';

// Default props for Remotion Studio preview
const defaultProps: SceneData = {
  config: {
    width: 1080,
    height: 1080,
    fps: 30,
    durationInSeconds: 16.5,
    format: 'square',
    theme: 'dark',
    brand: {
      name: 'BrandMover',
      primaryColor: '#72e1ff',
      accentColor: '#72e1ff',
      backgroundColor: '#0a0f1a',
      textColor: '#ffffff',
      fontFamily: 'Inter',
    },
  },
  scenes: [
    { type: 'title', label: 'INTRODUCING', headline: 'BrandMover', durationFrames: 75 },
    { type: 'tagline', supertext: 'AI MARKETING AGENT', lines: [{ text: 'Your brand,' }, { text: 'on autopilot.', accent: true }], durationFrames: 90 },
    { type: 'feature_count', count: 5, subtitle: 'tools. One brain.', durationFrames: 60 },
    { type: 'chat_demo', messages: [
      { text: 'Make a post about our launch', isUser: true },
      { text: '✓ Draft ready\nImage generated • Caption written • Hashtags added', isUser: false, label: 'BRANDMOVER' },
    ], durationFrames: 90 },
    { type: 'steps', title: 'GET STARTED', steps: [
      { number: '01', heading: 'Connect your brand', detail: 'Upload guidelines → AI learns your voice' },
      { number: '02', heading: 'Start chatting', detail: 'Tell it what to post. It handles the rest.' },
    ], durationFrames: 90 },
    { type: 'cta', lines: [{ text: 'Post smarter.' }, { text: 'Build faster.', accent: true }], url: 'brandmover.ai', buttonText: 'Try Free', durationFrames: 60 },
  ],
};

export const RemotionRoot: React.FC = () => {
  const totalFrames = defaultProps.scenes.reduce((sum, s) => sum + s.durationFrames, 0);

  return (
    <Composition
      id="PromoVideo"
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      component={PromoVideo as any}
      durationInFrames={totalFrames}
      fps={defaultProps.config.fps}
      width={defaultProps.config.width}
      height={defaultProps.config.height}
      defaultProps={defaultProps as any}
      calculateMetadata={({ props }) => {
        const p = props as unknown as SceneData;
        const frames = p.scenes.reduce((sum: number, s: { durationFrames: number }) => sum + s.durationFrames, 0);
        return {
          durationInFrames: frames,
          fps: p.config.fps,
          width: p.config.width,
          height: p.config.height,
        };
      }}
    />
  );
};
