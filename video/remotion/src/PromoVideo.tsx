import React from 'react';
import { AbsoluteFill, Sequence } from 'remotion';
import { SceneData, Scene } from './lib/types';
import { TitleScene } from './scenes/TitleScene';
import { TaglineScene } from './scenes/TaglineScene';
import { FeatureCountScene } from './scenes/FeatureCountScene';
import { ChatDemoScene } from './scenes/ChatDemoScene';
import { StepsScene } from './scenes/StepsScene';
import { CTAScene } from './scenes/CTAScene';

export const PromoVideo: React.FC<SceneData> = ({ config, scenes }) => {
  const { brand } = config;
  let frameOffset = 0;

  return (
    <AbsoluteFill style={{ backgroundColor: brand.backgroundColor }}>
      {scenes.map((scene, i) => {
        const from = frameOffset;
        frameOffset += scene.durationFrames;

        return (
          <Sequence key={i} from={from} durationInFrames={scene.durationFrames}>
            {renderScene(scene, brand)}
          </Sequence>
        );
      })}
    </AbsoluteFill>
  );
};

function renderScene(scene: Scene, brand: SceneData['config']['brand']): React.ReactNode {
  switch (scene.type) {
    case 'title':
      return <TitleScene label={scene.label} headline={scene.headline} brand={brand} />;
    case 'tagline':
      return <TaglineScene supertext={scene.supertext} line1={scene.line1} line2={scene.line2} accentLine={scene.accentLine} brand={brand} />;
    case 'feature_count':
      return <FeatureCountScene count={scene.count} subtitle={scene.subtitle} brand={brand} />;
    case 'chat_demo':
      return <ChatDemoScene messages={scene.messages} brand={brand} />;
    case 'steps':
      return <StepsScene title={scene.title} steps={scene.steps} brand={brand} />;
    case 'cta':
      return <CTAScene line1={scene.line1} line2={scene.line2} accentLine={scene.accentLine} url={scene.url} buttonText={scene.buttonText} brand={brand} />;
    default:
      return null;
  }
}
