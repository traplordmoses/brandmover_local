import React from 'react';
import { AbsoluteFill, Sequence } from 'remotion';
import { SceneData, Scene, BrandTheme } from './lib/types';
import { TitleScene } from './scenes/TitleScene';
import { TaglineScene } from './scenes/TaglineScene';
import { TextOnlyScene } from './scenes/TextOnlyScene';
import { StatScene } from './scenes/StatScene';
import { FeatureListScene } from './scenes/FeatureListScene';
import { ChatDemoScene } from './scenes/ChatDemoScene';
import { StepsScene } from './scenes/StepsScene';
import { IconGridScene } from './scenes/IconGridScene';
import { DataVizScene } from './scenes/DataVizScene';
import { StockFootageScene } from './scenes/StockFootageScene';
import { IconRevealScene } from './scenes/IconRevealScene';
import { FeatureCountScene } from './scenes/FeatureCountScene';
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

function renderScene(scene: Scene, brand: BrandTheme): React.ReactNode {
  switch (scene.type) {
    case 'title':
      return <TitleScene {...scene} brand={brand} />;
    case 'tagline':
      return <TaglineScene {...scene} brand={brand} />;
    case 'text_only':
      return <TextOnlyScene {...scene} brand={brand} />;
    case 'stat':
      return <StatScene {...scene} brand={brand} />;
    case 'feature_list':
      return <FeatureListScene {...scene} brand={brand} />;
    case 'chat_demo':
      return <ChatDemoScene {...scene} brand={brand} />;
    case 'steps':
      return <StepsScene {...scene} brand={brand} />;
    case 'icon_grid':
      return <IconGridScene {...scene} brand={brand} />;
    case 'data_viz':
      return <DataVizScene {...scene} brand={brand} />;
    case 'stock_footage':
      return <StockFootageScene {...scene} brand={brand} />;
    case 'icon_reveal':
      return <IconRevealScene {...scene} brand={brand} />;
    case 'feature_count':
      return <FeatureCountScene {...scene} brand={brand} />;
    case 'cta':
      return <CTAScene {...scene} brand={brand} />;
    default:
      return null;
  }
}
