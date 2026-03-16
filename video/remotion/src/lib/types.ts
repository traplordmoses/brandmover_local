// ─── Brand Theme ────────────────────────────────────────────────
export interface BrandTheme {
  name: string;
  primaryColor: string;
  accentColor: string;
  backgroundColor: string;
  textColor: string;
  fontFamily: string;
  accentFontFamily?: string;
  logoUrl?: string;
}

// ─── Video Config ───────────────────────────────────────────────
export interface VideoConfig {
  width: number;
  height: number;
  fps: number;
  durationInSeconds: number;
  format?: 'square' | 'landscape' | 'portrait';
  theme?: 'dark' | 'light';
  brand: BrandTheme;
  audio?: {
    voiceover?: boolean;
    voiceId?: string;
    backgroundMusic?: string;
    musicVolume?: number;
  };
}

// ─── Scene Types ────────────────────────────────────────────────
interface BaseScene {
  durationFrames: number;
  transition?: 'cut' | 'crossfade' | 'fade-to-black';
  narration?: string;
  background?: 'gradient' | 'clean' | 'dots' | 'custom';
}

export interface TitleScene extends BaseScene {
  type: 'title';
  label?: string;
  headline: string;
  subheadline?: string;
  disclaimer?: string;
}

export interface TaglineScene extends BaseScene {
  type: 'tagline';
  supertext?: string;
  lines: Array<{ text: string; accent?: boolean; style?: 'normal' | 'handwritten' | 'bold' }>;
}

export interface TextOnlyScene extends BaseScene {
  type: 'text_only';
  text: string;
  size?: 'medium' | 'large' | 'xlarge';
}

export interface StatScene extends BaseScene {
  type: 'stat';
  prefix?: string;
  value: string;
  suffix?: string;
  suffixStyle?: 'normal' | 'handwritten';
  rawNumber?: string;
  icon?: string;
  animate?: 'countUp' | 'fadeIn';
}

export interface FeatureListScene extends BaseScene {
  type: 'feature_list';
  title?: string;
  items: Array<{ text: string; accent?: boolean }>;
  layout: 'centered-stack' | 'left-aligned';
}

export interface ChatDemoScene extends BaseScene {
  type: 'chat_demo';
  messages: Array<{ text: string; isUser: boolean; label?: string }>;
}

export interface StepsScene extends BaseScene {
  type: 'steps';
  title?: string;
  steps: Array<{ number: string; heading: string; detail: string }>;
}

export interface IconGridScene extends BaseScene {
  type: 'icon_grid';
  icon: string;
  rows: number;
  cols: number;
  revealPattern: 'staggered-ltr' | 'random' | 'all-at-once';
  showCheckmarks?: boolean;
}

export interface DataVizScene extends BaseScene {
  type: 'data_viz';
  vizType: 'bracket' | 'dot_matrix_number' | 'dot_grid' | 'bar_chart';
  data: Record<string, any>;
}

export interface StockFootageScene extends BaseScene {
  type: 'stock_footage';
  query: string;
  assetPath?: string;
  display: 'full_bleed' | 'inset_centered';
  overlayText?: string;
  filter?: 'none' | 'grayscale' | 'desaturated';
}

export interface IconRevealScene extends BaseScene {
  type: 'icon_reveal';
  icons: Array<{ name: string; label?: string }>;
  caption?: string;
  layout: 'single_centered' | 'row';
}

export interface FeatureCountScene extends BaseScene {
  type: 'feature_count';
  count: number;
  subtitle: string;
}

export interface CTAScene extends BaseScene {
  type: 'cta';
  lines: Array<{ text: string; accent?: boolean }>;
  url?: string;
  buttonText?: string;
}

export type Scene =
  | TitleScene | TaglineScene | TextOnlyScene | StatScene
  | FeatureListScene | ChatDemoScene | StepsScene | IconGridScene
  | DataVizScene | StockFootageScene | IconRevealScene
  | FeatureCountScene | CTAScene;

export interface SceneData {
  config: VideoConfig;
  scenes: Scene[];
}
