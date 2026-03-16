export interface BrandTheme {
  name: string;
  primaryColor: string;
  accentColor: string;
  backgroundColor: string;
  fontFamily: string;
  logoUrl?: string;
}

export interface VideoConfig {
  width: number;
  height: number;
  fps: number;
  durationInSeconds: number;
  brand: BrandTheme;
}

export type Scene =
  | { type: 'title'; label: string; headline: string; durationFrames: number }
  | { type: 'tagline'; supertext: string; line1: string; line2: string; accentLine: 1 | 2; durationFrames: number }
  | { type: 'feature_count'; count: number; subtitle: string; durationFrames: number }
  | { type: 'chat_demo'; messages: Array<{ text: string; isUser: boolean; label?: string }>; durationFrames: number }
  | { type: 'steps'; title: string; steps: Array<{ number: string; heading: string; detail: string }>; durationFrames: number }
  | { type: 'cta'; line1: string; line2: string; accentLine: 1 | 2; url: string; buttonText: string; durationFrames: number };

export interface SceneData {
  config: VideoConfig;
  scenes: Scene[];
}
