import { registerRoot } from 'remotion';
import { RemotionRoot } from './Root';
import { continueRender, delayRender } from 'remotion';

// Load Google Fonts so they're available during rendering
const fontFamilies = ['Inter:wght@400;500;600;700;800;900', 'Orbitron:wght@400;500;600;700;800;900'];
const fontUrl = `https://fonts.googleapis.com/css2?${fontFamilies.map(f => `family=${f}`).join('&')}&display=swap`;

const waitForFonts = delayRender('Loading Google Fonts');
const link = document.createElement('link');
link.rel = 'stylesheet';
link.href = fontUrl;
link.onload = () => {
  // Give fonts a moment to apply after stylesheet loads
  document.fonts.ready.then(() => continueRender(waitForFonts));
};
link.onerror = () => continueRender(waitForFonts);
document.head.appendChild(link);

registerRoot(RemotionRoot);
