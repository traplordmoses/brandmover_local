import React from 'react';
import { useCurrentFrame, interpolate, spring, useVideoConfig } from 'remotion';

interface Props {
  rows: number;
  cols: number;
  icon?: string;
  accentColor: string;
  showCheckmarks?: boolean;
  revealPattern: 'staggered-ltr' | 'random' | 'all-at-once';
  delay?: number;
  cellSize?: number;
}

/**
 * Generate a stable pseudo-random order for cells based on a simple hash.
 */
function shuffledIndices(total: number): number[] {
  const indices = Array.from({ length: total }, (_, i) => i);
  // Simple deterministic shuffle (seeded by total)
  for (let i = total - 1; i > 0; i--) {
    const j = ((i * 7 + 13) * 31) % (i + 1);
    [indices[i], indices[j]] = [indices[j], indices[i]];
  }
  return indices;
}

export const StaggeredGrid: React.FC<Props> = ({
  rows,
  cols,
  icon,
  accentColor,
  showCheckmarks = false,
  revealPattern,
  delay = 0,
  cellSize = 56,
}) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const total = rows * cols;
  const staggerGap = 3; // frames between each cell reveal

  // Compute reveal order
  const revealOrder = React.useMemo(() => {
    if (revealPattern === 'all-at-once') {
      return Array.from({ length: total }, () => 0);
    }
    if (revealPattern === 'random') {
      return shuffledIndices(total);
    }
    // staggered-ltr: left to right, top to bottom
    return Array.from({ length: total }, (_, i) => i);
  }, [revealPattern, total]);

  const gridWidth = cols * cellSize;
  const gridHeight = rows * cellSize;

  return (
    <div
      style={{
        display: 'grid',
        gridTemplateColumns: `repeat(${cols}, ${cellSize}px)`,
        gridTemplateRows: `repeat(${rows}, ${cellSize}px)`,
        width: gridWidth,
        height: gridHeight,
        gap: 0,
      }}
    >
      {Array.from({ length: total }, (_, i) => {
        const cellDelay = delay + revealOrder[i] * staggerGap;

        const progress = spring({
          frame: frame - cellDelay,
          fps,
          config: { damping: 14, stiffness: 100, mass: 0.7 },
        });

        const opacity = interpolate(progress, [0, 1], [0, 1]);
        const scale = interpolate(progress, [0, 1], [0.4, 1]);

        // Checkmark appears after the cell settles
        const checkDelay = cellDelay + 10;
        const checkProgress = spring({
          frame: frame - checkDelay,
          fps,
          config: { damping: 14, stiffness: 120, mass: 0.5 },
        });
        const checkOpacity = interpolate(checkProgress, [0, 1], [0, 1]);
        const checkScale = interpolate(checkProgress, [0, 1], [0.3, 1]);

        return (
          <div
            key={i}
            style={{
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              opacity,
              transform: `scale(${scale})`,
              position: 'relative',
            }}
          >
            {icon ? (
              <span style={{ fontSize: cellSize * 0.55 }}>{icon}</span>
            ) : (
              <div
                style={{
                  width: cellSize * 0.5,
                  height: cellSize * 0.5,
                  borderRadius: '50%',
                  backgroundColor: accentColor,
                }}
              />
            )}
            {showCheckmarks && (
              <span
                style={{
                  position: 'absolute',
                  bottom: 2,
                  right: 4,
                  fontSize: cellSize * 0.3,
                  color: '#22c55e',
                  opacity: checkOpacity,
                  transform: `scale(${checkScale})`,
                }}
              >
                ✓
              </span>
            )}
          </div>
        );
      })}
    </div>
  );
};
