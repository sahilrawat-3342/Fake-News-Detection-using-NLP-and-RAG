import type { Transition, Variants } from "motion/react";

/**
 * Spring presets — real physics, never duration-based easing.
 * Tuned per the Playful Motion skill: bouncy for pops, smooth for layout,
 * snappy for taps.
 */
export const spring = {
  bouncy: { type: "spring", stiffness: 420, damping: 18, mass: 0.9 },
  smooth: { type: "spring", stiffness: 260, damping: 30, mass: 1 },
  snappy: { type: "spring", stiffness: 600, damping: 32, mass: 0.6 },
  gentle: { type: "spring", stiffness: 140, damping: 20, mass: 1 },
} satisfies Record<string, Transition>;

export const staggerParent = (stagger = 0.07, delay = 0): Variants => ({
  hidden: {},
  show: { transition: { staggerChildren: stagger, delayChildren: delay } },
});

export const riseIn: Variants = {
  hidden: { opacity: 0, y: 28, scale: 0.96 },
  show: { opacity: 1, y: 0, scale: 1, transition: spring.bouncy },
};

export const popIn: Variants = {
  hidden: { opacity: 0, scale: 0.6, rotate: -6 },
  show: { opacity: 1, scale: 1, rotate: 0, transition: spring.bouncy },
};
