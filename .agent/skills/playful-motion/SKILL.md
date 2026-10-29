---
name: playful-motion
description: Design and build energetic, tactile interfaces using real spring physics, staggered reveals, oversized rounded geometry, pill tokens, and purposeful micro-interactions (@reactbits-pro/skill-playful-motion).
---

# Playful Motion Design Skill (`@reactbits-pro/skill-playful-motion`)

Use this skill to craft lively, organic, high-craft interfaces that feel tactile and responsive through spring physics, oversized rounded geometry, staggered entrance choreography, and purposeful micro-interactions—without sacrificing clarity or usability.

## 1. Core Aesthetic & Geometry
- **Oversized Rounded Geometry**:
  - Primary containers & cards: `border-radius: 28px` to `36px`
  - Inner panels & Bento cells: `border-radius: 20px` to `24px`
  - Interactive badges, tags, status tokens, & CTAs: full pill geometry (`border-radius: 9999px`)
- **Tactile Depth & Surfaces**:
  - Layer translucent glass surfaces (`backdrop-filter: blur(16px)`) with subtle top-edge specular highlights (`inset 0 1px 0 rgba(255, 255, 255, 0.12)`) and soft ambient glow shadows.
  - Pair dark obsidian/slate canvases (`#07090e` / `#0d121c`) with vibrant, high-contrast playful accents (electric mint `#00FFC2`, sky cyan `#38BDF8`, energetic violet `#A855F7`, warm amber `#F59E0B`, coral rose `#FB7185`).
- **Floating Ambient Depth**:
  - Use subtle animated radial gradient orbs/blobs behind hero and sandbox regions that drift organically on slow keyframe loops.

## 2. Spring Physics & Motion Tokens
- **Never use flat linear transitions**. Approximate real spring physics (`stiffness: 320, damping: 22, mass: 1`) using custom cubic-bezier curves:
  - **Bouncy Spring (Hover / Pop / Scale)**: `cubic-bezier(0.34, 1.56, 0.64, 1)` (`--spring-bounce`)
  - **Smooth Spring (Reveal / Layout)**: `cubic-bezier(0.22, 1, 0.36, 1)` (`--spring-smooth`)
  - **Snappy Press (Active / Tap)**: `cubic-bezier(0.16, 1, 0.3, 1)` (`--spring-snappy`)
- **Staggered Entrance Choreography**:
  - Cascade child elements with incremental animation delays (`60ms`, `120ms`, `180ms`, `240ms`) rising from `translateY(24px) scale(0.96)` to `translateY(0) scale(1)`.
- **Purposeful Micro-Interactions**:
  - Cards lift and tilt/scale (`transform: translateY(-6px) scale(1.015)`) on hover with an energized border glow.
  - Step badges, icons, and pills bounce (`scale(1.12) rotate(-4deg)`) when their parent card is hovered.
  - Buttons compress slightly on active press (`scale(0.97)`) and spring upward on hover (`translateY(-3px) scale(1.01)`).
- **Accessibility**:
  - Always respect `@media (prefers-reduced-motion: reduce)` by collapsing animations and spring transforms for users who prefer reduced motion.

## 3. Typography & Hierarchy
- Pair expressive, high-impact geometric display headlines (`Plus Jakarta Sans` / `Space Grotesk` / `Inter`, weight `800–900`, tight tracking `-0.03em`) with crisp monospace telemetry (`JetBrains Mono` / `Fira Code`) for live metrics, latency badges, and code blocks.
- Use pill-shaped eyebrow kickers with live pulsing status dots above section titles.
