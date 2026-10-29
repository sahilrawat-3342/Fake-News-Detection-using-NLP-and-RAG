import { useEffect, useRef } from "react";
import { useInView, useMotionValue, useSpring, useTransform, motion } from "motion/react";

/** Spring-driven count-up that starts when scrolled into view. */
export function AnimatedNumber({
  value,
  decimals = 0,
  suffix = "",
  prefix = "",
  className,
}: {
  value: number;
  decimals?: number;
  suffix?: string;
  prefix?: string;
  className?: string;
}) {
  const ref = useRef<HTMLSpanElement>(null);
  const inView = useInView(ref, { once: true, margin: "-40px" });
  const raw = useMotionValue(0);
  const smooth = useSpring(raw, { stiffness: 90, damping: 22, mass: 1 });
  const display = useTransform(smooth, (v) =>
    `${prefix}${v.toLocaleString(undefined, { minimumFractionDigits: decimals, maximumFractionDigits: decimals })}${suffix}`,
  );

  useEffect(() => {
    // Setting the raw target lets the spring physics drive the count-up.
    if (inView) raw.set(value);
  }, [inView, value, raw]);

  return (
    <motion.span ref={ref} className={className}>
      {display}
    </motion.span>
  );
}
