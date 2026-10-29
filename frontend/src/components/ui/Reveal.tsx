import type { ReactNode } from "react";
import { motion } from "motion/react";
import { riseIn, staggerParent } from "@/lib/motion";
import { cn } from "@/lib/utils";

/** Scroll-triggered staggered reveal container. Children should use `RevealItem`. */
export function Reveal({
  children,
  className,
  stagger = 0.08,
  delay = 0,
  as = "div",
}: {
  children: ReactNode;
  className?: string;
  stagger?: number;
  delay?: number;
  as?: "div" | "section" | "ul";
}) {
  const Comp = motion[as];
  return (
    <Comp
      className={className}
      variants={staggerParent(stagger, delay)}
      initial="hidden"
      whileInView="show"
      viewport={{ once: true, margin: "-80px" }}
    >
      {children}
    </Comp>
  );
}

export function RevealItem({ children, className }: { children: ReactNode; className?: string }) {
  return (
    <motion.div variants={riseIn} className={cn(className)}>
      {children}
    </motion.div>
  );
}
