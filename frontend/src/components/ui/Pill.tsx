import type { ReactNode } from "react";
import { cn } from "@/lib/utils";

type Tone = "ink" | "tomato" | "mint" | "butter" | "berry" | "cobalt" | "paper";

const tones: Record<Tone, string> = {
  ink: "bg-ink text-paper",
  tomato: "bg-tomato text-paper",
  mint: "bg-mint-soft text-[#0b6b4a]",
  butter: "bg-butter-soft text-[#7a5800]",
  berry: "bg-berry-soft text-[#a3123b]",
  cobalt: "bg-cobalt-soft text-[#2234a8]",
  paper: "bg-white text-ink-soft border border-line",
};

export function Pill({
  tone = "paper",
  className,
  children,
  dot,
}: {
  tone?: Tone;
  className?: string;
  children: ReactNode;
  dot?: string;
}) {
  return (
    <span
      className={cn(
        "inline-flex items-center gap-1.5 rounded-full px-3 py-1 font-mono text-[0.72rem] font-medium tracking-tight",
        tones[tone],
        className,
      )}
    >
      {dot && <span className="size-1.5 rounded-full" style={{ background: dot }} aria-hidden />}
      {children}
    </span>
  );
}
