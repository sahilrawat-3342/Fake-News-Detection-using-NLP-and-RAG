import type { ReactNode } from "react";
import { RevealItem, Reveal } from "./Reveal";

export function SectionHeading({
  id,
  kicker,
  title,
  description,
  aside,
}: {
  id?: string;
  kicker: string;
  title: ReactNode;
  description?: string;
  aside?: ReactNode;
}) {
  return (
    <Reveal className="mb-10 flex flex-col gap-6 md:mb-14 md:flex-row md:items-end md:justify-between">
      <div className="max-w-2xl" id={id}>
        <RevealItem>
          <span className="inline-flex items-center gap-2 rounded-full border-2 border-ink bg-white px-3.5 py-1 font-mono text-xs font-medium uppercase tracking-wider shadow-pop">
            <span className="size-2 rounded-full bg-tomato" aria-hidden />
            {kicker}
          </span>
        </RevealItem>
        <RevealItem>
          <h2 className="mt-5 font-display text-4xl font-extrabold leading-[1.02] tracking-tight text-balance md:text-[3.4rem]">
            {title}
          </h2>
        </RevealItem>
        {description && (
          <RevealItem>
            <p className="mt-4 max-w-xl text-lg leading-relaxed text-ink-soft">{description}</p>
          </RevealItem>
        )}
      </div>
      {aside && <RevealItem>{aside}</RevealItem>}
    </Reveal>
  );
}
