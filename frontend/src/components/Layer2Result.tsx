import { forwardRef } from "react";
import { motion } from "motion/react";
import { ArrowUpRight, CircleCheck, CircleHelp, CircleX, Search } from "lucide-react";
import type { VerifyResponse } from "@/lib/api";
import { riseIn, spring, staggerParent } from "@/lib/motion";
import { cn, domainOf } from "@/lib/utils";
import { Pill } from "./ui/Pill";

const VERDICTS = {
  True: { title: "Supported.", bg: "bg-mint-soft", chip: "bg-mint", Icon: CircleCheck, line: "The retrieved evidence backs this claim." },
  False: { title: "Refuted.", bg: "bg-berry-soft", chip: "bg-berry", Icon: CircleX, line: "The retrieved evidence contradicts this claim." },
  Unverified: {
    title: "Can't confirm.",
    bg: "bg-butter-soft",
    chip: "bg-butter",
    Icon: CircleHelp,
    line: "The evidence was thin, mixed, or off-topic.",
  },
} as const;

export const Layer2Result = forwardRef<HTMLDivElement, { result: VerifyResponse }>(function Layer2Result({ result }, ref) {
  const v = VERDICTS[result.verdict] ?? VERDICTS.Unverified;
  const sources = result.sources?.length ? result.sources : result.search_results ?? [];
  const cited = new Set(result.source_ids ?? []);

  return (
    <motion.article
      ref={ref}
      layout
      initial={{ opacity: 0, y: 40, scale: 0.94 }}
      animate={{ opacity: 1, y: 0, scale: 1 }}
      exit={{ opacity: 0, scale: 0.94 }}
      transition={spring.bouncy}
      className="overflow-hidden rounded-[2rem] border-2 border-ink bg-white shadow-pop-lg"
    >
      <div className={cn("grid gap-6 p-6 md:grid-cols-[auto_1fr] md:items-center md:p-8", v.bg)}>
        <motion.div
          initial={{ scale: 0, rotate: 40 }}
          animate={{ scale: 1, rotate: 0 }}
          transition={{ ...spring.bouncy, delay: 0.12 }}
          className={cn("grid size-20 place-items-center rounded-full border-2 border-ink", v.chip)}
        >
          <v.Icon className="size-10" strokeWidth={2.2} />
        </motion.div>
        <div>
          <div className="flex flex-wrap gap-2">
            <Pill tone="ink">Layer 2 · Deep verify</Pill>
            <Pill>{result.elapsed_s}s</Pill>
            <Pill>
              <Search className="size-3" /> {result.query}
            </Pill>
          </div>
          <h3 className="mt-3 font-display text-4xl font-extrabold tracking-tight md:text-5xl">{v.title}</h3>
          <p className="mt-1 text-ink-soft">{v.line}</p>
        </div>
      </div>

      <div className="grid gap-px border-y-2 border-ink bg-ink md:grid-cols-2">
        <div className="bg-white p-6 md:p-8">
          <p className="font-mono text-xs uppercase tracking-wider text-ink-mute">The claim we checked</p>
          <p className="mt-3 font-display text-xl font-bold leading-snug tracking-tight">“{result.claim}”</p>
        </div>
        <div className="bg-white p-6 md:p-8">
          <p className="font-mono text-xs uppercase tracking-wider text-ink-mute">Why</p>
          <p className="mt-3 leading-relaxed text-ink-soft">{result.reasoning}</p>
        </div>
      </div>

      {(!result.parse_ok || !result.extraction_parse_ok) && (
        <div className="border-b-2 border-ink bg-butter-soft px-6 py-3 text-sm md:px-8">
          {!result.parse_ok && "The verifier returned malformed JSON; the verdict was salvaged and is lower-confidence. "}
          {!result.extraction_parse_ok && "Claim extraction fell back to a heuristic, so the search query may be less precise."}
        </div>
      )}

      <div className="p-6 md:p-8">
        <div className="mb-4 flex items-baseline justify-between">
          <h4 className="font-display text-xl font-bold tracking-tight">
            {result.sources?.length ? "Cited sources" : "Retrieved sources"}
          </h4>
          <span className="font-mono text-xs text-ink-mute">{sources.length} found</span>
        </div>

        {sources.length === 0 ? (
          <p className="rounded-2xl bg-paper p-5 text-ink-soft">No live sources were available for this run.</p>
        ) : (
          <motion.ul variants={staggerParent(0.07, 0.2)} initial="hidden" animate="show" className="grid gap-3 md:grid-cols-2">
            {sources.map((s) => (
              <motion.li key={s.id} variants={riseIn}>
                <motion.a
                  href={s.url}
                  target="_blank"
                  rel="noopener noreferrer"
                  whileHover={{ y: -4, rotate: -0.5 }}
                  transition={spring.bouncy}
                  className="group flex h-full flex-col rounded-[1.5rem] border-2 border-ink/10 bg-paper p-5 transition-colors duration-150 hover:border-ink"
                >
                  <div className="flex items-center justify-between gap-2">
                    <div className="flex items-center gap-2">
                      <span className="grid size-7 place-items-center rounded-full bg-ink font-mono text-xs text-paper">{s.id}</span>
                      <span className="truncate font-mono text-xs text-ink-mute">{domainOf(s.url)}</span>
                      {cited.has(s.id) && <Pill tone="mint">cited</Pill>}
                    </div>
                    <ArrowUpRight className="size-5 shrink-0 transition-transform duration-150 group-hover:-translate-y-0.5 group-hover:translate-x-0.5" />
                  </div>
                  <p className="mt-3 font-semibold leading-snug">{s.title}</p>
                  <p className="mt-2 line-clamp-3 text-sm leading-relaxed text-ink-soft">{s.snippet}</p>
                </motion.a>
              </motion.li>
            ))}
          </motion.ul>
        )}
      </div>
    </motion.article>
  );
});
