import { forwardRef, useState } from "react";
import { AnimatePresence, motion } from "motion/react";
import { ChevronDown, ShieldAlert, ShieldCheck } from "lucide-react";
import type { PredictResponse } from "@/lib/api";
import { spring } from "@/lib/motion";
import { cn } from "@/lib/utils";
import { Pill } from "./ui/Pill";

export const Layer1Result = forwardRef<HTMLDivElement, { result: PredictResponse }>(function Layer1Result({ result }, ref) {
  const [open, setOpen] = useState(false);
  const real = result.label === "real";
  const realP = result.real_probability ?? (real ? 95 : 5);
  const fakeP = result.fake_probability ?? 100 - realP;
  const conf = result.confidence ?? Math.max(realP, fakeP);
  const Icon = real ? ShieldCheck : ShieldAlert;

  return (
    <motion.article
      ref={ref}
      layout
      initial={{ opacity: 0, y: 40, scale: 0.94 }}
      animate={{ opacity: 1, y: 0, scale: 1 }}
      exit={{ opacity: 0, scale: 0.94 }}
      transition={spring.bouncy}
      className={cn("overflow-hidden rounded-[2rem] border-2 border-ink shadow-pop-lg", real ? "bg-mint-soft" : "bg-berry-soft")}
    >
      <div className="grid gap-6 p-6 md:grid-cols-[auto_1fr_auto] md:items-center md:p-8">
        <motion.div
          initial={{ scale: 0, rotate: -30 }}
          animate={{ scale: 1, rotate: 0 }}
          transition={{ ...spring.bouncy, delay: 0.12 }}
          className={cn("grid size-20 place-items-center rounded-[1.6rem] border-2 border-ink text-paper", real ? "bg-mint" : "bg-berry")}
        >
          <Icon className="size-10" strokeWidth={2.2} />
        </motion.div>

        <div>
          <div className="flex flex-wrap gap-2">
            <Pill tone="ink">Layer 1 · Quick scan</Pill>
            <Pill>{result.latency_ms.toFixed(1)} ms</Pill>
            <Pill>{result.token_count} tokens</Pill>
          </div>
          <h3 className="mt-3 font-display text-4xl font-extrabold tracking-tight md:text-5xl">
            {real ? "Looks real." : "Looks fake."}
          </h3>
          <p className="mt-1 text-ink-soft">
            {real
              ? "The wording matches patterns from verified wire and editorial reporting."
              : "The wording matches sensational or fabricated news patterns."}
          </p>
        </div>

        <div className="text-left md:text-right">
          <div className="font-display text-6xl font-extrabold tracking-tight">{conf.toFixed(1)}%</div>
          <div className="font-mono text-xs uppercase tracking-wider text-ink-soft">confidence</div>
        </div>
      </div>

      <div className="border-t-2 border-ink bg-white/70 px-6 py-5 md:px-8">
        <div className="mb-2 flex justify-between font-mono text-xs">
          <span>Real {realP.toFixed(1)}%</span>
          <span>Fake {fakeP.toFixed(1)}%</span>
        </div>
        <div className="flex h-4 overflow-hidden rounded-full border-2 border-ink bg-berry">
          <motion.div
            initial={{ width: "50%" }}
            animate={{ width: `${realP}%` }}
            transition={{ ...spring.gentle, delay: 0.2 }}
            className="h-full rounded-r-full bg-mint"
          />
        </div>

        <button
          type="button"
          onClick={() => setOpen((o) => !o)}
          className="mt-4 inline-flex items-center gap-1.5 text-sm font-medium text-ink-soft transition-colors duration-150 hover:text-ink"
          aria-expanded={open}
        >
          <motion.span animate={{ rotate: open ? 180 : 0 }} transition={spring.snappy} className="inline-flex">
            <ChevronDown className="size-4" />
          </motion.span>
          What the model actually saw
        </button>
        <AnimatePresence initial={false}>
          {open && (
            <motion.pre
              initial={{ height: 0, opacity: 0 }}
              animate={{ height: "auto", opacity: 1 }}
              exit={{ height: 0, opacity: 0 }}
              transition={spring.smooth}
              className="mt-3 overflow-hidden whitespace-pre-wrap break-words rounded-2xl bg-ink p-4 font-mono text-xs leading-relaxed text-paper"
            >
              {result.clean_text || "(empty)"}
            </motion.pre>
          )}
        </AnimatePresence>
      </div>
    </motion.article>
  );
});
