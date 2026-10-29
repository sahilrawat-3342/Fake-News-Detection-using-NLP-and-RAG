import { useState } from "react";
import { LayoutGroup, motion } from "motion/react";
import type { MetricsResponse } from "@/lib/api";
import { spring } from "@/lib/motion";
import { cn } from "@/lib/utils";
import { Reveal, RevealItem } from "./ui/Reveal";
import { SectionHeading } from "./ui/SectionHeading";

const METRIC_TABS = [
  { key: "accuracy", label: "Accuracy" },
  { key: "precision", label: "Precision" },
  { key: "recall", label: "Recall" },
  { key: "f1_score", label: "F1" },
] as const;

const MODEL_ORDER = [
  { key: "truthlens_layer1", color: "bg-tomato", short: "TruthLens (LogReg)" },
  { key: "svm_baseline", color: "bg-cobalt", short: "Linear SVM" },
  { key: "naive_bayes_baseline", color: "bg-butter", short: "Naive Bayes" },
];

export function Benchmarks({ metrics }: { metrics: MetricsResponse }) {
  const [metric, setMetric] = useState<(typeof METRIC_TABS)[number]["key"]>("accuracy");
  const cm = metrics.layer1.confusion_matrix ?? { true_negatives: 3397, false_positives: 82, false_negatives: 31, true_positives: 4153 };
  const total = cm.true_negatives + cm.false_positives + cm.false_negatives + cm.true_positives;

  // Zoom the bar axis so small differences between strong models stay visible.
  const FLOOR = 90;

  return (
    <section id="benchmarks" className="mx-auto w-[min(1160px,calc(100%-2rem))] scroll-mt-28 py-20 md:py-28">
      <SectionHeading
        kicker="Head to head"
        title={
          <>
            How it stacks up
            <br />
            against <span className="text-cobalt">baselines.</span>
          </>
        }
      />

      <Reveal className="grid gap-6 lg:grid-cols-[1.35fr_1fr]">
        <RevealItem>
          <div className="h-full rounded-[2rem] border-2 border-ink bg-white p-6 shadow-pop-lg md:p-8">
            <LayoutGroup id="metric-tabs">
              <div className="inline-flex flex-wrap gap-1 rounded-full bg-paper p-1.5" role="tablist">
                {METRIC_TABS.map((t) => (
                  <button
                    key={t.key}
                    role="tab"
                    aria-selected={metric === t.key}
                    onClick={() => setMetric(t.key)}
                    className="relative rounded-full px-4 py-2 text-sm font-medium"
                  >
                    {metric === t.key && (
                      <motion.span layoutId="tab-pill" transition={spring.bouncy} className="absolute inset-0 rounded-full bg-ink" />
                    )}
                    <span className={cn("relative transition-colors duration-150", metric === t.key ? "text-paper" : "text-ink-soft")}>
                      {t.label}
                    </span>
                  </button>
                ))}
              </div>
            </LayoutGroup>

            <div className="mt-8 space-y-6">
              {MODEL_ORDER.map((mo) => {
                const model = metrics.models[mo.key];
                if (!model) return null;
                const val = model.metrics[metric] * 100;
                const width = Math.max(4, ((val - FLOOR) / (100 - FLOOR)) * 100);
                return (
                  <div key={mo.key}>
                    <div className="mb-2 flex items-baseline justify-between gap-3">
                      <div>
                        <span className="font-semibold">{mo.short}</span>
                        <span className="ml-2 font-mono text-xs text-ink-mute">{model.algorithm}</span>
                      </div>
                      <motion.span key={metric + mo.key} initial={{ y: -8, opacity: 0 }} animate={{ y: 0, opacity: 1 }} transition={spring.bouncy} className="font-mono text-sm font-medium">
                        {val.toFixed(2)}%
                      </motion.span>
                    </div>
                    <div className="h-9 overflow-hidden rounded-full border-2 border-ink bg-paper">
                      <motion.div
                        className={cn("h-full rounded-full border-r-2 border-ink", mo.color)}
                        initial={{ width: 0 }}
                        animate={{ width: `${width}%` }}
                        transition={spring.bouncy}
                      />
                    </div>
                  </div>
                );
              })}
            </div>
            <p className="mt-6 font-mono text-[0.7rem] text-ink-mute">Axis starts at {FLOOR}% to make differences legible.</p>
          </div>
        </RevealItem>

        <RevealItem>
          <div className="h-full rounded-[2rem] border-2 border-ink bg-ink p-6 text-paper shadow-pop-lg md:p-8">
            <div className="flex items-baseline justify-between">
              <h3 className="font-display text-2xl font-bold tracking-tight">Where it slips up</h3>
              <span className="font-mono text-xs text-paper/60">n = {total.toLocaleString()}</span>
            </div>
            <p className="mt-1 text-sm text-paper/70">TruthLens confusion matrix on the holdout split.</p>

            <div className="mt-6 grid grid-cols-[auto_1fr_1fr] gap-2 text-sm">
              <span />
              <span className="pb-1 text-center font-mono text-[0.7rem] text-paper/60">predicted fake</span>
              <span className="pb-1 text-center font-mono text-[0.7rem] text-paper/60">predicted real</span>

              <span className="self-center pr-2 font-mono text-[0.7rem] text-paper/60 [writing-mode:vertical-rl] rotate-180">actually fake</span>
              <Cell value={cm.true_negatives} total={total} label="caught" className="bg-mint text-ink" />
              <Cell value={cm.false_positives} total={total} label="missed" className="bg-berry text-paper" />

              <span className="self-center pr-2 font-mono text-[0.7rem] text-paper/60 [writing-mode:vertical-rl] rotate-180">actually real</span>
              <Cell value={cm.false_negatives} total={total} label="wrongly flagged" className="bg-butter text-ink" />
              <Cell value={cm.true_positives} total={total} label="cleared" className="bg-mint text-ink" />
            </div>
          </div>
        </RevealItem>
      </Reveal>
    </section>
  );
}

function Cell({ value, total, label, className }: { value: number; total: number; label: string; className: string }) {
  return (
    <motion.div
      whileHover={{ scale: 1.05, rotate: -1 }}
      transition={spring.bouncy}
      className={cn("rounded-[1.25rem] p-4", className)}
    >
      <div className="font-display text-3xl font-extrabold tracking-tight">{value.toLocaleString()}</div>
      <div className="mt-1 text-xs font-medium opacity-80">
        {label} · {((value / total) * 100).toFixed(1)}%
      </div>
    </motion.div>
  );
}
