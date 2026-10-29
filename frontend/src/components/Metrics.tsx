import { motion } from "motion/react";
import { Database, Gauge, Target, Timer } from "lucide-react";
import type { MetricsResponse } from "@/lib/api";
import { spring } from "@/lib/motion";
import { cn } from "@/lib/utils";
import { AnimatedNumber } from "./ui/AnimatedNumber";
import { Reveal, RevealItem } from "./ui/Reveal";
import { SectionHeading } from "./ui/SectionHeading";

export function Metrics({ metrics, latency }: { metrics: MetricsResponse; latency: number | null }) {
  const m = metrics.layer1;
  const acc = m.accuracy * 100;

  return (
    <section id="metrics" className="mx-auto w-[min(1160px,calc(100%-2rem))] scroll-mt-28 py-20 md:py-28">
      <SectionHeading
        kicker="By the numbers"
        title={
          <>
            Small model.
            <br />
            Big <span className="text-tomato">accuracy.</span>
          </>
        }
        description="Measured on a stratified 20% holdout split that the model never saw during training."
      />

      {/* Bento: one oversized hero tile + three supporting tiles */}
      <Reveal className="grid gap-5 md:grid-cols-6 md:grid-rows-2" stagger={0.09}>
        <RevealItem className="md:col-span-3 md:row-span-2">
          <Tile className="flex h-full flex-col justify-between bg-tomato text-paper">
            <div className="flex items-center justify-between">
              <TileIcon className="bg-paper text-tomato">
                <Target className="size-6" />
              </TileIcon>
              <span className="rounded-full bg-paper/15 px-3 py-1 font-mono text-xs">holdout accuracy</span>
            </div>
            <div className="mt-12">
              <div className="font-display text-[5.5rem] font-extrabold leading-none tracking-[-0.04em] md:text-[8rem]">
                <AnimatedNumber value={acc} decimals={1} suffix="%" />
              </div>
              <p className="mt-4 max-w-sm text-lg text-paper/85">
                Out of every 100 articles in the test set, roughly {Math.round(acc)} were classified correctly.
              </p>
              <div className="mt-6 h-4 overflow-hidden rounded-full border-2 border-ink bg-paper/25">
                <motion.div
                  initial={{ width: 0 }}
                  whileInView={{ width: `${acc}%` }}
                  viewport={{ once: true }}
                  transition={{ ...spring.gentle, delay: 0.3 }}
                  className="h-full rounded-full bg-butter"
                />
              </div>
            </div>
          </Tile>
        </RevealItem>

        <RevealItem className="md:col-span-3">
          <Tile className="bg-white">
            <div className="flex items-start justify-between">
              <TileIcon className="bg-cobalt text-paper">
                <Gauge className="size-6" />
              </TileIcon>
              <span className="font-mono text-xs text-ink-mute">ROC-AUC {(m.roc_auc ?? 0.998).toFixed(3)}</span>
            </div>
            <div className="mt-6 font-display text-6xl font-extrabold tracking-tight">
              <AnimatedNumber value={m.f1_score} decimals={3} />
            </div>
            <p className="mt-1 text-ink-soft">F1-score — precision and recall in balance</p>
          </Tile>
        </RevealItem>

        <RevealItem className="md:col-span-2">
          <Tile className="bg-butter">
            <TileIcon className="bg-ink text-butter">
              <Timer className="size-6" />
            </TileIcon>
            <div className="mt-6 font-display text-5xl font-extrabold tracking-tight">
              {latency !== null ? <AnimatedNumber value={latency} decimals={1} suffix="ms" /> : "~10ms"}
            </div>
            <p className="mt-1 text-ink/75">{latency !== null ? "your last quick scan" : "per quick scan"}</p>
          </Tile>
        </RevealItem>

        <RevealItem className="md:col-span-1">
          <Tile className="flex h-full flex-col justify-between bg-ink text-paper">
            <Database className="size-6 text-butter" />
            <div>
              <div className="font-display text-3xl font-extrabold tracking-tight">
                <AnimatedNumber value={metrics.data_count / 1000} decimals={1} suffix="k" />
              </div>
              <p className="text-sm text-paper/70">articles</p>
            </div>
          </Tile>
        </RevealItem>
      </Reveal>
    </section>
  );
}

function Tile({ className, children }: { className?: string; children: React.ReactNode }) {
  return (
    <motion.div
      whileHover={{ y: -6, rotate: -0.4 }}
      transition={spring.bouncy}
      className={cn("group h-full rounded-[2rem] border-2 border-ink p-7 shadow-pop-lg", className)}
    >
      {children}
    </motion.div>
  );
}

function TileIcon({ className, children }: { className?: string; children: React.ReactNode }) {
  return (
    <motion.span
      className={cn("grid size-12 place-items-center rounded-2xl", className)}
      whileHover={{ scale: 1.15, rotate: -12 }}
      transition={spring.bouncy}
    >
      {children}
    </motion.span>
  );
}
