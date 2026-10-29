import { motion, useScroll, useTransform } from "motion/react";
import { ArrowDown, Globe2, Zap } from "lucide-react";
import { useRef } from "react";
import { popIn, riseIn, spring, staggerParent } from "@/lib/motion";
import { PillButton } from "./ui/PillButton";

const words = ["Catch", "the", "fake.", "Check", "the", "facts."];

export function Hero({ accuracy, samples }: { accuracy: number; samples: number }) {
  const ref = useRef<HTMLElement>(null);
  const { scrollYProgress } = useScroll({ target: ref, offset: ["start start", "end start"] });
  const stickerY = useTransform(scrollYProgress, [0, 1], [0, -120]);
  const stickerRotate = useTransform(scrollYProgress, [0, 1], [-8, 14]);
  const cardY = useTransform(scrollYProgress, [0, 1], [0, 80]);

  return (
    <section ref={ref} id="top" className="relative mx-auto w-[min(1160px,calc(100%-2rem))] pb-20 pt-20 md:pb-28 md:pt-28">
      <motion.div variants={staggerParent(0.06, 0.15)} initial="hidden" animate="show" className="relative z-10 max-w-4xl">
        <motion.div variants={riseIn}>
          <span className="inline-flex items-center gap-2 rounded-full bg-ink px-4 py-1.5 font-mono text-xs text-paper">
            <span className="size-1.5 rounded-full bg-butter" /> Hybrid ML + live RAG verification
          </span>
        </motion.div>

        <h1 className="mt-7 font-display text-[3.4rem] font-extrabold leading-[0.92] tracking-[-0.035em] sm:text-7xl md:text-[6.4rem]">
          {words.map((w, i) => (
            <motion.span
              key={i}
              variants={riseIn}
              className={
                "mr-[0.22em] inline-block " +
                (w === "fake." ? "text-tomato" : "") +
                (w === "facts." ? "relative" : "")
              }
            >
              {w === "facts." ? (
                <span className="relative">
                  <span className="relative z-10">{w}</span>
                  <motion.span
                    aria-hidden
                    initial={{ scaleX: 0 }}
                    animate={{ scaleX: 1 }}
                    transition={{ ...spring.smooth, delay: 0.9 }}
                    className="absolute inset-x-0 bottom-[0.08em] -z-0 h-[0.32em] origin-left rounded-full bg-butter"
                  />
                </span>
              ) : (
                w
              )}
              {i === 2 && <br />}
            </motion.span>
          ))}
        </h1>

        <motion.p variants={riseIn} className="mt-8 max-w-xl text-lg leading-relaxed text-ink-soft md:text-xl">
          Paste any headline or article. A TF-IDF classifier screens it in milliseconds, then an LLM checks the core claim against live
          web sources and shows its receipts.
        </motion.p>

        <motion.div variants={riseIn} className="mt-10 flex flex-wrap items-center gap-3">
          <PillButton size="lg" icon={<Zap className="size-5" />} onClick={() => document.getElementById("inspect")?.scrollIntoView()}>
            Start inspecting
          </PillButton>
          <PillButton
            size="lg"
            variant="ghost"
            icon={<ArrowDown className="size-5" />}
            onClick={() => document.getElementById("pipeline")?.scrollIntoView()}
          >
            How it works
          </PillButton>
        </motion.div>
      </motion.div>

      {/* Floating stat stickers — scroll-linked transforms */}
      <motion.div
        style={{ y: stickerY, rotate: stickerRotate }}
        className="pointer-events-none absolute right-2 top-24 hidden lg:block"
      >
        <motion.div
          variants={popIn}
          initial="hidden"
          animate="show"
          transition={{ ...spring.bouncy, delay: 0.7 }}
          className="pointer-events-auto grid size-48 place-items-center rounded-full border-2 border-ink bg-butter text-center shadow-pop-lg"
          whileHover={{ scale: 1.08, rotate: 6 }}
        >
          <div>
            <div className="font-display text-5xl font-extrabold tracking-tight">{(accuracy * 100).toFixed(1)}%</div>
            <div className="mt-1 font-mono text-[0.7rem] uppercase tracking-wider">holdout accuracy</div>
          </div>
        </motion.div>
      </motion.div>

      <motion.div style={{ y: cardY }} className="pointer-events-none absolute bottom-16 right-40 hidden lg:block">
        <motion.div
          initial={{ opacity: 0, scale: 0.5, rotate: 12 }}
          animate={{ opacity: 1, scale: 1, rotate: 6 }}
          transition={{ ...spring.bouncy, delay: 0.95 }}
          whileHover={{ rotate: -2, scale: 1.06 }}
          className="pointer-events-auto rounded-[1.75rem] border-2 border-ink bg-white p-5 shadow-pop-lg"
        >
          <div className="flex items-center gap-3">
            <span className="grid size-11 place-items-center rounded-2xl bg-cobalt text-paper">
              <Globe2 className="size-5" />
            </span>
            <div>
              <div className="font-display text-2xl font-extrabold">{samples.toLocaleString()}</div>
              <div className="font-mono text-[0.7rem] text-ink-mute">articles in training corpus</div>
            </div>
          </div>
        </motion.div>
      </motion.div>
    </section>
  );
}
