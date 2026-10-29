import { useMemo, useState } from "react";
import { AnimatePresence, motion } from "motion/react";
import { Eraser, Globe2, Newspaper, Rocket, Siren, Zap } from "lucide-react";
import { api, type PredictResponse, type VerifyResponse } from "@/lib/api";
import { spring } from "@/lib/motion";
import { cleanText, cn } from "@/lib/utils";
import { PillButton } from "./ui/PillButton";
import { SectionHeading } from "./ui/SectionHeading";
import { Layer1Result } from "./Layer1Result";
import { Layer2Result } from "./Layer2Result";
import { Reveal, RevealItem } from "./ui/Reveal";

const SAMPLES = [
  {
    key: "wire",
    label: "Wire report",
    icon: Newspaper,
    tone: "bg-mint-soft",
    text:
      "WASHINGTON (Reuters) - The Federal Reserve maintained its benchmark overnight interest rate steady on Wednesday while signaling that policymakers continue to monitor inflation and labor market conditions across the United States economy.",
  },
  {
    key: "viral",
    label: "Viral alert",
    icon: Siren,
    tone: "bg-berry-soft",
    text:
      "BREAKING: Secret leaked documents reveal that all global internet satellites will be permanently shut down for 14 days next month as part of an undisclosed mandatory worldwide digital reset, anonymous insiders claim! Share before deleted!",
  },
  {
    key: "claim",
    label: "Space claim",
    icon: Rocket,
    tone: "bg-cobalt-soft",
    text:
      "NASA's James Webb Space Telescope was launched into space on December 25, 2021, from Kourou, French Guiana, aboard an Ariane 5 rocket to study infrared astronomy.",
  },
];

const RAG_STEPS = ["Extracting the core claim", "Searching the live web", "Reading sources", "Weighing the evidence"];

export function Inspector({ onLatency }: { onLatency: (ms: number) => void }) {
  const [text, setText] = useState("");
  const [l1, setL1] = useState<PredictResponse | null>(null);
  const [l2, setL2] = useState<VerifyResponse | null>(null);
  const [busy, setBusy] = useState<"l1" | "l2" | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [step, setStep] = useState(0);

  const stats = useMemo(() => {
    const trimmed = text.trim();
    return {
      words: trimmed ? trimmed.split(/\s+/).length : 0,
      tokens: trimmed ? cleanText(trimmed).split(" ").filter(Boolean).length : 0,
      chars: text.length,
    };
  }, [text]);

  const empty = !text.trim();

  async function runL1() {
    if (empty) return;
    setBusy("l1");
    setError(null);
    try {
      const res = await api.predict(text);
      setL1(res);
      onLatency(res.latency_ms);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(null);
    }
  }

  async function runL2() {
    if (empty) return;
    setBusy("l2");
    setError(null);
    setStep(0);
    // Cosmetic progress ticker while the single request runs server-side.
    const ticker = window.setInterval(() => setStep((s) => Math.min(s + 1, RAG_STEPS.length - 1)), 2600);
    try {
      setL2(await api.verify(text));
    } catch (e) {
      setError((e as Error).message);
    } finally {
      window.clearInterval(ticker);
      setBusy(null);
    }
  }

  function clearAll() {
    setText("");
    setL1(null);
    setL2(null);
    setError(null);
  }

  return (
    <section id="inspect" className="mx-auto w-[min(1160px,calc(100%-2rem))] scroll-mt-28 py-20 md:py-28">
      <SectionHeading
        kicker="The inspector"
        title={
          <>
            Drop in a story.
            <br />
            Get a <span className="text-tomato">verdict.</span>
          </>
        }
        description="Run the fast lexical screen, the evidence-backed deep check, or both. Results stack below as they land."
      />

      <Reveal className="grid gap-6 lg:grid-cols-[1fr_300px]">
        <RevealItem>
          <div className="rounded-[2rem] border-2 border-ink bg-white p-3 shadow-pop-lg md:p-4">
            <div className="flex flex-wrap gap-2 p-2">
              {SAMPLES.map((s) => (
                <motion.button
                  key={s.key}
                  type="button"
                  onClick={() => setText(s.text)}
                  whileHover={{ y: -3, rotate: -1.5 }}
                  whileTap={{ scale: 0.94 }}
                  transition={spring.bouncy}
                  className={cn(
                    "inline-flex items-center gap-2 rounded-full border-2 border-transparent px-4 py-2 text-sm font-medium",
                    "transition-colors duration-150 hover:border-ink",
                    s.tone,
                  )}
                >
                  <s.icon className="size-4" />
                  {s.label}
                </motion.button>
              ))}
              <AnimatePresence>
                {!empty && (
                  <motion.button
                    type="button"
                    onClick={clearAll}
                    initial={{ opacity: 0, scale: 0.6 }}
                    animate={{ opacity: 1, scale: 1 }}
                    exit={{ opacity: 0, scale: 0.6 }}
                    transition={spring.bouncy}
                    whileTap={{ scale: 0.92 }}
                    className="ml-auto inline-flex items-center gap-1.5 rounded-full px-3 py-2 text-sm text-ink-mute transition-colors duration-150 hover:bg-paper hover:text-ink"
                  >
                    <Eraser className="size-4" /> Clear
                  </motion.button>
                )}
              </AnimatePresence>
            </div>

            <label htmlFor="payload" className="sr-only">
              Article or claim text
            </label>
            <textarea
              id="payload"
              value={text}
              onChange={(e) => setText(e.target.value)}
              maxLength={10000}
              rows={8}
              placeholder="Paste a headline, article body, or a single claim…"
              className="block w-full resize-y rounded-[1.5rem] bg-paper px-5 py-4 text-[1.05rem] leading-relaxed outline-none ring-0 transition-shadow duration-150 focus:shadow-[inset_0_0_0_2px_var(--color-ink)]"
            />

            <div className="flex flex-wrap items-center justify-between gap-3 px-2 pb-1 pt-3">
              <div className="flex flex-wrap gap-1.5 font-mono text-[0.72rem] text-ink-mute">
                <Counter label="words" value={stats.words} />
                <Counter label="tokens" value={stats.tokens} />
                <Counter label="chars" value={stats.chars} />
              </div>
              <div className="flex flex-wrap gap-2">
                <PillButton onClick={runL1} loading={busy === "l1"} disabled={empty || busy !== null} icon={<Zap className="size-4" />}>
                  Quick scan
                </PillButton>
                <PillButton
                  variant="ink"
                  onClick={runL2}
                  loading={busy === "l2"}
                  disabled={empty || busy !== null}
                  icon={<Globe2 className="size-4" />}
                >
                  Deep verify
                </PillButton>
              </div>
            </div>
          </div>
        </RevealItem>

        <RevealItem>
          <div className="flex h-full flex-col gap-4">
            <LayerCard
              n="01"
              color="bg-tomato"
              title="Quick scan"
              body="TF-IDF (1,2)-grams into a balanced Logistic Regression. Judges writing style, not facts."
              meta="~10 ms · offline"
            />
            <LayerCard
              n="02"
              color="bg-ink"
              title="Deep verify"
              body="Extracts the claim, searches the web, and has an LLM rule on it using only what it found."
              meta="~10–30 s · needs API keys"
            />
          </div>
        </RevealItem>
      </Reveal>

      <div className="mt-8 space-y-6">
        <AnimatePresence mode="popLayout">
          {error && (
            <motion.div
              key="err"
              layout
              initial={{ opacity: 0, y: 16, scale: 0.97 }}
              animate={{ opacity: 1, y: 0, scale: 1 }}
              exit={{ opacity: 0, scale: 0.95 }}
              transition={spring.bouncy}
              role="alert"
              className="rounded-[1.5rem] border-2 border-ink bg-berry-soft px-6 py-4 text-[#7d0f2e] shadow-pop"
            >
              <span className="font-semibold">Something went wrong.</span> {error}
            </motion.div>
          )}

          {busy === "l2" && (
            <motion.div
              key="progress"
              layout
              initial={{ opacity: 0, y: 16 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, scale: 0.96 }}
              transition={spring.smooth}
              className="flex flex-wrap items-center gap-2 rounded-[1.5rem] border-2 border-dashed border-ink/30 px-5 py-4"
            >
              {RAG_STEPS.map((s, i) => (
                <motion.span
                  key={s}
                  animate={{ scale: i === step ? 1.06 : 1, opacity: i <= step ? 1 : 0.4 }}
                  transition={spring.bouncy}
                  className={cn(
                    "rounded-full px-3.5 py-1.5 text-sm font-medium",
                    i < step ? "bg-mint-soft" : i === step ? "bg-butter" : "bg-paper-deep",
                  )}
                >
                  {i < step ? "✓ " : ""}
                  {s}
                </motion.span>
              ))}
            </motion.div>
          )}

          {l1 && <Layer1Result key="l1" result={l1} />}
          {l2 && <Layer2Result key="l2" result={l2} />}
        </AnimatePresence>
      </div>
    </section>
  );
}

function Counter({ label, value }: { label: string; value: number }) {
  return (
    <span className="rounded-full bg-paper px-2.5 py-1">
      <motion.span key={value} initial={{ y: -6, opacity: 0 }} animate={{ y: 0, opacity: 1 }} transition={spring.snappy} className="inline-block text-ink">
        {value.toLocaleString()}
      </motion.span>{" "}
      {label}
    </span>
  );
}

function LayerCard({ n, color, title, body, meta }: { n: string; color: string; title: string; body: string; meta: string }) {
  return (
    <motion.div
      whileHover={{ y: -4, rotate: -0.6 }}
      transition={spring.bouncy}
      className="group flex-1 rounded-[1.75rem] border-2 border-ink bg-white p-5 shadow-pop"
    >
      <div className="flex items-center gap-3">
        <motion.span
          className={cn("grid size-10 place-items-center rounded-2xl font-mono text-sm font-medium text-paper", color)}
          whileHover={{ scale: 1.15, rotate: -10 }}
          transition={spring.bouncy}
        >
          {n}
        </motion.span>
        <h3 className="font-display text-xl font-bold tracking-tight">{title}</h3>
      </div>
      <p className="mt-3 text-sm leading-relaxed text-ink-soft">{body}</p>
      <p className="mt-3 font-mono text-[0.7rem] text-ink-mute">{meta}</p>
    </motion.div>
  );
}
