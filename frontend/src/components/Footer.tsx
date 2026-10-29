import { useState } from "react";
import { motion } from "motion/react";
import { RefreshCw } from "lucide-react";
import { api, type Health, type RetrainResponse } from "@/lib/api";
import { spring } from "@/lib/motion";
import { PillButton } from "./ui/PillButton";

export function Footer({ health, onRetrained }: { health: Health | null; onRetrained: (r: RetrainResponse) => void }) {
  const [busy, setBusy] = useState(false);
  const [msg, setMsg] = useState<string | null>(null);

  async function retrain() {
    setBusy(true);
    setMsg(null);
    try {
      const r = await api.retrain();
      onRetrained(r);
      setMsg(`Retrained in ${r.train_time} — ${(r.accuracy * 100).toFixed(2)}% accuracy on ${r.data_count.toLocaleString()} articles.`);
    } catch (e) {
      setMsg((e as Error).message);
    } finally {
      setBusy(false);
    }
  }

  return (
    <footer className="mx-auto w-[min(1160px,calc(100%-2rem))] pb-10 pt-10">
      <motion.div
        initial={{ opacity: 0, y: 40, scale: 0.96 }}
        whileInView={{ opacity: 1, y: 0, scale: 1 }}
        viewport={{ once: true }}
        transition={spring.bouncy}
        className="rounded-[2.5rem] border-2 border-ink bg-ink p-8 text-paper shadow-pop-lg md:p-12"
      >
        <div className="flex flex-col gap-8 md:flex-row md:items-end md:justify-between">
          <div className="max-w-lg">
            <h2 className="font-display text-4xl font-extrabold leading-tight tracking-tight md:text-5xl">
              Fresh data?
              <br />
              <span className="text-butter">Retrain in one click.</span>
            </h2>
            <p className="mt-4 text-paper/70">
              Rebuilds the TF-IDF vocabulary and refits the classifier from <code className="font-mono text-paper">data/</code>. Runs on the
              local API server only.
            </p>
            {msg && <p className="mt-4 rounded-2xl bg-paper/10 px-4 py-3 text-sm">{msg}</p>}
          </div>
          <PillButton
            variant="primary"
            size="lg"
            onClick={retrain}
            loading={busy}
            disabled={health ? !health.retrain_available : false}
            icon={<RefreshCw className="size-5" />}
          >
            {health && !health.retrain_available ? "Datasets not deployed" : "Retrain model"}
          </PillButton>
        </div>

        <div className="mt-12 flex flex-wrap items-center justify-between gap-4 border-t border-paper/15 pt-6 font-mono text-xs text-paper/50">
          <span>TruthLens · Fake News Detection using NLP and RAG</span>
          <span>LLM: {health?.llm_model ?? "—"}</span>
        </div>
      </motion.div>
    </footer>
  );
}
