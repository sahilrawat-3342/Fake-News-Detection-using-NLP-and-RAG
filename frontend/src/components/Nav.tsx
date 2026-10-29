import { motion } from "motion/react";
import { ScanSearch } from "lucide-react";
import type { Health } from "@/lib/api";
import { spring } from "@/lib/motion";

const links = [
  { href: "#inspect", label: "Inspect" },
  { href: "#metrics", label: "Metrics" },
  { href: "#benchmarks", label: "Benchmarks" },
  { href: "#pipeline", label: "How it works" },
];

export function Nav({ health, apiError }: { health: Health | null; apiError: boolean }) {
  const l1 = apiError ? "offline" : health?.model_online ? "online" : "loading";
  const l2 = apiError ? "offline" : health?.rag_configured ? "ready" : health ? "keys missing" : "loading";

  return (
    <motion.header
      initial={{ y: -40, opacity: 0 }}
      animate={{ y: 0, opacity: 1 }}
      transition={spring.smooth}
      className="sticky top-4 z-50 mx-auto w-[min(1160px,calc(100%-2rem))]"
    >
      <nav className="flex items-center justify-between gap-4 rounded-full border-2 border-ink bg-white/90 py-2 pl-2 pr-3 shadow-pop backdrop-blur">
        <a href="#top" className="group flex items-center gap-2.5">
          <motion.span
            whileHover={{ rotate: -14, scale: 1.1 }}
            transition={spring.bouncy}
            className="grid size-10 place-items-center rounded-full bg-tomato text-paper"
          >
            <ScanSearch className="size-5" strokeWidth={2.4} />
          </motion.span>
          <span className="font-display text-lg font-extrabold tracking-tight">TruthLens</span>
        </a>

        <ul className="hidden items-center gap-1 md:flex">
          {links.map((l) => (
            <li key={l.href}>
              <motion.a
                href={l.href}
                whileHover={{ y: -2 }}
                transition={spring.snappy}
                className="block rounded-full px-4 py-2 text-sm font-medium text-ink-soft transition-colors duration-150 hover:bg-paper-deep hover:text-ink"
              >
                {l.label}
              </motion.a>
            </li>
          ))}
        </ul>

        <div className="flex items-center gap-1.5 font-mono text-[0.7rem]">
          <StatusDot label="L1" state={l1} good={l1 === "online"} />
          <StatusDot label="L2" state={l2} good={l2 === "ready"} />
        </div>
      </nav>
    </motion.header>
  );
}

function StatusDot({ label, state, good }: { label: string; state: string; good: boolean }) {
  const color = state === "loading" ? "var(--color-ink-mute)" : good ? "var(--color-mint)" : "var(--color-berry)";
  return (
    <span className="hidden items-center gap-1.5 rounded-full bg-paper px-2.5 py-1.5 sm:inline-flex" title={`${label}: ${state}`}>
      <span className="relative flex size-2">
        {good && <span className="absolute inset-0 animate-ping rounded-full opacity-60" style={{ background: color }} />}
        <span className="relative size-2 rounded-full" style={{ background: color }} />
      </span>
      <span className="text-ink-soft">
        {label} · {state}
      </span>
    </span>
  );
}
