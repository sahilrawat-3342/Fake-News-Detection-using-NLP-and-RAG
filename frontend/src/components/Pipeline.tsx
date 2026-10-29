import { useRef } from "react";
import { motion, useScroll, useSpring } from "motion/react";
import { spring } from "@/lib/motion";
import { cn } from "@/lib/utils";
import { Reveal, RevealItem } from "./ui/Reveal";
import { SectionHeading } from "./ui/SectionHeading";

const STAGES = [
  {
    n: "01",
    title: "Ingest",
    body: "Fake.csv and True.csv are merged, labelled 0/1, and deduplicated.",
    code: 'df = pd.concat([fake_df, true_df])\ndf = df.drop_duplicates(subset="text")',
    color: "bg-cobalt",
  },
  {
    n: "02",
    title: "Normalize",
    body: "Lowercase, strip URLs and HTML, drop punctuation, collapse whitespace.",
    code: "text = re.sub(r'https?://\\S+', ' ', text)\ntext = re.sub(r'[^a-z0-9\\s]', ' ', text)",
    color: "bg-butter",
  },
  {
    n: "03",
    title: "Vectorize",
    body: "TF-IDF over unigrams and bigrams turns each article into a sparse vector.",
    code: "TfidfVectorizer(ngram_range=(1, 2),\n  max_df=0.80, min_df=3, sublinear_tf=True)",
    color: "bg-mint",
  },
  {
    n: "04",
    title: "Classify",
    body: "A class-balanced Logistic Regression draws the real/fake boundary.",
    code: 'LogisticRegression(solver="liblinear",\n  class_weight="balanced")',
    color: "bg-tomato",
  },
  {
    n: "05",
    title: "Verify live",
    body: "An LLM extracts the claim, SerpAPI finds sources, and the verdict cites them.",
    code: "claim = extract_claim_and_query(text)\nhits = search_web(claim.query)\nverify_claim_with_context(claim, hits)",
    color: "bg-berry",
  },
];

export function Pipeline() {
  const ref = useRef<HTMLDivElement>(null);
  const { scrollYProgress } = useScroll({ target: ref, offset: ["start 70%", "end 60%"] });
  const progress = useSpring(scrollYProgress, { stiffness: 120, damping: 24 });

  return (
    <section id="pipeline" className="mx-auto w-[min(1160px,calc(100%-2rem))] scroll-mt-28 py-20 md:py-28">
      <SectionHeading
        kicker="Under the hood"
        title={
          <>
            Five steps from
            <br />
            raw text to <span className="text-tomato">receipts.</span>
          </>
        }
      />

      <div ref={ref} className="relative">
        {/* Scroll-linked progress rail */}
        <div className="absolute bottom-6 left-[27px] top-6 w-1.5 overflow-hidden rounded-full bg-paper-deep md:left-[31px]" aria-hidden>
          <motion.div style={{ scaleY: progress }} className="h-full w-full origin-top rounded-full bg-ink" />
        </div>

        <Reveal className="space-y-5" stagger={0.1}>
          {STAGES.map((s, i) => (
            <RevealItem key={s.n}>
              <motion.div
                whileHover={{ x: 6 }}
                transition={spring.bouncy}
                className="group relative grid gap-5 pl-20 md:grid-cols-[1fr_1.1fr] md:items-center md:pl-24"
              >
                <motion.span
                  className={cn(
                    "absolute left-0 top-1 grid size-14 place-items-center rounded-full border-2 border-ink font-mono text-sm font-medium shadow-pop md:size-16",
                    s.color,
                    s.color === "bg-tomato" || s.color === "bg-cobalt" || s.color === "bg-berry" ? "text-paper" : "text-ink",
                  )}
                  whileHover={{ scale: 1.15, rotate: -12 }}
                  transition={spring.bouncy}
                >
                  {s.n}
                </motion.span>
                <div className={cn("py-2", i % 2 === 1 && "md:order-2")}>
                  <h3 className="font-display text-3xl font-extrabold tracking-tight">{s.title}</h3>
                  <p className="mt-2 max-w-md text-ink-soft">{s.body}</p>
                </div>
                <pre className={cn("overflow-x-auto rounded-[1.5rem] border-2 border-ink bg-ink p-5 font-mono text-[0.8rem] leading-relaxed text-paper shadow-pop", i % 2 === 1 && "md:order-1")}>
                  {s.code}
                </pre>
              </motion.div>
            </RevealItem>
          ))}
        </Reveal>
      </div>
    </section>
  );
}
