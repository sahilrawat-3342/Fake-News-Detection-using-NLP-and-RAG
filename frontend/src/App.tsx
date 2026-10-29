import { useEffect, useState } from "react";
import { api, FALLBACK_METRICS, type Health, type MetricsResponse } from "@/lib/api";
import { Nav } from "./components/Nav";
import { Hero } from "./components/Hero";
import { Inspector } from "./components/Inspector";
import { Metrics } from "./components/Metrics";
import { Benchmarks } from "./components/Benchmarks";
import { Pipeline } from "./components/Pipeline";
import { Footer } from "./components/Footer";

export default function App() {
  const [health, setHealth] = useState<Health | null>(null);
  const [apiError, setApiError] = useState(false);
  const [metrics, setMetrics] = useState<MetricsResponse>(FALLBACK_METRICS);
  const [latency, setLatency] = useState<number | null>(null);

  useEffect(() => {
    api.health().then(setHealth).catch(() => setApiError(true));
    api
      .metrics()
      .then((m) => {
        if (m.layer1 && Object.keys(m.layer1).length) setMetrics(m);
      })
      .catch(() => {
        /* keep bundled snapshot */
      });
  }, []);

  return (
    <div className="relative overflow-x-clip">
      <Nav health={health} apiError={apiError} />
      <main>
        <Hero accuracy={metrics.layer1.accuracy} samples={metrics.data_count} />
        <Inspector onLatency={setLatency} />
        <Metrics metrics={metrics} latency={latency} />
        <Benchmarks metrics={metrics} />
        <Pipeline />
      </main>
      <Footer
        health={health}
        onRetrained={(r) =>
          setMetrics((m) => ({ ...m, data_count: r.data_count, train_time: r.train_time, layer1: { ...m.layer1, accuracy: r.accuracy } }))
        }
      />
    </div>
  );
}
