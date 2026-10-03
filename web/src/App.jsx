import { useCallback, useEffect, useState } from "react";
import PanelGrid from "./components/PanelGrid.jsx";
import TradeoffChart from "./components/TradeoffChart.jsx";
import {
  awaitJob,
  generateComic,
  getCapabilities,
  getDefaultScript,
  getResult,
  runSweep,
} from "./lib/api.js";

export default function App() {
  const [providers, setProviders] = useState([]);
  const [provider, setProvider] = useState("synthetic");
  const [script, setScript] = useState(null);
  const [strength, setStrength] = useState(0.75);
  const [comic, setComic] = useState(null);
  const [sweep, setSweep] = useState(null);
  const [job, setJob] = useState(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    getCapabilities().then(setProviders).catch((e) => setError(e.message));
    getDefaultScript().then(setScript).catch((e) => setError(e.message));
  }, []);

  const selected = providers.find((p) => p.name === provider);
  const available = selected && selected.identity_control !== "unavailable";

  const onGenerate = useCallback(async () => {
    if (!script) return;
    setBusy(true);
    setError(null);
    setComic(null);
    try {
      setComic(await generateComic({ script, provider, identity_strength: strength }));
    } catch (e) {
      setError(e.message);
    } finally {
      setBusy(false);
    }
  }, [script, provider, strength]);

  const onSweep = useCallback(async () => {
    if (!script) return;
    setBusy(true);
    setError(null);
    setSweep(null);
    try {
      const started = await runSweep({
        script,
        provider,
        strengths: [0, 0.25, 0.5, 0.75, 1],
        panels_per_condition: 4,
      });
      const finished = await awaitJob(started.id, setJob);
      setSweep(await getResult(finished.result_id));
    } catch (e) {
      setError(e.message);
    } finally {
      setBusy(false);
      setJob(null);
    }
  }, [script, provider]);

  return (
    <div className="shell">
      <header className="masthead">
        <p className="eyebrow">ForgeHacks 2026 · AI + Creativity</p>
        <h1>
          How much character identity can you buy <em>before you lose the panel?</em>
        </h1>
        <p>
          PanelCraft renders the same comic script across a sweep of identity-conditioning
          strength, measures each panel against reference portraits with a face embedding
          model, and plots the tradeoff against how varied the panels actually are.
        </p>
      </header>

      <div className="controls">
        <div className="field">
          <label htmlFor="provider">Generator</label>
          <select
            id="provider"
            value={provider}
            onChange={(e) => setProvider(e.target.value)}
          >
            {providers.map((p) => (
              <option key={p.name} value={p.name}>
                {p.name}
                {p.identity_control === "unavailable" ? " — needs API key" : ""}
              </option>
            ))}
          </select>
        </div>

        <div className="field">
          <label htmlFor="script">Script</label>
          <input
            id="script"
            type="text"
            value={script?.title ?? ""}
            disabled={!script}
            onChange={(e) => setScript({ ...script, title: e.target.value })}
          />
        </div>

        <div className="field">
          <label htmlFor="strength">Identity strength</label>
          <div className="strength-readout">{strength.toFixed(2)}</div>
          <input
            id="strength"
            type="range"
            min="0"
            max="1"
            step="0.05"
            value={strength}
            onChange={(e) => setStrength(Number(e.target.value))}
          />
          <div className="strength-scale">
            <span>0 · free</span>
            <span>1 · locked</span>
          </div>
        </div>
      </div>

      <div className="actions">
        <button onClick={onGenerate} disabled={busy || !available || !script}>
          {busy ? "Working…" : "Render comic"}
        </button>
        <button className="ghost" onClick={onSweep} disabled={busy || !available || !script}>
          Run full sweep
        </button>
      </div>

      {selected && !available && (
        <p className="callout">
          <strong>{provider}</strong> is unavailable: {selected.notes}. Set the provider's
          API key and restart the backend.
        </p>
      )}

      {job && (
        <>
          <p className="progress">
            {job.message} · {Math.round(job.progress * 100)}%
          </p>
          <div className="bar">
            <span style={{ width: `${job.progress * 100}%` }} />
          </div>
        </>
      )}

      {error && <p className="error">{error}</p>}

      {comic && (
        <section>
          <div className="section-head">
            <h2>Comic · {comic.provider}</h2>
            <p className="note">
              {comic.script.title} — {comic.script.character.name} at identity strength{" "}
              {comic.identity_strength.toFixed(2)}
              {comic.report && (
                <>
                  . Identity {comic.report.identity_mean.toFixed(3)} mean,{" "}
                  {comic.report.identity_min.toFixed(3)} worst panel against reference
                  portraits.
                </>
              )}
            </p>
          </div>
          <PanelGrid panels={comic.panels} report={comic.report} />
        </section>
      )}

      {sweep && (
        <section>
          <div className="section-head">
            <h2>Tradeoff curve</h2>
            <p className="note">
              {sweep.panels_per_condition} panels per condition, seeds fixed across the sweep.
            </p>
          </div>

          <div className="result-grid">
            <TradeoffChart
              tradeoff={sweep.tradeoff}
              recommended={sweep.recommended_strength}
              trustworthy={sweep.identity_axis_trustworthy}
            />

            <div>
              <dl className="stats">
                <div className="stat">
                  <dt>Provider</dt>
                  <dd>{sweep.provider}</dd>
                </div>
                <div className="stat">
                  <dt>Recommended</dt>
                  <dd>
                    {sweep.recommended_strength == null
                      ? "not reported"
                      : sweep.recommended_strength.toFixed(2)}
                  </dd>
                </div>
                <div className="stat">
                  <dt>AUROC (in-sweep)</dt>
                  <dd className={sweep.calibration?.auroc < 0.9 ? "warn" : undefined}>
                    {sweep.calibration ? sweep.calibration.auroc.toFixed(3) : "—"}
                  </dd>
                </div>
                <div className="stat">
                  <dt>Threshold</dt>
                  <dd>{sweep.calibration ? sweep.calibration.threshold.toFixed(3) : "—"}</dd>
                </div>
                {sweep.harness_validation && (
                  <div className="stat">
                    <dt>Harness vs truth</dt>
                    <dd>
                      ρ{" "}
                      {sweep.harness_validation.spearman_true_drift_vs_measured_similarity.toFixed(2)}
                    </dd>
                  </div>
                )}
              </dl>

              <div className={`callout${sweep.identity_axis_trustworthy ? " ok" : ""}`}>
                <strong>
                  {sweep.identity_axis_trustworthy
                    ? "Identity axis is measuring what it claims."
                    : "Identity axis is not trustworthy on this material."}
                </strong>{" "}
                {sweep.identity_axis_note}
              </div>

              {sweep.tradeoff.some((p) => p.true_identity_distance != null) && (
                <div className="callout ok">
                  <strong>Ground truth</strong> — the generator knows exactly how far it
                  moved each identity, so the true curve is available without trusting the
                  measurement:{" "}
                  {sweep.tradeoff
                    .map(
                      (p) =>
                        `${p.identity_strength} → ${p.true_identity_distance?.toFixed(2)}`,
                    )
                    .join("  ·  ")}
                </div>
              )}

              {sweep.notes?.length > 0 && (
                <ul className="notes">
                  {sweep.notes.map((note, i) => (
                    <li key={i}>{note}</li>
                  ))}
                </ul>
              )}
            </div>
          </div>
        </section>
      )}
    </div>
  );
}