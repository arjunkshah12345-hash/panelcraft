/** The generated comic, one panel per beat, with each panel's similarity meter. */
export default function PanelGrid({ panels, report }) {
  if (!panels?.length) return null;

  // Match panels to their scores by index; a sweep result keys them by panel_id.
  const scoreFor = (index, panelId) => {
    if (!report?.panels?.length) return null;
    return (
      report.panels.find((p) => p.panel_id === panelId) ?? report.panels[index] ?? null
    );
  };

  return (
    <div className="panels">
      {panels.map((panel, index) => {
        const score = scoreFor(index, panel.panel_id ?? `panel-${index + 1}`);
        const value = score?.sim_to_reference;
        // Cosine similarity on face embeddings lives in roughly [-1, 1]; clamp so a
        // negative score cannot render as a negative-width bar.
        const width = value == null ? null : `${Math.max(0, Math.min(1, value)) * 100}%`;
        const drifted = score?.verdict === "drift";

        return (
          <article className="panel" key={panel.image_url ?? index}>
            <figure>
              <img src={panel.image_url} alt={panel.caption || panel.beat || `Panel ${index + 1}`} loading="lazy" />
              <figcaption>{panel.caption || panel.beat || `Panel ${index + 1}`}</figcaption>
            </figure>

            {width !== null && (
              <>
                <div className={`meter${drifted ? " warn" : ""}`}>
                  <span style={{ width }} />
                </div>
                <div
                  style={{
                    fontFamily: "var(--mono)",
                    fontSize: 11,
                    color: drifted ? "var(--warn)" : "var(--muted)",
                    display: "flex",
                    justifyContent: "space-between",
                  }}
                >
                  <span>{score.verdict}</span>
                  <span>{value.toFixed(3)}</span>
                </div>
              </>
            )}
          </article>
        );
      })}
    </div>
  );
}