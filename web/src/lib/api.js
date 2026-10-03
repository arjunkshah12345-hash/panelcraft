const JSON_HEADERS = { "content-type": "application/json" };

async function request(path, options) {
  const response = await fetch(path, options);
  if (!response.ok) {
    let detail = `${response.status} ${response.statusText}`;
    try {
      const body = await response.json();
      if (body?.detail) detail = body.detail;
    } catch {
      /* response had no JSON body; the status line is all we have */
    }
    throw new Error(detail);
  }
  return response.json();
}

export const getHealth = () => request("/api/health");
export const getCapabilities = () => request("/api/capabilities");
export const getDefaultScript = () => request("/api/default-script");

export const generateComic = (body) =>
  request("/api/generate", { method: "POST", headers: JSON_HEADERS, body: JSON.stringify(body) });

export const runSweep = (body) =>
  request("/api/sweep", { method: "POST", headers: JSON_HEADERS, body: JSON.stringify(body) });

export const getJob = (id) => request(`/api/jobs/${id}`);
export const getResult = (id) => request(`/api/results/${id}`);
export const listResults = () => request("/api/results");

/** Poll a sweep job until it settles, reporting progress along the way. */
export async function awaitJob(id, onProgress) {
  for (;;) {
    const job = await getJob(id);
    onProgress?.(job);
    if (job.status === "done") return job;
    if (job.status === "error") throw new Error(job.error ?? "sweep failed");
    await new Promise((resolve) => setTimeout(resolve, 1500));
  }
}