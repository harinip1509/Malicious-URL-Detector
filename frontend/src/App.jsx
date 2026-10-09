import { useEffect, useState } from "react";
import "./styles.css";

// Dev: Vite on 5173 calls Flask on 5050. Build: Flask serves the UI, same origin.
// Port 5050 rather than 5000 because macOS AirPlay Receiver holds 5000.
const API = import.meta.env.VITE_API ?? (import.meta.env.DEV ? "http://localhost:5050" : "");
// Set when building the hosted copy that runs the demo scorer in the browser.
const HOSTED = import.meta.env.VITE_HOSTED === "1";

const SAMPLES = [
  "https://github.com/explore",
  "http://192.168.4.11:8080/secure/login/verify.php",
  "http://paypal-account-verify.ddns.net/webscr?cmd=_login",
];

const BATCH_EXAMPLE = [
  "https://github.com/explore",
  "https://en.wikipedia.org/wiki/Transformer_(deep_learning_architecture)",
  "https://docs.python.org/3/library/re.html",
  "http://192.168.4.11:8080/secure/login/verify.php",
  "http://paypal-account-verify.ddns.net/webscr?cmd=_login",
  "http://secure.account.update.bankofamerica.com-verify.info/login",
  "https://bit.ly/3xYz9Qa",
  "https://www.google.com/search?q=weather",
].join("\n");

const GROUPS = [
  { label: "Semantic branch", note: "fine-tuned DistilBERT on URL tokens", models: ["bert"] },
  {
    label: "Structural branch",
    note: "34 lexical features + BERT embedding",
    models: ["xgboost", "lightgbm", "catboost", "adaboost"],
  },
];

const NAMES = {
  bert: "DistilBERT",
  xgboost: "XGBoost",
  lightgbm: "LightGBM",
  catboost: "CatBoost",
  adaboost: "AdaBoost",
};

async function post(urls) {
  const r = await fetch(`${API}/api/scan`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ urls }),
  });
  const data = await r.json();
  if (!r.ok) throw new Error(data.error || "The scan did not complete.");
  return data;
}

/* ------------------------------------------------------------------ pieces */

function Diverging({ p, c = 1, strong = false }) {
  const side = p >= 0.5 ? "risk" : "safe";
  const width = Math.abs(p - 0.5) * 100;
  return (
    <div className={`track ${strong ? "strong" : ""}`}>
      <div
        className={`fill ${side}`}
        style={{ width: `${width}%`, opacity: strong ? 1 : 0.3 + c * 0.7 }}
      />
    </div>
  );
}

function Axis() {
  return (
    <div className="axis" aria-hidden="true">
      <span>benign</span>
      <span>0.5</span>
      <span>malicious</span>
    </div>
  );
}

function Meter({ score }) {
  return (
    <div className="meter" aria-label={`Risk score ${score.toFixed(3)}`}>
      <div className="meter-bar">
        <div className="meter-mark" style={{ left: `${score * 100}%` }} />
      </div>
      <div className="meter-scale">
        <span>0.0</span>
        <span>threshold 0.5</span>
        <span>1.0</span>
      </div>
    </div>
  );
}

function Breakdown({ result }) {
  const byName = Object.fromEntries(result.models.map((m) => [m.model, m]));
  return (
    <section className="breakdown">
      <header className="panel-head">
        <h3>Ensemble breakdown</h3>
        <p>
          Each bar is one model's vote. Its strength fades with that model's confidence
          c = 2|p − 0.5|, which is how much the meta-learner lets it count.
        </p>
      </header>

      <div className="grid-head">
        <span>Model</span>
        <span />
        <span className="num">p</span>
        <span className="num">c</span>
      </div>

      {GROUPS.map((g) => (
        <div className="group" key={g.label}>
          <div className="group-label">
            {g.label}
            <span>{g.note}</span>
          </div>
          {g.models.map((k) => {
            const m = byName[k];
            if (!m) return null;
            return (
              <div className="row" key={k}>
                <span className="name">{NAMES[k]}</span>
                <Diverging p={m.probability} c={m.confidence} />
                <span className="num">{m.probability.toFixed(3)}</span>
                <span className="num dim">{m.confidence.toFixed(2)}</span>
              </div>
            );
          })}
        </div>
      ))}

      <div className="group final">
        <div className="group-label">
          Meta-learner
          <span>logistic regression over confidence-weighted votes</span>
        </div>
        <div className="row">
          <span className="name">Final</span>
          <Diverging p={result.score} strong />
          <span className="num">{result.score.toFixed(3)}</span>
          <span className="num dim">{result.confidence.toFixed(2)}</span>
        </div>
      </div>

      <div className="row axis-row">
        <span />
        <Axis />
        <span />
        <span />
      </div>
    </section>
  );
}

function Result({ result, elapsed, demo }) {
  const malicious = result.verdict === "malicious";
  const agree = Math.round(result.agreement * 5);
  return (
    <article className={`result ${result.verdict}`}>
      <div className="summary">
        <div className="summary-main">
          <p className="eyebrow">Verdict</p>
          <h2>{malicious ? "Likely malicious" : "Looks safe"}</h2>
          <p className="summary-sub">
            {agree} of 5 models agree
            <span className="sep" />
            {elapsed} ms
            {demo && (
              <>
                <span className="sep" />
                simulated
              </>
            )}
          </p>
        </div>
        <div className="summary-score">
          <p className="eyebrow">Risk score</p>
          <p className="big">{result.score.toFixed(3)}</p>
        </div>
      </div>

      <Meter score={result.score} />
      <p className="scanned">{result.url}</p>

      <div className="result-body">
        <Breakdown result={result} />

        <aside className="signals">
          <header className="panel-head">
            <h3>Signals</h3>
            <p>Domain reputation and structural features behind the verdict.</p>
          </header>
          {result.signals.length ? (
            <ol>
              {result.signals.map((s) => (
                <li key={s}>{s}</li>
              ))}
            </ol>
          ) : (
            <p className="empty">
              Nothing unusual in the structure. The verdict rests on the learned models
              rather than any single obvious signal.
            </p>
          )}
        </aside>
      </div>
    </article>
  );
}

function VoteDots({ models }) {
  return (
    <span className="dots" aria-label="model votes">
      {models.map((m) => (
        <i
          key={m.model}
          title={`${NAMES[m.model]} ${m.probability.toFixed(3)}`}
          className={m.vote === "malicious" ? "risk" : "safe"}
          style={{ opacity: 0.3 + m.confidence * 0.7 }}
        />
      ))}
    </span>
  );
}

function BatchTable({ rows, elapsed }) {
  const flagged = rows.filter((r) => r.verdict === "malicious").length;
  return (
    <article className="batch-result">
      <header className="batch-head">
        <p>
          <b>{flagged}</b> of <b>{rows.length}</b> links flagged
        </p>
        <p className="dim">{elapsed} ms total</p>
      </header>
      <div className="table-wrap">
        <table>
          <thead>
            <tr>
              <th>URL</th>
              <th>Verdict</th>
              <th className="num">Score</th>
              <th>Votes</th>
              <th className="num">Agree</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => (
              <tr key={r.url}>
                <td className="url">{r.url}</td>
                <td>
                  <span className={`chip ${r.verdict}`}>
                    {r.verdict === "malicious" ? "Malicious" : "Benign"}
                  </span>
                </td>
                <td className="num">{r.score.toFixed(3)}</td>
                <td>
                  <VoteDots models={r.models} />
                </td>
                <td className="num">{Math.round(r.agreement * 5)}/5</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </article>
  );
}

function Method() {
  const steps = [
    ["Input", "raw URL string"],
    ["Semantic", "DistilBERT fine-tuned on URL tokens"],
    ["Structural", "34 lexical features + CLS embedding into four boosters"],
    ["Weighting", "each vote damped by its confidence"],
    ["Decision", "logistic meta-learner, threshold 0.5"],
  ];
  return (
    <section className="method">
      <h3>How a verdict is reached</h3>
      <ol>
        {steps.map(([t, d], i) => (
          <li key={t}>
            <span className="step-n">{String(i + 1).padStart(2, "0")}</span>
            <span className="step-t">{t}</span>
            <span className="step-d">{d}</span>
          </li>
        ))}
      </ol>
    </section>
  );
}

/* ------------------------------------------------------------------ app */

export default function App() {
  const [tab, setTab] = useState("single");
  const [url, setUrl] = useState("");
  const [batch, setBatch] = useState("");
  const [health, setHealth] = useState(null);
  const [single, setSingle] = useState(null);
  const [rows, setRows] = useState(null);
  const [error, setError] = useState(null);
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    fetch(`${API}/api/health`)
      .then((r) => r.json().then((d) => ({ ok: r.ok, d })))
      .then(({ ok, d }) => setHealth(ok ? d : { status: "unavailable", ...d }))
      .catch(() => setHealth({ status: "unreachable" }));
  }, []);

  const ready = health?.status === "ready";
  const demo = health?.mode === "demo";

  async function runSingle(target) {
    const value = (target ?? url).trim();
    if (!value) return;
    setBusy(true);
    setError(null);
    try {
      const d = await post([value]);
      setSingle({ result: d.results[0], elapsed: d.elapsed_ms });
    } catch (e) {
      setError(e.message);
      setSingle(null);
    } finally {
      setBusy(false);
    }
  }

  async function runBatch() {
    const urls = batch.split("\n").map((s) => s.trim()).filter(Boolean);
    if (!urls.length) return;
    setBusy(true);
    setError(null);
    try {
      const d = await post(urls.slice(0, 50));
      setRows({ results: d.results, elapsed: d.elapsed_ms });
    } catch (e) {
      setError(e.message);
      setRows(null);
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="shell">
      <header className="masthead">
        <div>
          <h1>URL Inspector</h1>
          <p className="tagline">
            Malicious URL detection with a hybrid ensemble of DistilBERT and gradient boosting
          </p>
        </div>
        <p className={`status ${ready ? (demo ? "demo" : "ready") : health ? "down" : ""}`}>
          <span className="dot" />
          {!health && "Connecting"}
          {ready && demo && "Demo mode"}
          {ready && !demo && (
            <>
              Trained ensemble
              {health.training_rows ? ` · ${health.training_rows.toLocaleString()} URLs` : ""}
            </>
          )}
          {health && !ready && "Backend offline"}
        </p>
      </header>

      <nav className="tabs" role="tablist">
        {[
          ["single", "Single link"],
          ["batch", "Batch"],
        ].map(([k, label]) => (
          <button
            key={k}
            role="tab"
            aria-selected={tab === k}
            className={tab === k ? "on" : ""}
            onClick={() => setTab(k)}
          >
            {label}
          </button>
        ))}
      </nav>

      {tab === "single" && (
        <section className="input-block">
          <label htmlFor="url">Paste a link to check before opening it</label>
          <div className="field">
            <input
              id="url"
              value={url}
              onChange={(e) => setUrl(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && runSingle()}
              placeholder="https://"
              spellCheck="false"
              autoComplete="off"
            />
            <button onClick={() => runSingle()} disabled={busy || !ready || !url.trim()}>
              {busy ? "Checking…" : "Check link"}
            </button>
          </div>
          <div className="samples">
            <span>Examples</span>
            {SAMPLES.map((s) => (
              <button
                key={s}
                onClick={() => {
                  setUrl(s);
                  runSingle(s);
                }}
                disabled={!ready}
              >
                {s.replace(/^https?:\/\//, "").slice(0, 38)}
                {s.replace(/^https?:\/\//, "").length > 38 ? "…" : ""}
              </button>
            ))}
          </div>
        </section>
      )}

      {tab === "batch" && (
        <section className="input-block">
          <div className="label-row">
            <label htmlFor="batch">One link per line, up to 50</label>
            <button className="link" onClick={() => setBatch(BATCH_EXAMPLE)}>
              Load example set
            </button>
          </div>
          <textarea
            id="batch"
            rows={7}
            value={batch}
            onChange={(e) => setBatch(e.target.value)}
            placeholder={"https://example.com\nhttp://…"}
            spellCheck="false"
          />
          <div className="batch-actions">
            <button className="primary" onClick={runBatch} disabled={busy || !ready || !batch.trim()}>
              {busy ? "Checking…" : "Check all"}
            </button>
          </div>
        </section>
      )}

      {demo && (
        <p className="notice">
          {HOSTED ? (
            <>
              Online demo: scores come from fixed rules over the 34 lexical features and
              run entirely in your browser, so no link you enter leaves this page. The
              trained ensemble runs on the project's backend.
            </>
          ) : (
            <>
              Demo mode: scores come from fixed rules over the lexical features, not trained
              models. Train with <code>python3 -m src.train</code> and restart to load the real
              ensemble.
            </>
          )}
        </p>
      )}
      {health && !ready && (
        <p className="notice error">
          Cannot reach the backend. Start it with <code>run.bat</code> on Windows or{" "}
          <code>./run.sh</code> on a Mac, then reload this page.
        </p>
      )}
      {error && <p className="notice error">{error}</p>}

      {tab === "single" && single && (
        <Result result={single.result} elapsed={single.elapsed} demo={demo} />
      )}
      {tab === "batch" && rows && <BatchTable rows={rows.results} elapsed={rows.elapsed} />}

      <Method />

      <footer className="foot">
        Reads the link text only and never opens the page, so a malicious site behind a
        clean-looking URL can still pass.
      </footer>
    </div>
  );
}
