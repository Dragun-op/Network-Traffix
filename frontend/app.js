const { useState, useMemo, useEffect } = React;

/* =====================================================================
   API LAYER
   Talks to the FastAPI backend (backend/main.py). If it can't be reached
   — not running yet, wrong port, CORS misconfigured — this quietly falls
   back to generated mock data so the UI never goes blank while you're
   wiring things up. Watch the console for a warning when that happens.
   ===================================================================== */

const API_BASE = "http://127.0.0.1:8000";

const ATTACK_TYPES = [
  "DDoS",
  "DoS",
  "Port Scan",
  "Brute Force",
  "Botnet",
  "Web Attack",
  "Infiltration",
];
const SEVERITIES = ["low", "medium", "high", "critical"];
const STATUSES = ["new", "investigating", "resolved"];
const FEATURE_POOL = [
  "Flow Duration",
  "Fwd Packet Length Mean",
  "Bwd Packets/s",
  "SYN Flag Count",
  "Packet Length Std",
  "Flow IAT Mean",
  "Total Fwd Packets",
  "Down/Up Ratio",
  "Active Mean",
  "Fwd PSH Flags",
];

function randInt(a, b) {
  return Math.floor(Math.random() * (b - a + 1)) + a;
}
function randIp() {
  return `${randInt(10, 203)}.${randInt(0, 255)}.${randInt(0, 255)}.${randInt(1, 254)}`;
}
function pick(arr) {
  return arr[randInt(0, arr.length - 1)];
}

function genExplanation() {
  const feats = [...FEATURE_POOL].sort(() => Math.random() - 0.5).slice(0, 3);
  let remaining = 100;
  return feats
    .map((f, i) => {
      const val =
        i === feats.length - 1
          ? remaining
          : randInt(10, Math.floor(remaining / 1.5));
      remaining -= val;
      return { feature: f, contribution: val };
    })
    .sort((a, b) => b.contribution - a.contribution);
}

function genMockIncidents(n) {
  const now = Date.now();
  return Array.from({ length: n })
    .map((_, i) => {
      const severity = pick(SEVERITIES);
      const conf =
        severity === "critical"
          ? randInt(88, 99)
          : severity === "high"
            ? randInt(72, 92)
            : severity === "medium"
              ? randInt(45, 80)
              : randInt(20, 60);
      return {
        id: `INC-${(1000 + i).toString()}`,
        timestamp: new Date(
          now - randInt(0, 60 * 60 * 36) * 1000,
        ).toISOString(),
        src_ip: randIp(),
        dst_ip: randIp(),
        protocol: pick(["TCP", "UDP", "ICMP"]),
        attack_type: pick(ATTACK_TYPES),
        severity,
        confidence: conf,
        status: pick(STATUSES),
        packet_count: randInt(12, 48000),
        explanation: genExplanation(),
      };
    })
    .sort((a, b) => new Date(b.timestamp) - new Date(a.timestamp));
}

const MOCK_INCIDENTS = genMockIncidents(42);

let usingMockData = false;

async function fetchIncidents() {
  try {
    const res = await fetch(`${API_BASE}/api/incidents?limit=500`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    return data.items;
  } catch (err) {
    console.warn(
      "Backend unreachable, falling back to mock incidents:",
      err.message,
    );
    usingMockData = true;
    await new Promise((r) => setTimeout(r, 200));
    return MOCK_INCIDENTS;
  }
}

async function fetchSummary() {
  try {
    const res = await fetch(`${API_BASE}/api/summary`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    return await res.json();
  } catch (err) {
    usingMockData = true;
    const bySeverity = Object.fromEntries(
      SEVERITIES.map((s) => [
        s,
        MOCK_INCIDENTS.filter((i) => i.severity === s).length,
      ]),
    );
    return { total: MOCK_INCIDENTS.length, ...bySeverity };
  }
}

// The list endpoint omits `explanation` to stay light; fetch it on demand
// when a row is expanded.
async function fetchIncidentDetail(id) {
  try {
    const res = await fetch(`${API_BASE}/api/incidents/${id}`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    return await res.json();
  } catch (err) {
    const mock = MOCK_INCIDENTS.find((i) => i.id === id);
    return mock || null;
  }
}

/* ===================================================================== */

function timeAgo(iso) {
  const diffMs = Date.now() - new Date(iso).getTime();
  const mins = Math.floor(diffMs / 60000);
  if (mins < 1) return "just now";
  if (mins < 60) return `${mins}m ago`;
  const hrs = Math.floor(mins / 60);
  if (hrs < 24) return `${hrs}h ago`;
  return `${Math.floor(hrs / 24)}d ago`;
}

function SeverityBadge({ level }) {
  return <span className={`sev-badge ${level}`}>{level}</span>;
}

function FilterBar({
  activeSeverities,
  toggleSeverity,
  search,
  setSearch,
  attackType,
  setAttackType,
  onClear,
}) {
  return (
    <div className="filter-bar">
      {SEVERITIES.map((sev) => (
        <button
          key={sev}
          className={`chip sev-${sev} ${activeSeverities.includes(sev) ? "active" : ""}`}
          onClick={() => toggleSeverity(sev)}
        >
          <span
            className="chip-dot"
            style={{ background: `var(--sev-${sev})` }}
          ></span>
          {sev}
        </button>
      ))}
      <input
        className="search-input"
        placeholder="Search by IP address…"
        value={search}
        onChange={(e) => setSearch(e.target.value)}
      />
      <select
        className="select-input"
        value={attackType}
        onChange={(e) => setAttackType(e.target.value)}
      >
        <option value="all">All attack types</option>
        {ATTACK_TYPES.map((t) => (
          <option key={t} value={t}>
            {t}
          </option>
        ))}
      </select>
      {(activeSeverities.length > 0 || search || attackType !== "all") && (
        <button className="clear-btn" onClick={onClear}>
          Clear filters
        </button>
      )}
    </div>
  );
}

function ExplainRow({ explanation }) {
  return (
    <tr className="explain-row">
      <td colSpan={8}>
        <div className="explain-title">
          Why this was flagged — top contributing features
        </div>
        {!explanation && (
          <div style={{ fontSize: "13px", color: "var(--color-muted)" }}>
            Loading explanation…
          </div>
        )}
        {explanation &&
          explanation.map((f) => (
            <div className="feature-bar-row" key={f.feature}>
              <div className="feature-name">{f.feature}</div>
              <div className="feature-track">
                <div
                  className="feature-fill"
                  style={{ width: `${f.contribution}%` }}
                ></div>
              </div>
              <div className="feature-pct">{f.contribution}%</div>
            </div>
          ))}
      </td>
    </tr>
  );
}

function IncidentsTable({ incidents, expandedId, onExpand, explanations }) {
  if (incidents.length === 0) {
    return (
      <div className="table-card">
        <div className="empty-state">
          No incidents match these filters. Try widening your search.
        </div>
      </div>
    );
  }
  return (
    <div className="table-card">
      <table>
        <thead>
          <tr>
            <th>Time</th>
            <th>Source IP</th>
            <th>Destination IP</th>
            <th>Protocol</th>
            <th>Attack type</th>
            <th>Risk</th>
            <th>Confidence</th>
            <th>Status</th>
          </tr>
        </thead>
        <tbody>
          {incidents.map((inc) => (
            <React.Fragment key={inc.id}>
              <tr onClick={() => onExpand(inc.id)}>
                <td>{timeAgo(inc.timestamp)}</td>
                <td className="mono">{inc.src_ip}</td>
                <td className="mono">{inc.dst_ip}</td>
                <td>{inc.protocol}</td>
                <td>{inc.attack_type}</td>
                <td>
                  <SeverityBadge level={inc.severity} />
                </td>
                <td>{inc.confidence}%</td>
                <td>
                  <span className="status-pill">{inc.status}</span>
                </td>
              </tr>
              {expandedId === inc.id && (
                <ExplainRow explanation={explanations[inc.id]} />
              )}
            </React.Fragment>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function App() {
  const [incidents, setIncidents] = useState([]);
  const [summary, setSummary] = useState(null);
  const [loading, setLoading] = useState(true);
  const [activeSeverities, setActiveSeverities] = useState([]);
  const [search, setSearch] = useState("");
  const [attackType, setAttackType] = useState("all");
  const [expandedId, setExpandedId] = useState(null);
  const [explanations, setExplanations] = useState({});

  useEffect(() => {
    fetchIncidents().then((data) => {
      setIncidents(data);
      setLoading(false);
    });
    fetchSummary().then(setSummary);
  }, []);

  function handleExpand(id) {
    if (expandedId === id) {
      setExpandedId(null);
      return;
    }
    setExpandedId(id);
    if (!explanations[id]) {
      fetchIncidentDetail(id).then((detail) => {
        if (detail && detail.explanation) {
          setExplanations((prev) => ({
            ...prev,
            [id]: detail.explanation,
          }));
        }
      });
    }
  }

  function toggleSeverity(sev) {
    setActiveSeverities((prev) =>
      prev.includes(sev) ? prev.filter((s) => s !== sev) : [...prev, sev],
    );
  }
  function clearFilters() {
    setActiveSeverities([]);
    setSearch("");
    setAttackType("all");
  }

  const filtered = useMemo(() => {
    return incidents.filter((inc) => {
      if (activeSeverities.length && !activeSeverities.includes(inc.severity))
        return false;
      if (attackType !== "all" && inc.attack_type !== attackType) return false;
      if (search) {
        const q = search.toLowerCase();
        if (!inc.src_ip.includes(q) && !inc.dst_ip.includes(q)) return false;
      }
      return true;
    });
  }, [incidents, activeSeverities, search, attackType]);

  return (
    <div className="app-shell">
      <div className="chrome">
        <div className="chrome-left">
          <div className="avatar">NT</div>
          <div className="handle-block">
            <span className="handle">Network-Traffix</span>
            <span className="subhandle">@security-ops · analyst console</span>
          </div>
        </div>
        <div className="chrome-right">
          <button className="live-pill">
            <span className="live-dot"></span>Live
          </button>
          <div className="toolbar-dots">
            <span></span>
            <span></span>
            <span></span>
          </div>
        </div>
      </div>

      <div className="hero">
        <h1>
          ALL HIGH-RISK TRAFFIC
          <br />
          leaves behind a <span className="accent">pattern</span>
          <br />
          before it becomes a breach.
        </h1>
        <p>
          Explainable, real-time detection across your network — every alert
          comes with the features that triggered it, so analysts trust the call,
          not just the score.
        </p>
      </div>

      <div className="annotation">
        <span className="corner tl"></span>
        <span className="corner tr"></span>
        <span className="corner bl"></span>
        <span className="corner br"></span>
        <div className="annotation-label">Analyst note</div>
        <p>
          Most of today's critical flags trace back to a SYN-flood pattern from
          a small cluster of source subnets — worth a firewall rule review.
        </p>
      </div>

      {summary && (
        <div className="stats-row">
          <div className="stat-card">
            <span className="num">{summary.total}</span>
            <span className="label">Total incidents</span>
          </div>
          <div className="stat-card">
            <span className="num">{summary.low}</span>
            <span className="label">Low</span>
          </div>
          <div className="stat-card">
            <span className="num">{summary.medium}</span>
            <span className="label">Medium</span>
          </div>
          <div className="stat-card">
            <span className="num">{summary.high}</span>
            <span className="label">High</span>
          </div>
          <div className="stat-card is-critical">
            <span className="num">{summary.critical}</span>
            <span className="label">Critical</span>
          </div>
        </div>
      )}

      <FilterBar
        activeSeverities={activeSeverities}
        toggleSeverity={toggleSeverity}
        search={search}
        setSearch={setSearch}
        attackType={attackType}
        setAttackType={setAttackType}
        onClear={clearFilters}
      />

      {loading ? (
        <div className="table-card">
          <div className="empty-state">Loading incidents…</div>
        </div>
      ) : (
        <IncidentsTable
          incidents={filtered}
          expandedId={expandedId}
          onExpand={handleExpand}
          explanations={explanations}
        />
      )}

      <div className="footnote">
        Showing {filtered.length} of {incidents.length} incidents
        {usingMockData
          ? " · backend unreachable, showing mock data"
          : " · live from FastAPI"}
      </div>
    </div>
  );
}

ReactDOM.createRoot(document.getElementById("root")).render(<App />);
