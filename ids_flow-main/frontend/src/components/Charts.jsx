import React, { useMemo } from "react";
import {
  ResponsiveContainer,
  AreaChart,
  Area,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
  BarChart,
  Bar,
  PieChart,
  Pie,
  Cell,
  Legend,
} from "recharts";
import {
  Box,
  Card,
  CardContent,
  Typography,
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableRow,
  Chip,
  Stack,
  Divider,
} from "@mui/material";

/* ---------------- utils ---------------- */

function safeNum(x, d = null) {
  const n = Number(x);
  return Number.isFinite(n) ? n : d;
}

function fmtTs(ts) {
  const n = safeNum(ts, 0);
  if (!n) return "—";
  return new Date(n * 1000).toLocaleString();
}

function protoName(p) {
  const n = safeNum(p, null);
  if (n === 6) return "TCP";
  if (n === 17) return "UDP";
  if (n === 1) return "ICMP";
  return n == null ? "—" : String(n);
}

function fmtAddr(ip, port) {
  if (!ip) return "—";
  if (port == null) return String(ip);
  return `${ip}:${port}`;
}

function verdictColor(v) {
  if (v === "attack") return "error";
  if (v === "suspicious") return "warning";
  if (v === "benign") return "success";
  return "default";
}

function pickDecision(r) {
  if (!r) return "—";
  if (r.stage === "rule" && r.rule_info) {
    const nm = r.rule_info.rule ?? r.rule_info.name ?? "rule";
    const sev = r.rule_info.severity ? `(${r.rule_info.severity})` : "";
    return `${nm} ${sev}`.trim();
  }
  if (typeof r.p_attack === "number") return `ML p=${r.p_attack.toFixed(4)}`;
  if (r.family) return `Family: ${r.family}`;
  return "—";
}

function trunc(s, n = 18) {
  const t = String(s ?? "");
  return t.length > n ? t.slice(0, n) + "…" : t;
}

/* ---------------- colors (SIEM) ---------------- */

const C_ATTACK = "#d32f2f";
const C_SUSP = "#ed6c02";
const C_BENIGN = "#2e7d32";

const F_ATTACK = "rgba(211,47,47,0.25)";
const F_SUSP = "rgba(237,108,2,0.25)";
const F_BENIGN = "rgba(46,125,50,0.20)";

const C_STAGE = "#1976d2"; // blue
const C_PROTO = "#6a1b9a"; // purple
const C_TOP = "#455a64"; // blue-grey

/* -------------- aggregation -------------- */

function minuteKey(tsSec) {
  const t = safeNum(tsSec, 0);
  if (!t) return null;
  const d = new Date(t * 1000);
  d.setSeconds(0, 0);
  return Math.floor(d.getTime() / 1000);
}

function bucketAlertsPerMinute(rows) {
  const m = new Map();
  for (const r of rows || []) {
    const k = minuteKey(r?.ts);
    if (!k) continue;
    const o = m.get(k) || { ts: k, benign: 0, suspicious: 0, attack: 0 };
    const v = String(r?.verdict ?? "unknown").toLowerCase();
    if (v === "benign") o.benign += 1;
    else if (v === "suspicious") o.suspicious += 1;
    else if (v === "attack") o.attack += 1;
    m.set(k, o);
  }
  return [...m.values()].sort((a, b) => a.ts - b.ts).slice(-60);
}

function countByKey(rows, keyFn, filterFn = null, limit = 10) {
  const c = new Map();
  for (const r of rows || []) {
    if (filterFn && !filterFn(r)) continue;
    const k = keyFn(r);
    if (!k) continue;
    c.set(k, (c.get(k) || 0) + 1);
  }
  const arr = [...c.entries()].map(([k, v]) => ({ key: k, count: v }));
  arr.sort((a, b) => b.count - a.count);
  return arr.slice(0, limit);
}

/* ---------------- charts ---------------- */

export function AlertsPerMinute({ rows, height = 260, showBenign = false }) {
  const data = useMemo(() => bucketAlertsPerMinute(rows), [rows]);

  const tick = (t) => {
    const n = safeNum(t, 0);
    if (!n) return "";
    const d = new Date(n * 1000);
    return `${String(d.getHours()).padStart(2, "0")}:${String(d.getMinutes()).padStart(2, "0")}`;
  };

  if (!data.length)
    return (
      <Typography variant="body2" color="text.secondary">
        No data
      </Typography>
    );

  return (
    <ResponsiveContainer width="100%" height={height}>
      <AreaChart
        data={data}
        margin={{ top: 10, right: 10, bottom: 0, left: 0 }}
      >
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis dataKey="ts" tickFormatter={tick} />
        <YAxis allowDecimals={false} />
        <Tooltip labelFormatter={(l) => fmtTs(l)} />
        <Legend />
        <Area
          type="monotone"
          dataKey="attack"
          stackId="1"
          name="attack"
          stroke={C_ATTACK}
          fill={F_ATTACK}
        />
        <Area
          type="monotone"
          dataKey="suspicious"
          stackId="1"
          name="suspicious"
          stroke={C_SUSP}
          fill={F_SUSP}
        />
        {showBenign ? (
          <Area
            type="monotone"
            dataKey="benign"
            stackId="1"
            name="benign"
            stroke={C_BENIGN}
            fill={F_BENIGN}
          />
        ) : null}
      </AreaChart>
    </ResponsiveContainer>
  );
}

export function StageSeverityBreakdown({ rows, height = 260 }) {
  const data = useMemo(() => {
    let rule = 0,
      binary = 0,
      family = 0,
      dede = 0,
      ganopt = 0,
      standard = 0;
    let ruleHigh = 0,
      ruleMed = 0;

    for (const r of rows || []) {
      const st = String(r?.stage ?? "");
      if (st === "rule") {
        rule += 1;
        const sev = String(r?.rule_info?.severity ?? "").toLowerCase();
        if (sev === "high") ruleHigh += 1;
        else if (sev === "medium") ruleMed += 1;
      } else if (st === "binary") binary += 1;
      else if (st === "family") family += 1;
      else if (st === "dede_blocked") dede += 1;
      else if (st === "ganopt_stacking") ganopt += 1;
      else if (st === "standard_stacking") standard += 1;
    }

    const res = [
      { stage: "rule", count: rule, high: ruleHigh, medium: ruleMed },
      { stage: "dede_blocked", count: dede, high: 0, medium: 0 },
      { stage: "ganopt_stacking", count: ganopt, high: 0, medium: 0 },
      { stage: "standard_stacking", count: standard, high: 0, medium: 0 },
    ];
    if (binary > 0) res.push({ stage: "binary", count: binary, high: 0, medium: 0 });
    if (family > 0) res.push({ stage: "family", count: family, high: 0, medium: 0 });
    return res;
  }, [rows]);

  return (
    <ResponsiveContainer width="100%" height={height}>
      <BarChart data={data} margin={{ top: 10, right: 10, bottom: 0, left: 0 }}>
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis dataKey="stage" />
        <YAxis allowDecimals={false} />
        <Tooltip
          formatter={(v, name, p) => {
            if (name === "count" && p?.payload?.stage === "rule") {
              const hi = p.payload.high ?? 0;
              const me = p.payload.medium ?? 0;
              return [`${v} (high:${hi}, medium:${me})`, "count"];
            }
            return [v, name];
          }}
        />
        <Bar dataKey="count" name="count" fill={C_STAGE} />
      </BarChart>
    </ResponsiveContainer>
  );
}

export function SuspiciousDistribution({ rows, height = 260 }) {
  const data = useMemo(() => {
    let benign = 0,
      suspicious = 0,
      attack = 0;
    for (const r of rows || []) {
      const v = String(r?.verdict ?? "").toLowerCase();
      if (v === "benign") benign += 1;
      else if (v === "suspicious") suspicious += 1;
      else if (v === "attack") attack += 1;
    }
    const raw = [
      { name: "benign", value: benign, color: C_BENIGN },
      { name: "suspicious", value: suspicious, color: C_SUSP },
      { name: "attack", value: attack, color: C_ATTACK },
    ];
    // ✅ hide benign (or any slice) if value=0 to avoid confusing legend
    return raw.filter((x) => x.value > 0);
  }, [rows]);

  const total = data.reduce((s, x) => s + x.value, 0);

  if (!data.length)
    return (
      <Typography variant="body2" color="text.secondary">
        No data
      </Typography>
    );

  return (
    <ResponsiveContainer width="100%" height={height}>
      <PieChart>
        <Tooltip />
        <Legend />
        <Pie
          data={data}
          dataKey="value"
          nameKey="name"
          innerRadius="55%"
          outerRadius="80%"
          label={(p) => `${p.name}: ${p.value}`}
        >
          {data.map((x, i) => (
            <Cell key={i} fill={x.color} />
          ))}
        </Pie>
        <text
          x="50%"
          y="50%"
          textAnchor="middle"
          dominantBaseline="middle"
          style={{ fontSize: 12 }}
        >
          {total}
        </text>
      </PieChart>
    </ResponsiveContainer>
  );
}

export function TopFamiliesChart({ rows, k = 8, height = 260 }) {
  const data = useMemo(() => {
    const top = countByKey(
      rows,
      (r) => r?.family,
      (r) => String(r?.verdict ?? "") === "attack" && !!r?.family,
      k,
    );
    return top.map((x) => ({ family: x.key, count: x.count }));
  }, [rows, k]);

  if (!data.length)
    return (
      <Typography variant="body2" color="text.secondary">
        No family data
      </Typography>
    );

  return (
    <ResponsiveContainer width="100%" height={height}>
      <BarChart data={data} margin={{ top: 10, right: 10, bottom: 0, left: 0 }}>
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis
          dataKey="family"
          interval={0}
          angle={-20}
          textAnchor="end"
          height={60}
        />
        <YAxis allowDecimals={false} />
        <Tooltip />
        <Bar dataKey="count" name="count" fill={C_ATTACK} />
      </BarChart>
    </ResponsiveContainer>
  );
}

export function ProtocolDistribution({ rows, height = 260 }) {
  const data = useMemo(() => {
    const top = countByKey(
      rows,
      (r) => protoName(r?.meta?.proto),
      (r) => {
        const v = String(r?.verdict ?? "").toLowerCase();
        return v === "attack" || v === "suspicious";
      },
      10,
    );
    return top.map((x) => ({ proto: x.key, count: x.count }));
  }, [rows]);

  if (!data.length)
    return (
      <Typography variant="body2" color="text.secondary">
        No data
      </Typography>
    );

  return (
    <ResponsiveContainer width="100%" height={height}>
      <BarChart data={data} margin={{ top: 10, right: 10, bottom: 0, left: 0 }}>
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis dataKey="proto" />
        <YAxis allowDecimals={false} />
        <Tooltip />
        <Bar dataKey="count" name="count" fill={C_PROTO} />
      </BarChart>
    </ResponsiveContainer>
  );
}

export function TopTargetsChart({ rows, k = 8, height = 260 }) {
  const data = useMemo(() => {
    const top = countByKey(
      rows,
      (r) => {
        const m = r?.meta || {};
        const dip = m.dst_ip;
        const dp = m.dst_port;
        if (!dip) return null;
        return dp == null ? `${dip}` : `${dip}:${dp}`;
      },
      (r) => {
        const v = String(r?.verdict ?? "").toLowerCase();
        return v === "attack" || v === "suspicious";
      },
      k,
    );
    return top.map((x) => ({ target: x.key, count: x.count }));
  }, [rows, k]);

  if (!data.length)
    return (
      <Typography variant="body2" color="text.secondary">
        No targets
      </Typography>
    );

  return (
    <ResponsiveContainer width="100%" height={height}>
      <BarChart
        data={data}
        margin={{ top: 10, right: 10, bottom: 10, left: 0 }}
        barCategoryGap="25%"
      >
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis
          dataKey="target"
          interval={0}
          angle={-35}
          textAnchor="end"
          height={90}
          tickFormatter={(v) => trunc(v, 18)}
        />
        <YAxis allowDecimals={false} />
        <Tooltip />
        <Bar dataKey="count" name="count" fill={C_TOP} />
      </BarChart>
    </ResponsiveContainer>
  );
}

export function TopSourcesChart({ rows, k = 8, height = 260 }) {
  const data = useMemo(() => {
    const top = countByKey(
      rows,
      (r) => r?.meta?.src_ip,
      (r) => {
        const v = String(r?.verdict ?? "").toLowerCase();
        return v === "attack" || v === "suspicious";
      },
      k,
    );
    return top.map((x) => ({ src: x.key, count: x.count }));
  }, [rows, k]);

  if (!data.length)
    return (
      <Typography variant="body2" color="text.secondary">
        No sources
      </Typography>
    );

  return (
    <ResponsiveContainer width="100%" height={height}>
      <BarChart
        data={data}
        margin={{ top: 10, right: 10, bottom: 10, left: 0 }}
        barCategoryGap="25%"
      >
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis
          dataKey="src"
          interval={0}
          angle={-35}
          textAnchor="end"
          height={90}
          tickFormatter={(v) => trunc(v, 18)}
        />
        <YAxis allowDecimals={false} />
        <Tooltip />
        <Bar dataKey="count" name="count" fill={C_TOP} />
      </BarChart>
    </ResponsiveContainer>
  );
}

/* ---------------- table ---------------- */

export function TopSuspiciousTable({ rows, limit = 15 }) {
  const data = useMemo(() => {
    const arr = Array.isArray(rows) ? rows.slice() : [];
    arr.sort((a, b) => safeNum(b?.ts, 0) - safeNum(a?.ts, 0));
    return arr.slice(0, limit);
  }, [rows, limit]);

  return (
    <Card>
      <CardContent>
        <Stack
          direction="row"
          alignItems="center"
          justifyContent="space-between"
          sx={{ mb: 1 }}
        >
          <Typography variant="h6" sx={{ fontWeight: 900 }}>
            Recent alerts (drill-down)
          </Typography>
          <Typography variant="body2" sx={{ opacity: 0.7 }}>
            newest first
          </Typography>
        </Stack>
        <Divider sx={{ mb: 1.5 }} />

        <Box sx={{ overflow: "auto" }}>
          <Table size="small" stickyHeader>
            <TableHead>
              <TableRow>
                <TableCell sx={{ fontWeight: 900, width: 170 }}>Time</TableCell>
                <TableCell sx={{ fontWeight: 900, width: 110 }}>
                  Verdict
                </TableCell>
                <TableCell sx={{ fontWeight: 900, width: 80 }}>Proto</TableCell>
                <TableCell sx={{ fontWeight: 900 }}>Source</TableCell>
                <TableCell sx={{ fontWeight: 900 }}>Destination</TableCell>
                <TableCell sx={{ fontWeight: 900 }} align="right">
                  Pkts
                </TableCell>
                <TableCell sx={{ fontWeight: 900 }} align="right">
                  Bytes
                </TableCell>
                <TableCell sx={{ fontWeight: 900 }} align="right">
                  Dur(ms)
                </TableCell>
                <TableCell sx={{ fontWeight: 900, width: 90 }}>Stage</TableCell>
                <TableCell sx={{ fontWeight: 900, width: 160 }}>
                  Decision
                </TableCell>
              </TableRow>
            </TableHead>

            <TableBody>
              {data.map((r, idx) => {
                const m = r?.meta || {};
                const v = String(r?.verdict ?? "unknown");
                const src = fmtAddr(m.src_ip, m.src_port);
                const dst = fmtAddr(m.dst_ip, m.dst_port);
                const pkts = safeNum(m.pkts, null);
                const bytes = safeNum(m.bytes, null);
                const dur = safeNum(m.dur_ms, null);

                return (
                  <TableRow key={idx} hover>
                    <TableCell sx={{ whiteSpace: "nowrap" }}>
                      {fmtTs(r?.ts)}
                    </TableCell>
                    <TableCell>
                      <Chip
                        size="small"
                        label={v}
                        color={verdictColor(v)}
                        variant="outlined"
                      />
                    </TableCell>
                    <TableCell>{protoName(m.proto)}</TableCell>
                    <TableCell sx={{ fontFamily: "monospace" }}>
                      {src}
                    </TableCell>
                    <TableCell sx={{ fontFamily: "monospace" }}>
                      {dst}
                    </TableCell>
                    <TableCell align="right">
                      {pkts == null ? "—" : pkts.toFixed(0)}
                    </TableCell>
                    <TableCell align="right">
                      {bytes == null ? "—" : bytes.toFixed(0)}
                    </TableCell>
                    <TableCell align="right">
                      {dur == null ? "—" : dur.toFixed(0)}
                    </TableCell>
                    <TableCell>{r?.stage ?? "—"}</TableCell>
                    <TableCell>{pickDecision(r)}</TableCell>
                  </TableRow>
                );
              })}

              {data.length === 0 && (
                <TableRow>
                  <TableCell
                    colSpan={10}
                    sx={{ py: 3, textAlign: "center", color: "text.secondary" }}
                  >
                    No alerts yet (try replay CSV or generate traffic).
                  </TableCell>
                </TableRow>
              )}
            </TableBody>
          </Table>
        </Box>
      </CardContent>
    </Card>
  );
}
