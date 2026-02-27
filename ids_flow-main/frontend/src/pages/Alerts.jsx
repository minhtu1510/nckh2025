import { useEffect, useMemo, useState } from "react";
import { getAttacks } from "../api";
import {
  Box,
  Card,
  CardContent,
  Chip,
  Stack,
  Typography,
  Tabs,
  Tab,
} from "@mui/material";
import AlertsTable from "../components/AlertsTable";
import AlertDetailDialog from "../components/AlertDetailDialog";

function deriveSource(r) {
  const hasRule = !!r?.rule_info?.rule;
  const hasML = Number.isFinite(Number(r?.p_attack));
  if (hasRule && hasML) return "both";
  if (hasRule) return "rule";
  if (hasML) return "ml";
  return "none";
}

function normalizeAttack(r) {
  if (!r || typeof r !== "object") return {};

  // meta có thể đang nằm ở r.meta, hoặc backend trả flat fields
  const meta = r.meta ?? {
    src_ip: r.src_ip ?? r.IPV4_SRC_ADDR,
    dst_ip: r.dst_ip ?? r.IPV4_DST_ADDR,
    src_port: r.src_port ?? r.L4_SRC_PORT,
    dst_port: r.dst_port ?? r.L4_DST_PORT,
    proto: r.proto ?? r.PROTOCOL,
    l7_proto: r.l7_proto ?? r.L7_PROTO,
  };

  const stats = r.stats ?? {
    in_bytes: r.in_bytes ?? r.IN_BYTES,
    out_bytes: r.out_bytes ?? r.OUT_BYTES,
    in_pkts: r.in_pkts ?? r.IN_PKTS,
    out_pkts: r.out_pkts ?? r.OUT_PKTS,
    duration_ms:
      r.duration_ms ?? r.FLOW_DURATION_MILLISECONDS ?? r.FLOW_DURATION ?? null,
  };

  const out = {
    ts: r.ts ?? null,
    verdict: r.verdict ?? "unknown",
    stage: r.stage ?? "binary",
    p_attack: r.p_attack ?? null,
    family: r.family ?? null,
    family_conf: r.family_conf ?? null,
    gt_attack: r.gt_attack ?? null,
    gt_label: r.gt_label ?? null,
    rule_info: r.rule_info ?? null,

    meta,
    stats,
  };

  out.source = deriveSource(out); // rule | ml | both | none
  return out;
}

export default function Alerts() {
  const [rows, setRows] = useState([]);
  const [selected, setSelected] = useState(null);

  // tab: all | rule | ml | both
  const [tab, setTab] = useState("all");

  async function refresh() {
    const data = await getAttacks(800);
    const arr = Array.isArray(data) ? data : [];
    const norm = arr.map(normalizeAttack);
    setRows(norm);
  }

  useEffect(() => {
    refresh();
    const t = setInterval(refresh, 1500);
    return () => clearInterval(t);
  }, []);

  const filtered = useMemo(() => {
    if (tab === "all") return rows;
    return rows.filter((r) => r?.source === tab);
  }, [rows, tab]);

  const summary = useMemo(() => {
    let attack = 0,
      suspicious = 0,
      benign = 0,
      ruleOnly = 0,
      mlOnly = 0,
      both = 0;

    for (const r of rows) {
      if (r?.verdict === "attack") attack++;
      else if (r?.verdict === "suspicious") suspicious++;
      else benign++;

      if (r?.source === "rule") ruleOnly++;
      if (r?.source === "ml") mlOnly++;
      if (r?.source === "both") both++;
    }

    return {
      attack,
      suspicious,
      benign,
      ruleOnly,
      mlOnly,
      both,
      total: rows.length,
    };
  }, [rows]);

  return (
    <Box>
      <Card sx={{ mb: 2 }}>
        <CardContent>
          <Typography variant="h6" sx={{ fontWeight: 800 }}>
            Live Alerts
          </Typography>

          <Stack direction="row" spacing={1} sx={{ mt: 1, flexWrap: "wrap" }}>
            <Chip label={`total: ${summary.total}`} />
            <Chip color="error" label={`attack: ${summary.attack}`} />
            <Chip color="warning" label={`suspicious: ${summary.suspicious}`} />
            <Chip label={`benign: ${summary.benign}`} />
            <Chip label={`rule-only: ${summary.ruleOnly}`} />
            <Chip label={`ml-only: ${summary.mlOnly}`} />
            <Chip label={`both: ${summary.both}`} />
          </Stack>

          <Tabs
            value={tab}
            onChange={(_, v) => setTab(v)}
            sx={{ mt: 1 }}
            variant="scrollable"
            scrollButtons="auto"
          >
            <Tab value="all" label="All" />
            <Tab value="rule" label="Rule-only" />
            <Tab value="ml" label="ML-only" />
            <Tab value="both" label="Both" />
          </Tabs>

          <Typography variant="caption" sx={{ opacity: 0.7 }}>
            Click 1 row để xem detail kiểu “Wireshark” (Flow / Rule / ML).
          </Typography>
        </CardContent>
      </Card>

      <AlertsTable rows={filtered} onSelect={setSelected} />
      <AlertDetailDialog item={selected} onClose={() => setSelected(null)} />
    </Box>
  );
}
