import {
  Box,
  Chip,
  Dialog,
  DialogContent,
  DialogTitle,
  Divider,
  Stack,
  Typography,
  Accordion,
  AccordionSummary,
  AccordionDetails,
} from "@mui/material";
import ExpandMoreIcon from "@mui/icons-material/ExpandMore";

function fmtTs(ts) {
  if (ts == null) return "N/A";
  const n = Number(ts);
  if (!Number.isFinite(n) || n <= 0) return "N/A";
  return new Date(n * 1000).toLocaleString();
}

function fmtNum(x) {
  if (x == null) return "—";
  const n = Number(x);
  if (!Number.isFinite(n)) return "—";
  return n.toLocaleString();
}

function fmtP(x, k = 4) {
  if (x == null) return "—";
  const n = Number(x);
  if (!Number.isFinite(n)) return "—";
  return n.toFixed(k);
}

function fmtMs(ms) {
  const n = Number(ms);
  if (!Number.isFinite(n) || n < 0) return "—";
  if (n < 1000) return `${n.toFixed(0)} ms`;
  return `${(n / 1000).toFixed(2)} s`;
}

function protoName(proto) {
  const n = Number(proto);
  if (n === 6) return "TCP";
  if (n === 17) return "UDP";
  if (n === 1) return "ICMP";
  return Number.isFinite(n) ? String(n) : "—";
}

function fmtAddr(ip, port) {
  if (!ip) return "—";
  if (port == null || port === "" || port === "—") return String(ip);
  return `${ip}:${port}`;
}

function verdictColor(v) {
  if (v === "attack") return "error";
  if (v === "suspicious") return "warning";
  if (v === "benign") return "success";
  return "default";
}

function sevColor(sev) {
  if (sev === "high") return "error";
  if (sev === "medium") return "warning";
  return "default";
}

function KeyVals({ rows }) {
  return (
    <Box sx={{ display: "grid", gap: 1 }}>
      {rows.map(([k, v]) => (
        <Box
          key={k}
          sx={{ display: "flex", justifyContent: "space-between", gap: 2 }}
        >
          <Typography variant="body2" sx={{ opacity: 0.7 }}>
            {k}
          </Typography>
          <Typography
            variant="body2"
            sx={{ fontWeight: 800, textAlign: "right" }}
          >
            {String(v)}
          </Typography>
        </Box>
      ))}
    </Box>
  );
}

export default function AlertDetailDialog({ item, onClose }) {
  const open = !!item;
  const verdict = item?.verdict ?? "unknown";
  const stage = item?.stage ?? "—";

  const m = item?.meta || {};
  const rule = item?.rule_info || null;

  const src = fmtAddr(m.src_ip, m.src_port);
  const dst = fmtAddr(m.dst_ip, m.dst_port);
  const proto = protoName(m.proto);

  const pkts = m.pkts ?? "—";
  const bytes = m.bytes ?? "—";
  const dur = m.dur_ms ?? "—";

  const pAttack =
    typeof item?.p_attack === "number" ? fmtP(item.p_attack, 6) : "—";

  const family =
    item?.family != null
      ? `${item.family} (conf: ${
          typeof item?.family_conf === "number"
            ? fmtP(item.family_conf, 3)
            : "—"
        })`
      : "—";

  const ruleName =
    rule?.rule ?? rule?.name ?? item?.rule_name ?? item?.rule?.name ?? null;
  const sev = rule?.severity ?? null;
  const reason = rule?.reason ?? null;

  return (
    <Dialog open={open} onClose={onClose} maxWidth="md" fullWidth>
      <DialogTitle sx={{ fontWeight: 900 }}>
        Alert detail
        <Typography variant="body2" sx={{ opacity: 0.7, mt: 0.5 }}>
          {fmtTs(item?.ts)}
        </Typography>
      </DialogTitle>

      <DialogContent>
        {/* Summary */}
        <Stack direction="row" spacing={1} sx={{ flexWrap: "wrap", mb: 1.5 }}>
          <Chip
            size="small"
            label={verdict}
            color={verdictColor(verdict)}
            variant="outlined"
          />
          <Chip size="small" label={`stage: ${stage}`} variant="outlined" />
          <Chip size="small" label={`proto: ${proto}`} variant="outlined" />
          <Chip size="small" label={`src: ${src}`} variant="outlined" />
          <Chip size="small" label={`dst: ${dst}`} variant="outlined" />
          {ruleName ? (
            <Chip size="small" label={`rule: ${ruleName}`} variant="outlined" />
          ) : null}
          {sev ? (
            <Chip
              size="small"
              label={`sev: ${sev}`}
              color={sevColor(sev)}
              variant="outlined"
            />
          ) : null}
        </Stack>

        <Divider sx={{ mb: 1.5 }} />

        {/* Flow stats */}
        <Typography sx={{ fontWeight: 900, mb: 1 }}>Flow</Typography>
        <KeyVals
          rows={[
            ["pkts", fmtNum(pkts)],
            ["bytes", fmtNum(bytes)],
            ["duration", fmtMs(dur)],
            ["p_attack", pAttack],
            ["family", family],
          ]}
        />

        <Divider sx={{ my: 1.5 }} />

        {/* Decision */}
        <Typography sx={{ fontWeight: 900, mb: 1 }}>Decision</Typography>
        {rule ? (
          <KeyVals
            rows={[
              ["rule", ruleName ?? "—"],
              ["severity", sev ?? "—"],
              ["reason", reason ?? "—"],
            ]}
          />
        ) : (
          <Typography variant="body2" sx={{ opacity: 0.8 }}>
            No rule hit. Decision mainly from ML (binary/family).
          </Typography>
        )}

        <Divider sx={{ my: 1.5 }} />

        {/* GT */}
        <Typography sx={{ fontWeight: 900, mb: 1 }}>
          Ground truth (replay)
        </Typography>
        <KeyVals
          rows={[
            ["gt_attack", item?.gt_attack ?? "—"],
            ["gt_label", item?.gt_label ?? "—"],
          ]}
        />

        {/* Advanced (optional) */}
        <Accordion sx={{ mt: 2 }}>
          <AccordionSummary expandIcon={<ExpandMoreIcon />}>
            <Typography sx={{ fontWeight: 900 }}>Advanced</Typography>
          </AccordionSummary>
          <AccordionDetails>
            <Typography variant="body2" sx={{ opacity: 0.8, mb: 1 }}>
              Raw event (debug)
            </Typography>
            <pre style={{ margin: 0, whiteSpace: "pre-wrap" }}>
              {JSON.stringify(item, null, 2)}
            </pre>
          </AccordionDetails>
        </Accordion>
      </DialogContent>
    </Dialog>
  );
}
