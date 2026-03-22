import { useEffect, useRef, useState } from "react";
import { getAttacks, getConfig, getFlows, getStats, setConfig, uploadCsv, uploadCsvChunked, resetStore } from "../api";

import {
  Box,
  Button,
  Card,
  CardContent,
  Divider,
  FormControlLabel,
  Grid,
  Switch,
  TextField,
  Typography,
  Chip,
  Stack,
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableRow,
  Tooltip,
  TableContainer,
  Alert,
  LinearProgress,
} from "@mui/material";

import CloudUploadIcon from "@mui/icons-material/CloudUpload";
import DeleteSweepIcon from "@mui/icons-material/DeleteSweep";

import StatCards from "../components/StatCards";

import {
  AlertsPerMinute,
  StageSeverityBreakdown,
  SuspiciousDistribution,
  TopFamiliesChart,
  TopSuspiciousTable,
  TopTargetsChart,
  TopSourcesChart,
  ProtocolDistribution,
} from "../components/Charts";

/* -------- Upload CSV Card -------- */

// File nhỏ hơn ngưỡng này: dùng upload cũ (stream batch).  Lớn hơn: dùng chunked.
const LARGE_FILE_THRESHOLD_MB = 10;
const CHUNK_SIZE_MB = 4;  // kích thước mỗi chunk gửi lên

function UploadCSVCard({ onUploaded }) {
  const [file, setFile] = useState(null);
  const [maxRows, setMaxRows] = useState(0);        // 0 = không giới hạn
  const [batchSize, setBatchSize] = useState(500);    // số dòng / batch
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [dragging, setDragging] = useState(false);
  const [progress, setProgress] = useState(0);        // 0-100
  const [progressLabel, setProgressLabel] = useState("");
  const inputRef = useRef();
  const abortRef = useRef(null);   // AbortController để hủy upload

  const handleFile = (f) => {
    if (!f) return;
    setFile(f);
    setResult(null);
    setError(null);
    setProgress(0);
    setProgressLabel("");
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setDragging(false);
    const f = e.dataTransfer.files?.[0];
    if (f) handleFile(f);
  };

  const handleUpload = async () => {
    if (!file) return;
    setLoading(true);
    setResult(null);
    setError(null);
    setProgress(0);

    const fileMB = file.size / (1024 * 1024);
    const useChunked = fileMB > LARGE_FILE_THRESHOLD_MB;

    if (useChunked) {
      // ── Chunked upload cho file lớn ─────────────────────────────
      const ctrl = new AbortController();
      abortRef.current = ctrl;
      try {
        const res = await uploadCsvChunked(file, {
          chunkSizeMB: CHUNK_SIZE_MB,
          maxRows: maxRows,
          batchSize: batchSize,
          useAdaptiveDede: true,
          signal: ctrl.signal,
          onProgress: (pct, totalIngested, chunkIdx, totalChunks) => {
            setProgress(pct);
            setProgressLabel(
              `Chunk ${chunkIdx + 1}/${totalChunks} — Đã phân tích ${totalIngested.toLocaleString()} flows (${pct}%)`
            );
          },
        });
        setResult(res);
        if (onUploaded) onUploaded();
      } catch (e) {
        if (e?.name === "AbortError") {
          setError("⚠️ Upload đã bị hủy.");
        } else {
          setError(e?.response?.data?.error || e?.message || "Upload failed");
        }
      } finally {
        abortRef.current = null;
        setLoading(false);
      }
    } else {
      // ── Legacy upload cho file nhỏ ──────────────────────────────
      setProgressLabel("Đang xử lý...");
      try {
        const res = await uploadCsv(file, maxRows || 5000, batchSize);
        setProgress(100);
        setResult(res);
        if (onUploaded) onUploaded();
      } catch (e) {
        setError(e?.response?.data?.error || e?.message || "Upload failed");
      } finally {
        setLoading(false);
      }
    }
  };

  const handleCancel = () => {
    abortRef.current?.abort();
  };

  const handleReset = async () => {
    if (!window.confirm("Xóa toàn bộ dữ liệu dashboard?")) return;
    try {
      await resetStore();
      setResult(null);
      setFile(null);
      setProgress(0);
      setProgressLabel("");
      if (onUploaded) onUploaded();
    } catch (e) {
      setError(e?.message || "Reset failed");
    }
  };

  const fileMB = file ? (file.size / (1024 * 1024)) : 0;
  const willUseChunked = fileMB > LARGE_FILE_THRESHOLD_MB;

  return (
    <Card
      sx={{
        background: "linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%)",
        border: "1px solid rgba(99,179,237,0.3)",
        borderRadius: 3,
        mb: 2,
      }}
    >
      <CardContent>
        <Stack direction="row" alignItems="center" justifyContent="space-between" sx={{ mb: 2 }}>
          <Stack direction="row" alignItems="center" spacing={1}>
            <CloudUploadIcon sx={{ color: "#63b3ed" }} />
            <Typography variant="h6" sx={{ fontWeight: 900, color: "#e2e8f0" }}>
              Upload CICIDS Dataset (CSV)
            </Typography>
          </Stack>
          <Tooltip title="Reset / Xóa toàn bộ dữ liệu dashboard">
            <Button
              startIcon={<DeleteSweepIcon />}
              variant="outlined"
              color="error"
              size="small"
              onClick={handleReset}
              sx={{ borderRadius: 2 }}
            >
              Reset Data
            </Button>
          </Tooltip>
        </Stack>

        <Stack direction={{ xs: "column", sm: "row" }} spacing={2} alignItems="flex-start">
          {/* Drop zone */}
          <Box
            onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
            onDragLeave={() => setDragging(false)}
            onDrop={handleDrop}
            onClick={() => !loading && inputRef.current?.click()}
            sx={{
              flex: 1,
              minHeight: 100,
              border: `2px dashed ${dragging ? "#63b3ed" : "rgba(99,179,237,0.4)"}`,
              borderRadius: 2,
              display: "flex",
              flexDirection: "column",
              alignItems: "center",
              justifyContent: "center",
              cursor: loading ? "not-allowed" : "pointer",
              bgcolor: dragging ? "rgba(99,179,237,0.1)" : "rgba(255,255,255,0.03)",
              transition: "all 0.2s",
              px: 2, py: 2,
              "&:hover": !loading ? { bgcolor: "rgba(99,179,237,0.08)", borderColor: "#63b3ed" } : {},
            }}
          >
            <input
              ref={inputRef}
              type="file"
              accept=".csv"
              style={{ display: "none" }}
              onChange={(e) => handleFile(e.target.files?.[0])}
            />
            <CloudUploadIcon sx={{ fontSize: 36, color: "rgba(99,179,237,0.7)", mb: 1 }} />
            <Typography variant="body2" sx={{ color: "#94a3b8", textAlign: "center" }}>
              {file ? (
                <span style={{ color: "#63b3ed", fontWeight: 700 }}>📄 {file.name}</span>
              ) : (
                "Kéo thả hoặc click để chọn file CSV (CICIDS2017/2018)"
              )}
            </Typography>
            {file && (
              <Typography variant="caption" sx={{ color: "#64748b", mt: 0.5 }}>
                {fileMB >= 1024
                  ? `${(fileMB / 1024).toFixed(2)} GB`
                  : fileMB >= 1
                    ? `${fileMB.toFixed(1)} MB`
                    : `${(file.size / 1024).toFixed(1)} KB`}
                {willUseChunked && (
                  <span style={{ color: "#f6ad55", marginLeft: 8 }}>
                    ⚡ Chunked upload ({CHUNK_SIZE_MB}MB/chunk)
                  </span>
                )}
              </Typography>
            )}
          </Box>

          {/* Controls */}
          <Stack spacing={1.5} sx={{ minWidth: 210 }}>
            <TextField
              size="small"
              label="Max rows (0=tất cả)"
              type="number"
              value={maxRows}
              onChange={(e) => setMaxRows(Number(e.target.value) || 0)}
              inputProps={{ min: 0, max: 500000, step: 1000 }}
              sx={{
                "& .MuiInputBase-root": { bgcolor: "rgba(255,255,255,0.06)", color: "#e2e8f0" },
                "& .MuiInputLabel-root": { color: "#94a3b8" },
                "& .MuiOutlinedInput-notchedOutline": { borderColor: "rgba(99,179,237,0.3)" },
              }}
            />
            <TextField
              size="small"
              label="Batch size (RAM)"
              type="number"
              value={batchSize}
              onChange={(e) => setBatchSize(Math.max(50, Number(e.target.value) || 500))}
              inputProps={{ min: 50, max: 2000, step: 50 }}
              sx={{
                "& .MuiInputBase-root": { bgcolor: "rgba(255,255,255,0.06)", color: "#e2e8f0" },
                "& .MuiInputLabel-root": { color: "#94a3b8" },
                "& .MuiOutlinedInput-notchedOutline": { borderColor: "rgba(99,179,237,0.3)" },
              }}
            />
            <Button
              variant="contained"
              onClick={handleUpload}
              disabled={!file || loading}
              startIcon={<CloudUploadIcon />}
              sx={{
                background: "linear-gradient(135deg, #667eea, #764ba2)",
                color: "white",
                fontWeight: 700,
                borderRadius: 2,
                "&:disabled": { opacity: 0.5 },
                "&:hover": { background: "linear-gradient(135deg, #764ba2, #667eea)" },
              }}
            >
              {loading ? "Đang xử lý..." : "Upload & Phân tích"}
            </Button>
            {loading && (
              <Button
                variant="outlined"
                color="error"
                size="small"
                onClick={handleCancel}
                sx={{ borderRadius: 2 }}
              >
                ❌ Hủy upload
              </Button>
            )}
          </Stack>
        </Stack>

        {loading && (
          <Box sx={{ mt: 2 }}>
            <LinearProgress
              variant={willUseChunked ? "determinate" : "indeterminate"}
              value={progress}
              sx={{ borderRadius: 2, height: 8 }}
            />
            <Typography variant="caption" sx={{ color: "#94a3b8", mt: 0.5, display: "block" }}>
              {progressLabel || "Đang chạy Exp9 inference..."}
            </Typography>
          </Box>
        )}

        {result && (
          <Alert
            severity="success"
            sx={{ mt: 2, bgcolor: "rgba(72,187,120,0.1)", color: "#68d391", border: "1px solid rgba(72,187,120,0.3)" }}
          >
            ✅ Phân tích xong <strong>{result.ingested?.toLocaleString()}</strong> flows
            {result.total_csv_rows ? ` / ${result.total_csv_rows.toLocaleString()} dòng CSV` : ""}
            &nbsp;|&nbsp;Features mapped: <strong>{result.mapped_features}</strong>/76
            &nbsp;| Missing: <strong>{result.missing_features}</strong>
            {result.dede_mode && <>&nbsp;| Mode: <strong>{result.dede_mode}</strong></>}
          </Alert>
        )}

        {error && (
          <Alert severity="error" sx={{ mt: 2 }}>
            {error}
          </Alert>
        )}

        <Typography variant="caption" sx={{ color: "#475569", mt: 1.5, display: "block" }}>
          💡 File &lt;{LARGE_FILE_THRESHOLD_MB}MB: upload thường. File &gt;{LARGE_FILE_THRESHOLD_MB}MB: tự động chia chunk {CHUNK_SIZE_MB}MB/chunk, RAM backend thấp.
          Batch size nhỏ hơn = ít RAM hơn (nhưng chậm hơn). Giảm xuống 100–200 nếu máy yếu.
        </Typography>
      </CardContent>
    </Card>
  );
}

/* -------- helpers -------- */

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

function rangeToSince(range) {
  const now = Date.now() / 1000;
  if (range === "5m") return now - 5 * 60;
  if (range === "15m") return now - 15 * 60;
  if (range === "1h") return now - 60 * 60;
  if (range === "24h") return now - 24 * 60 * 60;
  return null; // "all"
}

/* -------- table -------- */

function FlowsTable({
  rows,
  height = 760,
  paused,
  onPaused,
  limit,
  onLimit,
  // flow filters:
  srcIp,
  setSrcIp,
  dstIp,
  setDstIp,
  qText,
  setQText,
}) {
  return (
    <Card sx={{ height }}>
      <CardContent
        sx={{ height: "100%", display: "flex", flexDirection: "column" }}
      >
        <Stack
          direction="row"
          alignItems="center"
          justifyContent="space-between"
          sx={{ mb: 1 }}
        >
          <Typography variant="h6" sx={{ fontWeight: 900 }}>
            Flows (filtered)
          </Typography>

          <Stack direction="row" spacing={1} alignItems="center">
            <TextField
              size="small"
              label="limit"
              type="number"
              inputProps={{ min: 50, max: 5000, step: 50 }}
              value={limit}
              onChange={(e) => onLimit(Number(e.target.value || 1000))}
              sx={{ width: 110 }}
            />
            <FormControlLabel
              control={
                <Switch
                  checked={paused}
                  onChange={(e) => onPaused(e.target.checked)}
                />
              }
              label="Pause"
            />
          </Stack>
        </Stack>

        {/* ✅ flow filters placed here (less confusing) */}
        <Stack direction="row" spacing={1} sx={{ flexWrap: "wrap", mb: 1.5 }}>
          <TextField
            size="small"
            label="src_ip (contains)"
            value={srcIp}
            onChange={(e) => setSrcIp(e.target.value)}
            sx={{ width: 180 }}
            placeholder="vd: 172.31."
          />
          <TextField
            size="small"
            label="dst_ip (contains)"
            value={dstIp}
            onChange={(e) => setDstIp(e.target.value)}
            sx={{ width: 180 }}
            placeholder="vd: 172.31."
          />
          <TextField
            size="small"
            label="search (q)"
            value={qText}
            onChange={(e) => setQText(e.target.value)}
            sx={{ width: 340 }}
            placeholder="vd: udp / :53 / ddos / 172.31.0.2:53"
          />
        </Stack>

        <Divider sx={{ mb: 1.5 }} />

        <TableContainer sx={{ maxHeight: "100%" }}>
          <Table stickyHeader size="small">
            <TableHead>
              <TableRow>
                <TableCell sx={{ fontWeight: 900, whiteSpace: "nowrap" }}>
                  Time
                </TableCell>
                <TableCell sx={{ fontWeight: 900 }}>Proto</TableCell>
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
                <TableCell sx={{ fontWeight: 900 }}>Verdict</TableCell>
                <TableCell sx={{ fontWeight: 900 }}>Stage</TableCell>
                <TableCell sx={{ fontWeight: 900 }} align="right">
                  p_attack
                </TableCell>
                <TableCell sx={{ fontWeight: 900 }}>Rule</TableCell>
                <TableCell sx={{ fontWeight: 900 }}>Family</TableCell>
              </TableRow>
            </TableHead>

            <TableBody>
              {(rows || []).map((r, idx) => {
                const m = r?.meta || {};
                const v = String(r?.verdict ?? "unknown");
                const stage = String(r?.stage ?? "—");
                const p = safeNum(r?.p_attack, null);

                const src = fmtAddr(m.src_ip, m.src_port);
                const dst = fmtAddr(m.dst_ip, m.dst_port);

                const pkts = safeNum(m.pkts, null);
                const bytes = safeNum(m.bytes, null);
                const dur = safeNum(m.dur_ms, null);

                const fam = r?.family ?? "—";
                const famConf = safeNum(r?.family_conf, null);

                const ruleName =
                  r?.rule_name ??
                  r?.rule?.name ??
                  (typeof r?.rule_info === "string"
                    ? r.rule_info
                    : (r?.rule_info?.rule ?? r?.rule_info?.name));

                return (
                  <TableRow key={idx} hover>
                    <TableCell sx={{ whiteSpace: "nowrap" }}>
                      {fmtTs(r?.ts)}
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

                    <TableCell>
                      <Chip
                        size="small"
                        label={v}
                        color={verdictColor(v)}
                        variant="outlined"
                      />
                    </TableCell>

                    <TableCell>{stage}</TableCell>
                    <TableCell align="right">
                      {p == null ? "—" : p.toFixed(6)}
                    </TableCell>

                    <TableCell>
                      {ruleName ? (
                        <Tooltip
                          title={
                            r?.rule_info
                              ? typeof r.rule_info === "string"
                                ? r.rule_info
                                : JSON.stringify(r.rule_info, null, 2)
                              : ruleName
                          }
                        >
                          <Chip
                            size="small"
                            variant="outlined"
                            label={ruleName}
                          />
                        </Tooltip>
                      ) : (
                        "—"
                      )}
                    </TableCell>

                    <TableCell>
                      <Stack direction="row" spacing={1} alignItems="center">
                        <span>{fam}</span>
                        {famConf != null && (
                          <Chip
                            size="small"
                            variant="outlined"
                            label={(famConf * 100).toFixed(1) + "%"}
                          />
                        )}
                      </Stack>
                    </TableCell>
                  </TableRow>
                );
              })}

              {(!rows || rows.length === 0) && (
                <TableRow>
                  <TableCell
                    colSpan={12}
                    sx={{ py: 4, textAlign: "center", color: "text.secondary" }}
                  >
                    No flows match current filters.
                  </TableCell>
                </TableRow>
              )}
            </TableBody>
          </Table>
        </TableContainer>

        <Typography variant="caption" sx={{ mt: 1, color: "text.secondary" }}>
          Backend keeps up to max_flows (deque). UI fetches “limit” newest rows.
        </Typography>
      </CardContent>
    </Card>
  );
}

/* -------- page -------- */

export default function Dashboard() {
  const [stats, setStats] = useState(null);
  const [cfg, setCfg] = useState(null);

  const [flows, setFlows] = useState([]);
  const [attacks, setAttacks] = useState([]);

  const [paused, setPaused] = useState(false);
  const [flowLimit, setFlowLimit] = useState(1000);

  // global filters (less confusing)
  const [verdictF, setVerdictF] = useState(""); // ""=all
  const [range, setRange] = useState("15m");

  // flow filters (placed inside flow table card)
  const [srcIp, setSrcIp] = useState("");
  const [dstIp, setDstIp] = useState("");
  const [qText, setQText] = useState("");

  async function refresh() {
    const since = rangeToSince(range);

    const params = {
      verdict: verdictF || "",
      since,
      src_ip: srcIp.trim() || "",
      dst_ip: dstIp.trim() || "",
      q: qText.trim() || "",
    };

    const [s, c, a, f] = await Promise.all([
      getStats(),
      getConfig(),
      // alerts feed (events): mostly suspicious/attack/rule
      getAttacks(5000, params),
      // flows feed (all)
      getFlows(flowLimit, params),
    ]);

    setStats(s);
    setCfg(c);
    setAttacks(Array.isArray(a) ? a : []);
    setFlows(Array.isArray(f) ? f : []);
  }

  useEffect(() => {
    refresh();
    const t = setInterval(() => {
      if (!paused) refresh();
    }, 1200);
    return () => clearInterval(t);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [paused, flowLimit, verdictF, range, srcIp, dstIp, qText]);

  if (!stats || !cfg) return <Typography>Loading...</Typography>;

  const counts = stats.counts || {};
  const last = stats.last_ts
    ? new Date(stats.last_ts * 1000).toLocaleString()
    : "N/A";

  return (
    <Box sx={{ pb: 3 }}>
      <UploadCSVCard onUploaded={refresh} />
      <StatCards counts={counts} queueLen={stats.queue_len} last={last} />


      {/* ✅ global filters only (verdict + time) */}
      <Box
        sx={{
          mt: 1,
          display: "flex",
          gap: 1,
          flexWrap: "wrap",
          alignItems: "center",
        }}
      >
        <Stack direction="row" spacing={1} sx={{ flexWrap: "wrap" }}>
          {["", "benign", "suspicious", "attack"].map((v) => (
            <Chip
              key={v || "all"}
              label={v || "all"}
              variant={verdictF === v ? "filled" : "outlined"}
              onClick={() => setVerdictF(v)}
            />
          ))}
        </Stack>

        <TextField
          size="small"
          label="time"
          select
          SelectProps={{ native: true }}
          value={range}
          onChange={(e) => setRange(e.target.value)}
          sx={{ width: 130 }}
        >
          <option value="15m">last 15m</option>
          <option value="5m">last 5m</option>
          <option value="1h">last 1h</option>
          <option value="24h">last 24h</option>
          <option value="all">all</option>
        </TextField>
      </Box>

      {/* Wazuh-like layout */}
      <Grid container spacing={2} sx={{ mt: 1 }} alignItems="stretch">
        {/* Row 1: Timeline big + Verdict donut */}
        <Grid item xs={12} lg={8}>
          <Card sx={{ height: 360 }}>
            <CardContent>
              <Typography variant="h6" sx={{ fontWeight: 900, mb: 1 }}>
                Alerts timeline (per minute)
              </Typography>
              <Box sx={{ height: 290 }}>
                {/* showBenign=false => avoid confusion (events mostly not benign) */}
                <AlertsPerMinute
                  rows={attacks}
                  height={290}
                  showBenign={false}
                />
              </Box>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} lg={4}>
          <Card sx={{ height: 360 }}>
            <CardContent>
              <Typography variant="h6" sx={{ fontWeight: 900, mb: 1 }}>
                Verdict distribution
              </Typography>
              <Box sx={{ height: 290 }}>
                <SuspiciousDistribution rows={attacks} height={290} />
              </Box>
            </CardContent>
          </Card>
        </Grid>

        {/* Row 2: Top talkers side-by-side (k=8, tick fixed) */}
        <Grid item xs={12} lg={6}>
          <Card sx={{ height: 360 }}>
            <CardContent>
              <Typography variant="h6" sx={{ fontWeight: 900, mb: 1 }}>
                Top targets (dst_ip:port)
              </Typography>
              <Box sx={{ height: 290 }}>
                <TopTargetsChart rows={attacks} k={8} height={290} />
              </Box>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} lg={6}>
          <Card sx={{ height: 360 }}>
            <CardContent>
              <Typography variant="h6" sx={{ fontWeight: 900, mb: 1 }}>
                Top sources (src_ip)
              </Typography>
              <Box sx={{ height: 290 }}>
                <TopSourcesChart rows={attacks} k={8} height={290} />
              </Box>
            </CardContent>
          </Card>
        </Grid>

        {/* Row 3: Stage + Protocol + Families */}
        <Grid item xs={12} md={4}>
          <Card sx={{ height: 340 }}>
            <CardContent>
              <Typography variant="h6" sx={{ fontWeight: 900, mb: 1 }}>
                Stage breakdown
              </Typography>
              <Box sx={{ height: 260 }}>
                {/* ✅ use flows to show binary correctly */}
                <StageSeverityBreakdown rows={flows} height={260} />
              </Box>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} md={4}>
          <Card sx={{ height: 340 }}>
            <CardContent>
              <Typography variant="h6" sx={{ fontWeight: 900, mb: 1 }}>
                Protocol distribution
              </Typography>
              <Box sx={{ height: 260 }}>
                <ProtocolDistribution rows={attacks} height={260} />
              </Box>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} md={4}>
          <Card sx={{ height: 340 }}>
            <CardContent>
              <Typography variant="h6" sx={{ fontWeight: 900, mb: 1 }}>
                Top families (attack)
              </Typography>
              <Box sx={{ height: 260 }}>
                <TopFamiliesChart rows={attacks} k={8} height={260} />
              </Box>
            </CardContent>
          </Card>
        </Grid>

        {/* Row 4: Recent alerts */}
        <Grid item xs={12}>
          <TopSuspiciousTable rows={attacks} limit={15} />
        </Grid>

        {/* Row 5: Flows table + flow filters inside */}
        <Grid item xs={12}>
          <FlowsTable
            rows={flows}
            height={760}
            paused={paused}
            onPaused={setPaused}
            limit={flowLimit}
            onLimit={setFlowLimit}
            srcIp={srcIp}
            setSrcIp={setSrcIp}
            dstIp={dstIp}
            setDstIp={setDstIp}
            qText={qText}
            setQText={setQText}
          />
        </Grid>

        {/* Runtime Config */}
        <Grid item xs={12}>
          <Card>
            <CardContent>
              <Typography variant="h6" sx={{ fontWeight: 900 }}>
                Runtime Config
              </Typography>
              <Divider sx={{ my: 1.5 }} />
              <Grid container spacing={2} alignItems="center">
                <Grid item>
                  <TextField
                    size="small"
                    label="tau_low"
                    type="number"
                    inputProps={{ step: 0.01 }}
                    value={cfg.tau_low}
                    onChange={(e) =>
                      setCfg({ ...cfg, tau_low: Number(e.target.value) })
                    }
                  />
                </Grid>
                <Grid item>
                  <TextField
                    size="small"
                    label="tau_high"
                    type="number"
                    inputProps={{ step: 0.01 }}
                    value={cfg.tau_high}
                    onChange={(e) =>
                      setCfg({ ...cfg, tau_high: Number(e.target.value) })
                    }
                  />
                </Grid>
                <Grid item>
                  <FormControlLabel
                    control={
                      <Switch
                        checked={!!cfg.enable_rules}
                        onChange={(e) =>
                          setCfg({ ...cfg, enable_rules: e.target.checked })
                        }
                      />
                    }
                    label="enable_rules"
                  />
                </Grid>
                <Grid item>
                  <Button
                    variant="contained"
                    onClick={async () => setCfg(await setConfig(cfg))}
                  >
                    Save
                  </Button>
                </Grid>
                <Grid item>
                  <Button variant="outlined" onClick={refresh}>
                    Refresh
                  </Button>
                </Grid>
              </Grid>
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Box>
  );
}
