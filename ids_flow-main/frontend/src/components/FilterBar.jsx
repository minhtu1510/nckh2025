import { Box, Chip, Stack, TextField } from "@mui/material";

export function rangeToSince(range) {
  const now = Date.now() / 1000;
  if (range === "5m") return now - 5 * 60;
  if (range === "15m") return now - 15 * 60;
  if (range === "1h") return now - 60 * 60;
  if (range === "24h") return now - 24 * 60 * 60;
  return null; // "all"
}

export default function FilterBar({
  verdict,
  setVerdict,
  range,
  setRange,
  srcIp,
  setSrcIp,
  dstIp,
  setDstIp,
  q,
  setQ,
}) {
  return (
    <Box
      sx={{
        display: "flex",
        gap: 1,
        flexWrap: "wrap",
        alignItems: "center",
        mt: 1,
      }}
    >
      <Stack direction="row" spacing={1} sx={{ flexWrap: "wrap" }}>
        {["", "benign", "suspicious", "attack"].map((v) => (
          <Chip
            key={v || "all"}
            label={v || "all"}
            variant={verdict === v ? "filled" : "outlined"}
            onClick={() => setVerdict(v)}
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

      <TextField
        size="small"
        label="src_ip"
        value={srcIp}
        onChange={(e) => setSrcIp(e.target.value)}
        sx={{ width: 170 }}
        placeholder="172.31.64.85"
      />
      <TextField
        size="small"
        label="dst_ip"
        value={dstIp}
        onChange={(e) => setDstIp(e.target.value)}
        sx={{ width: 170 }}
        placeholder="172.31.0.2"
      />

      <TextField
        size="small"
        label="search (q)"
        value={q}
        onChange={(e) => setQ(e.target.value)}
        sx={{ width: 320 }}
        placeholder='vd: "172.31.0.2:53" / "UDP" / "DDoS"'
      />
    </Box>
  );
}
