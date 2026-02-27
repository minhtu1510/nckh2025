import { DataGrid } from "@mui/x-data-grid";
import { Box, Chip } from "@mui/material";

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

function n0(x) {
  const n = Number(x);
  return Number.isFinite(n) ? n : null;
}

function verdictColor(v) {
  if (v === "attack") return "error";
  if (v === "suspicious") return "warning";
  if (v === "benign") return "success";
  return "default";
}

function fmtTime(ts) {
  const n = Number(ts);
  if (!Number.isFinite(n) || n <= 0) return "—";
  return new Date(n * 1000).toLocaleString();
}

export default function AlertsTable({ rows, onSelect }) {
  const safeRows = Array.isArray(rows) ? rows.filter(Boolean) : [];

  const columns = [
    {
      field: "ts",
      headerName: "Time",
      width: 170,
      // ✅ valueGetter phải an toàn
      valueGetter: (p) => p?.row?.ts ?? null,
      renderCell: (p) => fmtTime(p?.row?.ts),
      sortable: true,
    },
    {
      field: "verdict",
      headerName: "Verdict",
      width: 120,
      sortable: true,
      renderCell: (p) => {
        const v = p?.row?.verdict ?? "unknown";
        return (
          <Chip
            size="small"
            label={v}
            color={verdictColor(v)}
            variant="outlined"
          />
        );
      },
    },
    {
      field: "proto",
      headerName: "Proto",
      width: 80,
      valueGetter: (p) => p?.row?.meta?.proto ?? null,
      renderCell: (p) => protoName(p?.row?.meta?.proto),
      sortable: false,
    },
    {
      field: "src",
      headerName: "Source",
      flex: 1,
      minWidth: 180,
      sortable: false,
      renderCell: (p) => {
        const m = p?.row?.meta || {};
        return (
          <span style={{ fontFamily: "monospace" }}>
            {fmtAddr(m.src_ip, m.src_port)}
          </span>
        );
      },
    },
    {
      field: "dst",
      headerName: "Destination",
      flex: 1,
      minWidth: 180,
      sortable: false,
      renderCell: (p) => {
        const m = p?.row?.meta || {};
        return (
          <span style={{ fontFamily: "monospace" }}>
            {fmtAddr(m.dst_ip, m.dst_port)}
          </span>
        );
      },
    },
    // {
    //   field: "pkts",
    //   headerName: "Pkts",
    //   width: 90,
    //   align: "right",
    //   headerAlign: "right",
    //   valueGetter: (p) => n0(p?.row?.meta?.pkts),
    //   sortable: false,
    // },
    // {
    //   field: "bytes",
    //   headerName: "Bytes",
    //   width: 110,
    //   align: "right",
    //   headerAlign: "right",
    //   valueGetter: (p) => n0(p?.row?.meta?.bytes),
    //   sortable: false,
    // },
    // {
    //   field: "dur_ms",
    //   headerName: "Dur(ms)",
    //   width: 110,
    //   align: "right",
    //   headerAlign: "right",
    //   valueGetter: (p) => n0(p?.row?.meta?.dur_ms),
    //   sortable: false,
    // },
    { field: "stage", headerName: "Stage", width: 90, sortable: true },

    {
      field: "decision",
      headerName: "Decision",
      flex: 1.2,
      minWidth: 220,
      sortable: false,
      renderCell: (p) => {
        const r = p?.row;
        if (!r) return "—";

        if (r.stage === "rule" && r.rule_info) {
          const sev = r.rule_info.severity ? ` (${r.rule_info.severity})` : "";
          const nm = r.rule_info.rule ?? r.rule_info.name ?? "rule";
          return `${nm}${sev}`;
        }
        if (typeof r.p_attack === "number")
          return `ML p=${r.p_attack.toFixed(4)}`;
        if (r.family) return `Family: ${r.family}`;
        return "—";
      },
    },

    {
      field: "gt",
      headerName: "GT",
      width: 110,
      sortable: false,
      renderCell: (p) => p?.row?.gt_attack ?? "—",
    },
  ];

  return (
    <Box sx={{ height: 640, width: "100%" }}>
      <DataGrid
        rows={safeRows}
        columns={columns}
        // ✅ getRowId phải handle thiếu field
        getRowId={(r) =>
          r?.id ??
          `${r?.ts ?? 0}-${r?.meta?.src_ip ?? "x"}-${r?.meta?.dst_ip ?? "x"}-${
            r?.meta?.src_port ?? "x"
          }-${r?.meta?.dst_port ?? "x"}-${r?.stage ?? "x"}-${r?.verdict ?? "x"}`
        }
        pageSizeOptions={[25, 50, 100]}
        initialState={{
          pagination: { paginationModel: { pageSize: 50, page: 0 } },
        }}
        disableRowSelectionOnClick
        onRowClick={(p) => onSelect?.(p?.row)}
      />
    </Box>
  );
}
