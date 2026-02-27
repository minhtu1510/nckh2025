import {
  Dialog,
  DialogContent,
  DialogTitle,
  IconButton,
  Stack,
  Typography,
} from "@mui/material";
import CloseIcon from "@mui/icons-material/Close";

export default function AlertDetailDialog({ item, onClose }) {
  return (
    <Dialog open={!!item} onClose={onClose} maxWidth="md" fullWidth>
      <DialogTitle
        sx={{
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
        }}
      >
        Alert Detail
        <IconButton onClick={onClose}>
          <CloseIcon />
        </IconButton>
      </DialogTitle>
      <DialogContent>
        {item && (
          <Stack spacing={1}>
            <Typography>
              <b>verdict:</b> {item.verdict}
            </Typography>
            <Typography>
              <b>stage:</b> {item.stage}
            </Typography>
            <Typography>
              <b>p_attack:</b> {item.p_attack ?? "—"}
            </Typography>
            <Typography>
              <b>family:</b> {item.family ?? "—"} (conf:{" "}
              {item.family_conf ?? "—"})
            </Typography>
            <Typography>
              <b>rule_info:</b>{" "}
              {item.rule_info ? JSON.stringify(item.rule_info) : "—"}
            </Typography>
            <Typography>
              <b>gt_attack (replay):</b> {item.gt_attack ?? "—"}
            </Typography>

            <Typography sx={{ mt: 1, opacity: 0.75 }}>Raw JSON</Typography>
            <pre
              style={{
                margin: 0,
                padding: 12,
                borderRadius: 12,
                background: "rgba(255,255,255,0.05)",
                overflow: "auto",
              }}
            >
              {JSON.stringify(item, null, 2)}
            </pre>
          </Stack>
        )}
      </DialogContent>
    </Dialog>
  );
}
