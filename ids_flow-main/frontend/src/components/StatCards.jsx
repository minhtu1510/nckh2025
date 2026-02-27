import { Card, CardContent, Grid, Typography } from "@mui/material";

function Stat({ title, value, tone }) {
  return (
    <Card sx={{ border: "1px solid rgba(255,255,255,0.06)" }}>
      <CardContent>
        <Typography variant="body2" sx={{ opacity: 0.75 }}>
          {title}
        </Typography>
        <Typography
          variant="h4"
          sx={{ fontWeight: 900, mt: 0.5, color: tone || "inherit" }}
        >
          {value}
        </Typography>
      </CardContent>
    </Card>
  );
}

export default function StatCards({ counts, queueLen, last }) {
  const benign = counts.benign || 0;
  const suspicious = counts.suspicious || 0;
  const attack = counts.attack || 0;
  const total = benign + suspicious + attack;

  return (
    <Grid container spacing={2}>
      <Grid item xs={12} sm={6} md={3}>
        <Stat title="Total seen" value={total} />
      </Grid>
      <Grid item xs={12} sm={6} md={3}>
        <Stat title="Benign" value={benign} tone="#4ade80" />
      </Grid>
      <Grid item xs={12} sm={6} md={3}>
        <Stat title="Suspicious" value={suspicious} tone="#fbbf24" />
      </Grid>
      <Grid item xs={12} sm={6} md={3}>
        <Stat title="Attack" value={attack} tone="#fb7185" />
      </Grid>

      <Grid item xs={12} sm={6} md={3}>
        <Stat title="Queue len" value={queueLen} />
      </Grid>
      <Grid item xs={12} sm={6} md={9}>
        <Stat title="Last event" value={last} />
      </Grid>
    </Grid>
  );
}
