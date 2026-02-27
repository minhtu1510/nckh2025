import { useMemo, useState } from "react";
import Dashboard from "./pages/Dashboard";
import Alerts from "./pages/Alerts";
import {
  AppBar,
  Box,
  Container,
  Tab,
  Tabs,
  Toolbar,
  Typography,
} from "@mui/material";
import ShieldIcon from "@mui/icons-material/Security";

export default function App() {
  const [tab, setTab] = useState(0);

  const page = useMemo(() => {
    if (tab === 0) return <Dashboard />;
    return <Alerts />;
  }, [tab]);

  return (
    <Box sx={{ minHeight: "100vh" }}>
      <AppBar
        position="sticky"
        elevation={0}
        sx={{ borderBottom: "1px solid rgba(255,255,255,0.08)" }}
      >
        <Toolbar>
          <ShieldIcon sx={{ mr: 1 }} />
          <Typography variant="h6" sx={{ fontWeight: 800, mr: 3 }}>
            IDS Web
          </Typography>

          <Tabs
            value={tab}
            onChange={(_, v) => setTab(v)}
            textColor="inherit"
            indicatorColor="primary"
          >
            <Tab label="Dashboard" />
            <Tab label="Alerts" />
          </Tabs>
        </Toolbar>
      </AppBar>

      <Container maxWidth="lg" sx={{ py: 3 }}>
        {page}
      </Container>
    </Box>
  );
}
