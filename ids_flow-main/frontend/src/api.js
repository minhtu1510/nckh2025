// import axios from "axios";

// export async function getStats() {
//   const r = await axios.get("/api/stats");
//   return r.data;
// }
// export async function getAttacks(limit = 300) {
//   const r = await axios.get(`/api/attacks?limit=${limit}`);
//   return r.data;
// }
// export async function getConfig() {
//   const r = await axios.get("/api/config");
//   return r.data;
// }
// export async function setConfig(cfg) {
//   const r = await axios.post("/api/config", cfg);
//   return r.data;
// }

// export async function getFlows(limit = 1000) {
//   const r = await axios.get(`/api/flows?limit=${limit}`);
//   return r.data;
// }

import axios from "axios";

export async function getStats() {
  const r = await axios.get("/api/stats");
  return r.data;
}
export async function getConfig() {
  const r = await axios.get("/api/config");
  return r.data;
}
export async function setConfig(cfg) {
  const r = await axios.post("/api/config", cfg);
  return r.data;
}

export async function getFlows(limit = 1000, params = {}) {
  const sp = new URLSearchParams({ limit: String(limit) });
  for (const [k, v] of Object.entries(params)) {
    if (v === undefined || v === null || v === "") continue;
    sp.set(k, String(v));
  }
  const r = await axios.get(`/api/flows?${sp.toString()}`);
  return r.data;
}

export async function getAttacks(limit = 300, params = {}) {
  const sp = new URLSearchParams({ limit: String(limit) });
  for (const [k, v] of Object.entries(params)) {
    if (v === undefined || v === null || v === "") continue;
    sp.set(k, String(v));
  }
  const r = await axios.get(`/api/attacks?${sp.toString()}`);
  return r.data;
}
