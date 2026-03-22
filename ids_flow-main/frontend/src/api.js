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

/**
 * Upload 1 file CSV CICIDS lên backend (legacy, stream batch – OK cho file <200MB).
 * @param {File} file       – File object
 * @param {number} maxRows  – Giới hạn dòng (default 5000, 0 = không giới hạn)
 * @param {number} batchSize – Số dòng mỗi batch backend xử lý (default 500)
 */
export async function uploadCsv(file, maxRows = 5000, batchSize = 500) {
  const fd = new FormData();
  fd.append("file", file);
  fd.append("max_rows", String(maxRows));
  fd.append("batch_size", String(batchSize));
  const r = await axios.post("/api/upload_csv", fd, {
    headers: { "Content-Type": "multipart/form-data" },
    // Không set timeout – file lớn cần thời gian dài
    timeout: 0,
  });
  return r.data;
}

/**
 * Chunked CSV upload – dùng cho file lớn (>200MB, thậm chí 2GB).
 * Chia file thành các chunk nhỏ (mặc định 4MB), gửi tuần tự.
 * RAM backend chỉ giữ 1 chunk tại một thời điểm → không tràn RAM.
 *
 * @param {File}   file        – File object từ <input type="file">
 * @param {object} opts
 *   @param {number}   opts.chunkSizeMB    – Kích thước mỗi chunk (MB), default 4
 *   @param {number}   opts.maxRows        – Giới hạn tổng số dòng (0 = không giới hạn)
 *   @param {number}   opts.batchSize      – Batch size xử lý backend, default 500
 *   @param {boolean}  opts.useAdaptiveDede – Bật Adaptive DeDe, default true
 *   @param {Function} opts.onProgress     – Callback(pct, totalIngested, chunkIdx, totalChunks)
 *   @param {AbortSignal} opts.signal      – Để cancel upload
 * @returns {Promise<{ok, ingested, mapped_features, missing_features, dede_mode}>}
 */
export async function uploadCsvChunked(file, opts = {}) {
  const {
    chunkSizeMB = 4,
    maxRows = 0,
    batchSize = 500,
    useAdaptiveDede = false,   // false = bypass DeDe → Standard Stacking, tránh false positive
    onProgress = null,
    signal = null,
  } = opts;

  const CHUNK_BYTES = chunkSizeMB * 1024 * 1024;
  const totalChunks = Math.max(1, Math.ceil(file.size / CHUNK_BYTES));
  // ID duy nhất cho session này
  const uploadId = `${Date.now()}_${Math.random().toString(36).slice(2, 8)}`;

  let lastResp = null;

  for (let i = 0; i < totalChunks; i++) {
    if (signal?.aborted) throw new DOMException("Upload cancelled", "AbortError");

    const start = i * CHUNK_BYTES;
    const end = Math.min(start + CHUNK_BYTES, file.size);
    const blob = file.slice(start, end);
    const isLast = (i === totalChunks - 1);

    const fd = new FormData();
    fd.append("chunk", blob, file.name);
    fd.append("upload_id", uploadId);
    fd.append("chunk_index", String(i));
    fd.append("total_chunks", String(totalChunks));
    fd.append("is_last", isLast ? "1" : "0");
    fd.append("max_rows", String(maxRows));
    fd.append("batch_size", String(batchSize));
    fd.append("use_adaptive_dede", useAdaptiveDede ? "true" : "false");

    const r = await axios.post("/api/upload_csv_chunk", fd, {
      headers: { "Content-Type": "multipart/form-data" },
      timeout: 120_000,   // 2 phút / chunk
      signal,
    });

    lastResp = r.data;
    const pct = Math.round(((i + 1) / totalChunks) * 100);
    onProgress?.(pct, lastResp.total_ingested ?? 0, i, totalChunks);
  }

  return {
    ok: true,
    ingested: lastResp?.total_ingested ?? 0,
    total_csv_rows: lastResp?.total_rows ?? 0,
    mapped_features: lastResp?.mapped_features ?? 0,
    missing_features: lastResp?.missing_features ?? 0,
    dede_mode: lastResp?.dede_mode ?? "bypass",
    adaptive_thr: lastResp?.adaptive_thr ?? {},
  };
}

/**
 * Xóa toàn bộ dữ liệu trong store (reset dashboard).
 */
export async function resetStore() {
  const r = await axios.post("/api/reset");
  return r.data;
}
