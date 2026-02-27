from collections import deque, Counter
import threading
import time


def _proto_name(p):
    if isinstance(p, str):
        s = p.strip().lower()
        if s in ("tcp", "udp", "icmp"):
            return s
        try:
            p = float(s)
        except:
            return ""

    try:
        n = int(float(p))
    except:
        return ""

    if n == 6:  return "tcp"
    if n == 17: return "udp"
    if n == 1:  return "icmp"
    return str(n)

def _safe_int(x):
    try:
        return int(str(x).strip())
    except:
        return None


def _get_rule_name(x):
    ri = x.get("rule_info")
    if isinstance(ri, dict):
        return str(ri.get("rule") or ri.get("name") or "")
    return ""


def _build_blob(x):
    """
    Unified blob for substring fallback search.
    Includes verdict/stage/family/proto src/dst ip:port + rule info if available.
    """
    m = x.get("meta") or {}
    proto_s = _proto_name(m.get("proto"))
    family = x.get("family") or ""

    rule_name = ""
    rule_sev = ""
    ri = x.get("rule_info")
    if isinstance(ri, dict):
        rule_name = ri.get("rule") or ri.get("name") or ""
        rule_sev = ri.get("severity") or ""

    return (
        f"{x.get('verdict','')} {x.get('stage','')} {family} "
        f"{proto_s} "
        f"{m.get('src_ip','')}:{m.get('src_port','')} "
        f"{m.get('dst_ip','')}:{m.get('dst_port','')} "
        f"{rule_name} {rule_sev}"
    ).lower()


def _compile_query(q: str):
    """
    Parse query string q -> list of predicates. AND semantics.
    Supports:
      - dst_ip=1 (contains '1')
      - dst_ip=172. (starts with 172.)
      - dst_ip=172.* (starts with 172.)
      - src_ip=...
      - udp/tcp/icmp
      - family=ddos (or token 'ddos' fallback to family/blob)
      - rule=dns_ttl (or token 'dns_ttl' -> rule/blob)
      - :53 (port 53 in src/dst)
      - 53 (also treated as port)
    """
    ql = (q or "").strip().lower()
    if not ql:
        return []

    tokens = [t for t in ql.split() if t]
    preds = []

    for t in tokens:
        # key=value
        if "=" in t:
            k, v = t.split("=", 1)
            k = k.strip()
            v = v.strip()
            if not v:
                continue

            # dst_ip=...
            if k in ("dst_ip", "dip", "dst"):
                v2 = v[:-2] if v.endswith(".*") else v
                if v2.endswith(".") or v.endswith(".*"):
                    preds.append(
                        lambda x, v2=v2: str((x.get("meta") or {}).get("dst_ip") or "").startswith(v2)
                    )
                else:
                    preds.append(
                        lambda x, v=v: v in str((x.get("meta") or {}).get("dst_ip") or "")
                    )
                continue

            # src_ip=...
            if k in ("src_ip", "sip", "src"):
                v2 = v[:-2] if v.endswith(".*") else v
                if v2.endswith(".") or v.endswith(".*"):
                    preds.append(
                        lambda x, v2=v2: str((x.get("meta") or {}).get("src_ip") or "").startswith(v2)
                    )
                else:
                    preds.append(
                        lambda x, v=v: v in str((x.get("meta") or {}).get("src_ip") or "")
                    )
                continue

            # family=...
            if k in ("family", "fam"):
                preds.append(lambda x, v=v: v in str(x.get("family") or "").lower())
                continue

            # rule=...
            if k in ("rule", "r"):
                preds.append(lambda x, v=v: v in _get_rule_name(x).lower())
                continue

            # proto=udp / proto=17 ...
            if k in ("proto", "protocol"):
                if v in ("udp", "tcp", "icmp"):
                    preds.append(
                        lambda x, v=v: _proto_name((x.get("meta") or {}).get("proto")).lower() == v
                    )
                else:
                    vn = _safe_int(v)
                    preds.append(
                        lambda x, vn=vn: vn is not None
                        and _safe_int((x.get("meta") or {}).get("proto")) == vn
                    )
                continue

            # port=53
            if k in ("port", "p"):
                pn = _safe_int(v)
                if pn is not None:
                    preds.append(
                        lambda x, pn=pn: _safe_int(((x.get("meta") or {}).get("src_port"))) == pn
                        or _safe_int(((x.get("meta") or {}).get("dst_port"))) == pn
                    )
                continue

            # unknown key -> fallback to substring on blob
            preds.append(lambda x, t=t: t in _build_blob(x))
            continue

        # :53 -> port
        if t.startswith(":") and len(t) > 1:
            pn = _safe_int(t[1:])
            if pn is not None:
                preds.append(
                    lambda x, pn=pn: _safe_int(((x.get("meta") or {}).get("src_port"))) == pn
                    or _safe_int(((x.get("meta") or {}).get("dst_port"))) == pn
                )
                continue

        # proto keyword
        if t in ("udp", "tcp", "icmp"):
            preds.append(
                lambda x, t=t: _proto_name((x.get("meta") or {}).get("proto")).lower() == t
            )
            continue

        # pure number => treat as port
        pn = _safe_int(t)
        if pn is not None:
            preds.append(
                lambda x, pn=pn: _safe_int(((x.get("meta") or {}).get("src_port"))) == pn
                or _safe_int(((x.get("meta") or {}).get("dst_port"))) == pn
            )
            continue

        # heuristic: rule-ish token (dns_ttl, http_*, etc.)
        if "_" in t or t.startswith("dns"):
            preds.append(
                lambda x, t=t: t in _get_rule_name(x).lower() or t in _build_blob(x)
            )
            continue

        # default fallback:
        # prefer matching family OR blob
        preds.append(
            lambda x, t=t: t in str(x.get("family") or "").lower() or t in _build_blob(x)
        )

    return preds


class InMemoryStore:
    def __init__(self, max_events=5000, max_flows=20000):
        self.lock = threading.Lock()
        self.events = deque(maxlen=max_events)  # only suspicious/attack/rule
        self.flows = deque(maxlen=max_flows)
        self.counts = Counter()
        self.last_ts = 0.0

    def add(self, ev: dict):
        with self.lock:
            ts = float(ev.get("ts", time.time()))
            ev["ts"] = ts
            self.last_ts = ts

            verdict = ev.get("verdict", "unknown")
            self.counts[verdict] += 1

            self.flows.appendleft(ev)

            if verdict in ("suspicious", "attack") or ev.get("stage") == "rule":
                self.events.appendleft(ev)

    def get_events(self, limit=200, verdict="", since=None, until=None, src_ip=None, dst_ip=None, q=""):
        with self.lock:
            out = []
            src_q = (src_ip or "").strip()
            dst_q = (dst_ip or "").strip()
            preds = _compile_query(q)

            for x in self.events:
                ts = float(x.get("ts", 0) or 0)
                if since is not None and ts < since:
                    continue
                if until is not None and ts > until:
                    continue

                if verdict in ("benign", "suspicious", "attack") and x.get("verdict") != verdict:
                    continue

                m = x.get("meta") or {}
                sip = str(m.get("src_ip") or "")
                dip = str(m.get("dst_ip") or "")

                if src_q and src_q not in sip:
                    continue
                if dst_q and dst_q not in dip:
                    continue

                if preds and not all(p(x) for p in preds):
                    continue

                out.append(x)
                if len(out) >= limit:
                    break

            return out

    def get_flows(self, limit=500, verdict="", since=None, until=None, src_ip=None, dst_ip=None, q=""):
        with self.lock:
            out = []
            src_q = (src_ip or "").strip()
            dst_q = (dst_ip or "").strip()
            preds = _compile_query(q)

            for x in self.flows:
                ts = float(x.get("ts", 0) or 0)
                if since is not None and ts < since:
                    continue
                if until is not None and ts > until:
                    continue

                if verdict in ("benign", "suspicious", "attack") and x.get("verdict") != verdict:
                    continue

                m = x.get("meta") or {}
                sip = str(m.get("src_ip") or "")
                dip = str(m.get("dst_ip") or "")

                if src_q and src_q not in sip:
                    continue
                if dst_q and dst_q not in dip:
                    continue

                if preds and not all(p(x) for p in preds):
                    continue

                out.append(x)
                if len(out) >= limit:
                    break

            return out

    def get_stats(self):
        with self.lock:
            return {
                "counts": dict(self.counts),
                "last_ts": self.last_ts,
                "queue_len": len(self.events),
            }


STORE = InMemoryStore(max_events=5000, max_flows=20000)