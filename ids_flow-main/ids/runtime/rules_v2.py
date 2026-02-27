from dataclasses import dataclass
from typing import Optional, Dict, Any

@dataclass
class RuleHit:
    name: str
    severity: str  # "high" | "medium"
    score: float   # 0..1
    reason: str

def _f(flow: Dict[str, Any], key: str, default: float = 0.0) -> float:
    v = flow.get(key, default)
    try:
        return float(v)
    except Exception:
        return float(default)

def _i(flow: Dict[str, Any], key: str, default: int = 0) -> int:
    v = flow.get(key, default)
    try:
        return int(float(v))
    except Exception:
        return int(default)

DNS_QTYPES = {1, 2, 5, 6, 12, 15, 16, 28, 33, 255}  # A,NS,CNAME,SOA,PTR,MX,TXT,AAAA,SRV,ANY

def run_rules(flow: Dict[str, Any]) -> Optional[RuleHit]:
    """
    Rule engine v2:
    - 2–3 HIGH rules: rất chắc -> stage=rule (block luôn)
    - 5–10 MEDIUM rules: gợi ý -> attach rule_info, vẫn qua ML gate
    """
    proto = _i(flow, "PROTOCOL", -1)
    in_pkts  = _f(flow, "IN_PKTS")
    out_pkts = _f(flow, "OUT_PKTS")
    pkts = in_pkts + out_pkts

    in_bytes  = _f(flow, "IN_BYTES")
    out_bytes = _f(flow, "OUT_BYTES")
    bytes_ = in_bytes + out_bytes

    dur_ms = _f(flow, "FLOW_DURATION_MILLISECONDS")
    dur_s = max(1e-6, dur_ms / 1000.0)

    rin_pkts  = _f(flow, "RETRANSMITTED_IN_PKTS")
    rout_pkts = _f(flow, "RETRANSMITTED_OUT_PKTS")
    rratio = (rin_pkts + rout_pkts) / max(1.0, pkts)

    thr_in  = _f(flow, "SRC_TO_DST_AVG_THROUGHPUT")
    thr_out = _f(flow, "DST_TO_SRC_AVG_THROUGHPUT")
    thr = max(thr_in, thr_out)

    icmp_type  = _i(flow, "ICMP_TYPE", -1)
    icmp4_type = _i(flow, "ICMP_IPV4_TYPE", -1)

    # ports (có thì dùng, không có thì -1)
    src_port = _i(flow, "SRC_PORT", -1)
    dst_port = _i(flow, "DST_PORT", -1)

    # DNS fields (có thì dùng)
    dns_id    = _i(flow, "DNS_QUERY_ID", -1)
    dns_qtype = _i(flow, "DNS_QUERY_TYPE", -1)
    dns_ttl   = _f(flow, "DNS_TTL_ANSWER", -1.0)

    is_dns = (dns_id > 0) or (dns_qtype in (1, 2, 5, 6, 12, 15, 16, 28, 33, 255))
    # chỉ xét nếu có dấu hiệu DNS thật + có traffic tối thiểu
    if is_dns and dns_ttl == 0 and pkts >= 3 and dur_ms > 0:
        return RuleHit(
            name="DNS_TTL_ANOMALY",
            severity="medium",
            score=0.55,
            reason=f"dns_id={dns_id} qtype={dns_qtype} ttl={dns_ttl} pkts={pkts:.0f}",
        )


    # =========================
    # HIGH confidence
    # =========================

    # H1) ICMP burst: pkts cực nhiều trong thời gian ngắn
    if (icmp_type != -1 or icmp4_type != -1) and pkts >= 2000 and 0 < dur_ms <= 5000:
        return RuleHit(
            name="ICMP_BURST",
            severity="high",
            score=0.97,
            reason=f"icmp={icmp_type}/{icmp4_type} pkts={pkts:.0f} dur_ms={dur_ms:.0f}",
        )

    # H2) Retransmission ratio cực cao (flow đủ lớn)
    if pkts >= 200 and rratio >= 0.60:
        return RuleHit(
            name="HIGH_RETRANSMISSION_RATIO",
            severity="high",
            score=0.95,
            reason=f"rratio={rratio:.2f} pkts={pkts:.0f}",
        )

    # H3) Throughput cực cao trong short duration (ngưỡng sẽ tune sau)
    if 0 < dur_ms <= 3000 and thr >= 1e8:
        return RuleHit(
            name="THROUGHPUT_EXTREME_SHORT",
            severity="high",
            score=0.93,
            reason=f"thr={thr:.1e} dur_ms={dur_ms:.0f} pkts={pkts:.0f}",
        )

    # =========================
    # MEDIUM confidence
    # =========================

    # M1) DNS TTL anomaly (bớt bắn nhầm)
    # - cần dấu hiệu DNS thật: qtype hợp lệ hoặc dns_id>0
    # - ưu tiên nếu thấy port 53
    # - yêu cầu pkts/bytes tối thiểu để tránh dòng rác
    is_dns_hint = (dns_id > 0) or (dns_qtype in DNS_QTYPES)
    is_dns_port = (src_port == 53) or (dst_port == 53)
    if is_dns_hint and (is_dns_port or dns_qtype in DNS_QTYPES) and pkts >= 2 and bytes_ >= 60:
        if dns_ttl == 0:
            return RuleHit(
                name="DNS_TTL_ANOMALY",
                severity="medium",
                score=0.55,
                reason=f"dns_id={dns_id} qtype={dns_qtype} ttl={dns_ttl} sport={src_port} dport={dst_port}",
            )

    # M2) Short burst pkts
    if 0 < dur_ms <= 1000 and pkts >= 500:
        return RuleHit(
            name="SHORT_PKT_BURST",
            severity="medium",
            score=0.65,
            reason=f"pkts={pkts:.0f} dur_ms={dur_ms:.0f}",
        )

    # M3) Short burst bytes
    if 0 < dur_ms <= 2000 and bytes_ >= 5e7:
        return RuleHit(
            name="SHORT_BYTES_BURST",
            severity="medium",
            score=0.62,
            reason=f"bytes={bytes_:.0f} dur_ms={dur_ms:.0f}",
        )

    # M4) Throughput spike (nhẹ hơn HIGH)
    if thr >= 5e7:
        return RuleHit(
            name="THROUGHPUT_SPIKE",
            severity="medium",
            score=0.60,
            reason=f"thr={thr:.1e}",
        )

    # M5) Retrans ratio đáng nghi
    if pkts >= 100 and rratio >= 0.35:
        return RuleHit(
            name="RETRANSMISSION_SUSPECT",
            severity="medium",
            score=0.58,
            reason=f"rratio={rratio:.2f} pkts={pkts:.0f}",
        )

    # M6) UDP burst
    if proto == 17 and 0 < dur_ms <= 2000 and pkts >= 800:
        return RuleHit(
            name="UDP_BURST",
            severity="medium",
            score=0.62,
            reason=f"pkts={pkts:.0f} dur_ms={dur_ms:.0f}",
        )

    return None
