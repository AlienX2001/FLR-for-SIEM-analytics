from __future__ import annotations

from bisect import bisect_right
from decimal import Decimal, InvalidOperation
from typing import Any


NUMERIC_BUCKETING_VERSION = 1
NUMERIC_BUCKETING_STRATEGY = "fixed-port-and-log-magnitude-buckets"

PORT_COLUMNS = frozenset(
    {
        "src_port",
        "dst_port",
        "local_port",
        "remote_port",
    }
)

PROCESS_ID_COLUMNS = frozenset(
    {
        "process_pid",
        "process_ppid",
        "process_tgid",
    }
)

MEASUREMENT_COLUMNS = frozenset(
    {
        "flow_duration",
        "duration",
        "rate",
        "srate",
        "drate",
        "header_length",
        "total_size",
        "total_sum",
        "packet_number",
        "iat",
        "ack_count",
        "syn_count",
        "fin_count",
        "urg_count",
        "rst_count",
    }
)

HIGH_CARDINALITY_NUMERIC_COLUMNS = frozenset(
    PORT_COLUMNS | PROCESS_ID_COLUMNS | MEASUREMENT_COLUMNS
)

# Boundaries are fixed across organizations and runs. This avoids leaking local
# distributions and guarantees that equal values map to equal feature tokens.
MAGNITUDE_BUCKET_UPPER_BOUNDS = (
    Decimal("0.000001"),
    Decimal("0.001"),
    Decimal("0.01"),
    Decimal("0.1"),
    Decimal("1"),
    Decimal("2"),
    Decimal("4"),
    Decimal("8"),
    Decimal("16"),
    Decimal("32"),
    Decimal("64"),
    Decimal("128"),
    Decimal("256"),
    Decimal("512"),
    Decimal("1024"),
    Decimal("2048"),
    Decimal("4096"),
    Decimal("8192"),
    Decimal("16384"),
    Decimal("32768"),
    Decimal("65536"),
    Decimal("131072"),
    Decimal("262144"),
    Decimal("524288"),
    Decimal("1048576"),
    Decimal("2097152"),
    Decimal("4194304"),
    Decimal("8388608"),
    Decimal("16777216"),
    Decimal("33554432"),
    Decimal("67108864"),
    Decimal("134217728"),
    Decimal("268435456"),
    Decimal("536870912"),
    Decimal("1073741824"),
)

MAGNITUDE_BUCKET_LABELS = (
    "lt_1e-6",
    "1e-6_to_lt_1e-3",
    "1e-3_to_lt_1e-2",
    "1e-2_to_lt_1e-1",
    "1e-1_to_lt_1",
    "1_to_lt_2",
    "2_to_lt_4",
    "4_to_lt_8",
    "8_to_lt_16",
    "16_to_lt_32",
    "32_to_lt_64",
    "64_to_lt_128",
    "128_to_lt_256",
    "256_to_lt_512",
    "512_to_lt_1024",
    "1024_to_lt_2048",
    "2048_to_lt_4096",
    "4096_to_lt_8192",
    "8192_to_lt_16384",
    "16384_to_lt_32768",
    "32768_to_lt_65536",
    "65536_to_lt_131072",
    "131072_to_lt_262144",
    "262144_to_lt_524288",
    "524288_to_lt_1048576",
    "1048576_to_lt_2097152",
    "2097152_to_lt_4194304",
    "4194304_to_lt_8388608",
    "8388608_to_lt_16777216",
    "16777216_to_lt_33554432",
    "33554432_to_lt_67108864",
    "67108864_to_lt_134217728",
    "134217728_to_lt_268435456",
    "268435456_to_lt_536870912",
    "536870912_to_lt_1073741824",
    "ge_1073741824",
)

if len(MAGNITUDE_BUCKET_LABELS) != len(MAGNITUDE_BUCKET_UPPER_BOUNDS) + 1:
    raise RuntimeError("Numeric bucket labels and boundaries are inconsistent")


# Exact tokens are retained only for this fixed, low-cardinality service set.
# All other valid ports are represented solely by their IANA range class.
WELL_KNOWN_PORT_SERVICES = {
    20: "ftp_data",
    21: "ftp",
    22: "ssh",
    23: "telnet",
    25: "smtp",
    53: "dns",
    67: "dhcp_server",
    68: "dhcp_client",
    69: "tftp",
    80: "http",
    110: "pop3",
    123: "ntp",
    135: "rpc",
    137: "netbios_ns",
    138: "netbios_dgm",
    139: "netbios_ssn",
    143: "imap",
    161: "snmp",
    162: "snmptrap",
    389: "ldap",
    443: "https",
    445: "smb",
    465: "smtps",
    514: "syslog",
    587: "submission",
    636: "ldaps",
    993: "imaps",
    995: "pop3s",
    1433: "mssql",
    1521: "oracle",
    2049: "nfs",
    2375: "docker",
    3306: "mysql",
    3389: "rdp",
    5432: "postgresql",
    5672: "amqp",
    5900: "vnc",
    6379: "redis",
    8080: "http_alt",
    8443: "https_alt",
    9200: "elasticsearch",
    27017: "mongodb",
}


def _as_finite_decimal(value: Any) -> Decimal | None:
    try:
        numeric = Decimal(str(value).strip())
    except (InvalidOperation, ValueError):
        return None
    if not numeric.is_finite():
        return None
    return numeric


def _magnitude_bucket(numeric: Decimal) -> str:
    if numeric < 0:
        return "invalid_negative"
    if numeric == 0:
        return "zero"
    index = bisect_right(MAGNITUDE_BUCKET_UPPER_BOUNDS, numeric)
    return MAGNITUDE_BUCKET_LABELS[index]


def _port_tokens(field: str, numeric: Decimal) -> list[str]:
    if numeric != numeric.to_integral_value() or numeric < 0 or numeric > 65535:
        return [f"{field}:class=invalid"]

    port = int(numeric)
    if port == 0:
        port_class = "reserved"
    elif port <= 1023:
        port_class = "system"
    elif port <= 49151:
        port_class = "registered"
    else:
        port_class = "dynamic"

    tokens = [f"{field}:class={port_class}"]
    service = WELL_KNOWN_PORT_SERVICES.get(port)
    if service is not None:
        tokens.extend((f"{field}:service={service}", f"{field}={port}"))
    return tokens


def numeric_bucket_tokens(field: str, value: Any) -> list[str] | None:
    """Return fixed bucket tokens, or ``None`` for non-bucketed/malformed values.

    Returning ``None`` lets the shared field-aware tokenizer preserve its
    existing fail-open string behavior for malformed direct-call inputs.
    """
    if field not in HIGH_CARDINALITY_NUMERIC_COLUMNS:
        return None

    numeric = _as_finite_decimal(value)
    if numeric is None:
        return None
    if field in PORT_COLUMNS:
        return _port_tokens(field, numeric)
    if field in PROCESS_ID_COLUMNS and numeric != numeric.to_integral_value():
        return [f"{field}:bucket=invalid"]
    return [f"{field}:bucket={_magnitude_bucket(numeric)}"]


def numeric_bucketing_metadata() -> dict[str, object]:
    return {
        "version": NUMERIC_BUCKETING_VERSION,
        "fields": sorted(HIGH_CARDINALITY_NUMERIC_COLUMNS),
        "strategy": NUMERIC_BUCKETING_STRATEGY,
    }
