# Policy Filter

`policy_filter` is a local, pre-SIEM allow-list filter for reducing log volume before logs are forwarded to downstream analytics.

The central rule is:

```text
Suppress only when definitely allowed.
Forward whenever uncertain.
```

The filter does not classify attacks, use ground-truth labels, load trained models, call threat-intelligence services, or perform online lookups. It only evaluates each raw CSV row against an organization-owned YAML or JSON policy.

Allow-list compliance is not proof that an event is benign. A compromised authorized user or an authorized program can still produce malicious activity. This module is a communication-reduction prefilter, not the final SIEM detection engine.

## Requirements

The module uses the Python standard library plus `PyYAML` for YAML policy files. JSON policies are also supported.

Install project dependencies with the repository environment, for example:

```bash
conda run -n LR python -m pip install -r requirements.txt
```

## CLI

Single CSV:

```bash
python -m policy_filter.cli \
  --logs data/org_0_logs.csv \
  --policy policy_filter/examples/org_0_policy.yaml \
  --output outputs/org_0_forwarded_logs.csv
```

Multiple CSVs:

```bash
python -m policy_filter.cli \
  --logs data/org_0_system.csv data/org_0_network.csv \
  --policy policy_filter/examples/org_0_policy.yaml \
  --output outputs/org_0_forwarded_logs.csv
```

Optional arguments:

```text
--encoding utf-8
--delimiter ,
--default-timezone UTC
--strict-policy-validation
--log-level INFO
```

The command exits nonzero if the policy is invalid or an input file cannot be opened. Malformed individual records are forwarded when possible.

## Output Columns

The output CSV contains only forwarded rows. These metadata columns are written before the original columns:

```text
source_file
source_row_number
source_line_number
filter_action
filter_reason
filter_reason_details
evaluated_category
matched_policy_id
```

Original CSV columns are preserved as strings. When multiple input files have different headers, the output uses the union of all original headers and leaves unavailable fields empty. Suppressed rows are not written.

`source_row_number` is the 1-based data row number, excluding the header. `source_line_number` is the 1-based physical line where the CSV record begins, including the header line. Quoted records containing embedded newlines are handled by the streaming CSV reader.

## Policy Schema

Policies are versioned. Version `1` supports:

- `field_mappings`: logical fields to ordered CSV column aliases.
- `category_aliases`: data-driven aliases for `system` and `network`.
- `system_policies`: host/IP selectors, identities, users, groups, allowed programs, paths, parent programs, command-line rules, validity dates, and time windows.
- `network_policies`: local selectors, remote IPs, CIDRs, domains, protocols, ports, direction, validity dates, and time windows.
- `preprocessing`: supported options passed into the shared federated preprocessing adapter.

Unknown fields are rejected in strict mode. In non-strict mode they are warned about, but unknown fields never broaden an allow-list. Missing optional allow-lists authorize nothing unless explicitly configured by a supported schema field.

Edit `policy_filter/examples/org_0_policy.yaml` to add hosts, users, programs, domains, CIDRs, protocols, ports, time windows, aliases, or policy rules without changing Python code.

## Field Resolution

For each logical field, aliases are checked in configured order. The first nonempty value is selected only if all populated aliases agree after shared canonical preprocessing. Conflicting alias values cause the row to be forwarded with `CONFLICTING_FIELD_VALUES`.

The raw row is never mutated. Policy matching uses canonical/preprocessed values from the federated pipeline adapter, while output preserves original CSV values.

## Preprocessing Reuse

`policy_filter.preprocessing_adapter` is a thin adapter over the existing federated preprocessing implementation:

- `federated_lr_pipeline.local_training.build_token_counters`
- `federated_lr_pipeline.specialized_models._field_value`
- `federated_lr_pipeline.specialized_models._row_text`
- `federated_lr_pipeline.specialized_models.field_aware_tokens`
- `federated_lr_pipeline.specialized_models.cross_tokens_for_row`

This keeps policy matching aligned with the model pipeline's row normalization and evidence construction without invoking vocabulary generation, PRF tagging, training, aggregation, inference, or IoC generation.

## Category Handling

If a configured category field is present, its value is matched against policy `category_aliases`. If no category field is present, applicability is inferred conservatively:

- system evidence: user, group, program, program path, parent program, or command line.
- network evidence: source/destination IP, domain, protocol, port, or direction.

Unknown categories are forwarded with `UNKNOWN_CATEGORY`. Rows with both system and network evidence are suppressed only when both evaluations are allowed.

## System Matching

A system row is suppressed only when all relevant checks succeed:

- an enabled and valid system policy applies to the host or local IP;
- an authorized identity matches the user or group;
- required timestamps fall inside authorized time windows;
- the program is explicitly allowed;
- configured path, parent program, and command-line constraints all match;
- every policy-required field is present and unambiguous.

Empty allow-lists do not mean allow everything.

## Network Matching

A network row is suppressed only when one configured authorized connection fully matches:

- local host/IP/network selector;
- direction;
- remote IP, CIDR, or domain;
- protocol;
- destination port or port range;
- configured time windows.

Supported network syntax includes IPv4, IPv6, CIDR ranges, exact domains, leading-wildcard domains such as `*.office.com`, integer ports, and inclusive ranges such as `8000-8100`.

Wildcard domains are intentionally limited. `*.example.com` matches `api.example.com`, but not `example.com` and not `maliciousexample.com`.

## Time Windows

Time handling is timezone-aware and never uses the machine's implicit local timezone. Policies support named weekdays, IANA timezones, multiple windows, policy validity dates, and overnight windows such as `22:00` to `06:00`.

Rows with missing, malformed, or ambiguous timestamps are forwarded when a time constraint is required.

## Reason Codes

Forwarded rows use stable reason codes, including:

```text
NO_APPLICABLE_POLICY
UNKNOWN_CATEGORY
MALFORMED_ROW
CONFLICTING_FIELD_VALUES
MISSING_REQUIRED_FIELD
INVALID_TIMESTAMP
OUTSIDE_AUTHORIZED_TIME
UNAUTHORIZED_HOST
UNAUTHORIZED_LOCAL_IP
UNAUTHORIZED_USER
UNAUTHORIZED_PROGRAM
UNAUTHORIZED_PROGRAM_PATH
UNAUTHORIZED_PARENT_PROGRAM
COMMAND_LINE_POLICY_VIOLATION
UNKNOWN_NETWORK_DIRECTION
UNAUTHORIZED_REMOTE_IP
UNAUTHORIZED_REMOTE_DOMAIN
UNAUTHORIZED_PROTOCOL
UNAUTHORIZED_PORT
NETWORK_POLICY_MISMATCH
SYSTEM_POLICY_MISMATCH
INTERNAL_FILTER_ERROR
```

If multiple checks fail, `filter_reason` contains the primary reason and `filter_reason_details` gives a short explanation. Stack traces are not written to output CSVs.

## Security Notes

- The filter never uses `eval` or `exec`.
- It never executes commands from logs or policies.
- It never resolves domains over the network.
- It does not trust an IP because it is associated with an allowed domain, or trust a domain because it resolves to an allowed IP.
- It does not use substring domain matching.
- Metadata fields are protected against CSV formula injection; original log values are preserved.
- Output writing is atomic through a temporary file followed by rename.
- Regex policies are treated as trusted administrative configuration and validated when the policy is loaded.

## Example

Given a policy that allows `alice` on `HOST-01` to run `browser.exe` during work hours and allows outbound TCP/443 to `203.0.113.20`:

- `alice` running `browser.exe` to `203.0.113.20:443/tcp` during the window is suppressed.
- `alice` running `powershell.exe` is forwarded with `UNAUTHORIZED_PROGRAM`.
- traffic to an unknown destination IP is forwarded with `UNAUTHORIZED_REMOTE_IP`.
- otherwise allowed activity outside the time window is forwarded with `OUTSIDE_AUTHORIZED_TIME`.
