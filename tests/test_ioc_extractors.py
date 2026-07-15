from __future__ import annotations

from ioc_generation.extractors import extract_iocs


def test_extracts_ips_domains_urls_and_hashes() -> None:
    text = (
        "GET http://evil.example.com/path from 192.168.1.5 with "
        "md5 d41d8cd98f00b204e9800998ecf8427e and "
        "sha256 e3b0c44298fc1c149afbf4c8996fb924"
        "27ae41e4649b934ca495991b7852b855"
    )

    extracted = {(candidate.indicator_type, candidate.value) for candidate in extract_iocs(text)}

    assert ("ipv4", "192.168.1.5") in extracted
    assert ("url", "http://evil.example.com/path") in extracted
    assert ("domain", "evil.example.com") in extracted
    assert ("md5", "d41d8cd98f00b204e9800998ecf8427e") in extracted
    assert (
        "sha256",
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    ) in extracted


def test_does_not_extract_identifier_fields_as_hashes() -> None:
    text = (
        "normalized_id=368c710a8a0a763186eab09971d12717 "
        "entity_id=000062ac9da6cdaebe3fc1477043fc57 "
        "md5 d41d8cd98f00b204e9800998ecf8427e"
    )

    extracted = {(candidate.indicator_type, candidate.value) for candidate in extract_iocs(text)}

    assert ("md5", "368c710a8a0a763186eab09971d12717") not in extracted
    assert ("md5", "000062ac9da6cdaebe3fc1477043fc57") not in extracted
    assert ("md5", "d41d8cd98f00b204e9800998ecf8427e") in extracted


def test_does_not_extract_iocs_from_url_paths() -> None:
    text = (
        "remote_address=https://ellechina.online/01_logo_HLW-300x168.jpg "
        "download=http://200.98.142.12/system/MA-1.0.0.0/fbclient.dll"
    )

    extracted = {(candidate.indicator_type, candidate.value) for candidate in extract_iocs(text)}

    assert ("url", "https://ellechina.online/01_logo_HLW-300x168.jpg") in extracted
    assert ("domain", "ellechina.online") in extracted
    assert ("domain", "-300x168.jpg") not in extracted
    assert ("domain", "300x168.jpg") not in extracted
    assert ("ipv4", "200.98.142.12") in extracted
    assert ("ipv4", "1.0.0.0") not in extracted
