"""Tests for crawler/verify.py.

All I/O is replaced with tmp_path (no real HTTP, no real LanceDB).
deindex_permanent_failures() accepts a _delete_fn parameter for test injection,
so boto3 / lancedb dependencies are never required during test collection.

Coverage:
  reconcile()                → all URL status paths, drop alert logic
  fetch_llms_txt_urls()      → URL extraction from llms.txt body
  classify_crawl_cadence()   → changefreq + priority → cadence string
  deindex_permanent_failures() → only 4xx URLs de-indexed; report updated
  save_report / load_latest_report → round-trip
  save_failures / load_failures   → round-trip
"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import MagicMock

import pytest

from crawler.models import DiscoveredURL, FetchResult
from crawler.verify import (
    CrawlReport,
    LlmsCoverageGap,
    UrlRecord,
    classify_crawl_cadence,
    deindex_permanent_failures,
    fetch_llms_txt_urls,
    load_failures,
    load_latest_report,
    load_url_aliases,
    merge_llms_urls,
    reconcile,
    save_failures,
    save_report,
    update_url_aliases,
)

# ── Helpers ───────────────────────────────────────────────────────────────────


def _disc(url: str, changefreq: str | None = None, priority: float | None = None) -> DiscoveredURL:
    return DiscoveredURL(url=url, changefreq=changefreq, priority=priority)


def _ok(url: str, final_url: str | None = None) -> FetchResult:
    return FetchResult(
        url=url,
        final_url=final_url or url,
        status_code=200,
        html="<html>content</html>",
        fetched_at=datetime.now(UTC).isoformat(),
    )


def _redirect(url: str, final_url: str) -> FetchResult:
    return FetchResult(
        url=url,
        final_url=final_url,
        status_code=200,
        html="<html>content</html>",
        fetched_at=datetime.now(UTC).isoformat(),
    )


def _perm_fail(url: str, code: int = 404) -> FetchResult:
    return FetchResult(
        url=url,
        final_url=url,
        status_code=code,
        fetched_at=datetime.now(UTC).isoformat(),
    )


def _transient(url: str, code: int = 503) -> FetchResult:
    return FetchResult(
        url=url,
        final_url=url,
        status_code=code,
        fetched_at=datetime.now(UTC).isoformat(),
    )


def _network_err(url: str) -> FetchResult:
    return FetchResult(
        url=url,
        final_url=url,
        status_code=0,
        error="timeout: Connection timed out",
        fetched_at=datetime.now(UTC).isoformat(),
    )


def _prev_report(ok_count: int) -> CrawlReport:
    return CrawlReport(
        run_id="2026-06-01T00:00:00+00:00",
        discovered=ok_count,
        robots_excluded=0,
        fetched=ok_count,
        redirected=0,
        failed_permanent=0,
        failed_transient=0,
        not_attempted=0,
        deindexed=0,
        page_count_delta=None,
        drop_alert=False,
    )


# ── reconcile() — status classification ──────────────────────────────────────


def test_reconcile_all_fetched():
    disc = [_disc("https://example.com/a"), _disc("https://example.com/b")]
    results = [_ok("https://example.com/a"), _ok("https://example.com/b")]
    report = reconcile(disc, results)

    assert report.fetched == 2
    assert report.redirected == 0
    assert report.failed_permanent == 0
    assert report.failed_transient == 0
    assert report.not_attempted == 0
    assert report.drop_alert is False


def test_reconcile_redirected():
    disc = [_disc("https://example.com/old")]
    results = [_redirect("https://example.com/old", "https://example.com/new")]
    report = reconcile(disc, results)

    assert report.redirected == 1
    assert report.fetched == 0
    rec = report.url_records[0]
    assert rec.status == "redirected"
    assert rec.final_url == "https://example.com/new"


def test_reconcile_permanent_failure_404():
    disc = [_disc("https://example.com/gone")]
    results = [_perm_fail("https://example.com/gone", code=404)]
    report = reconcile(disc, results)

    assert report.failed_permanent == 1
    assert report.url_records[0].status == "failed_permanent"
    assert report.url_records[0].http_code == 404


def test_reconcile_permanent_failure_410():
    disc = [_disc("https://example.com/deleted")]
    results = [_perm_fail("https://example.com/deleted", code=410)]
    report = reconcile(disc, results)

    assert report.failed_permanent == 1


def test_reconcile_transient_failure_5xx():
    disc = [_disc("https://example.com/flaky")]
    results = [_transient("https://example.com/flaky", code=503)]
    report = reconcile(disc, results)

    assert report.failed_transient == 1
    assert report.url_records[0].status == "failed_transient"


def test_reconcile_transient_failure_network_error():
    disc = [_disc("https://example.com/slow")]
    results = [_network_err("https://example.com/slow")]
    report = reconcile(disc, results)

    assert report.failed_transient == 1
    assert report.url_records[0].error is not None


def test_reconcile_transient_failure_429():
    disc = [_disc("https://example.com/throttled")]
    results = [_transient("https://example.com/throttled", code=429)]
    report = reconcile(disc, results)

    assert report.failed_transient == 1


def test_reconcile_not_attempted():
    disc = [
        _disc("https://example.com/fetched"),
        _disc("https://example.com/missed"),
    ]
    # Only the first URL has a FetchResult
    results = [_ok("https://example.com/fetched")]
    report = reconcile(disc, results)

    assert report.fetched == 1
    assert report.not_attempted == 1
    missed = next(r for r in report.url_records if r.url == "https://example.com/missed")
    assert missed.status == "not_attempted"


def test_reconcile_mixed_statuses():
    disc = [
        _disc("https://example.com/a"),
        _disc("https://example.com/b"),
        _disc("https://example.com/c"),
        _disc("https://example.com/d"),
    ]
    results = [
        _ok("https://example.com/a"),
        _perm_fail("https://example.com/b"),
        _transient("https://example.com/c"),
        _redirect("https://example.com/d", "https://example.com/d-new"),
    ]
    report = reconcile(disc, results)

    assert report.fetched == 1
    assert report.redirected == 1
    assert report.failed_permanent == 1
    assert report.failed_transient == 1
    assert report.not_attempted == 0
    assert report.ok_count == 2


def test_reconcile_robots_excluded_count_in_report():
    disc = [_disc("https://example.com/allowed")]
    results = [_ok("https://example.com/allowed")]
    report = reconcile(disc, results, robots_filtered_count=5)

    assert report.robots_excluded == 5


def test_reconcile_discovered_count_matches_input():
    disc = [_disc(f"https://example.com/{i}") for i in range(10)]
    results = [_ok(f"https://example.com/{i}") for i in range(10)]
    report = reconcile(disc, results)

    assert report.discovered == 10


# ── Drop alert ────────────────────────────────────────────────────────────────


def test_drop_alert_fires_on_large_drop():
    # Previous run: 100 ok pages; current: 90 ok (10% drop > 5% threshold)
    prev = _prev_report(ok_count=100)
    disc = [_disc(f"https://example.com/{i}") for i in range(100)]
    results = [_ok(f"https://example.com/{i}") for i in range(90)] + [
        _perm_fail(f"https://example.com/{i}") for i in range(90, 100)
    ]
    report = reconcile(disc, results, previous_report=prev)

    assert report.drop_alert is True
    assert report.page_count_delta == -10


def test_drop_alert_does_not_fire_on_small_drop():
    # Previous run: 100 ok pages; current: 97 ok (3% drop < 5% threshold)
    prev = _prev_report(ok_count=100)
    disc = [_disc(f"https://example.com/{i}") for i in range(100)]
    results = [_ok(f"https://example.com/{i}") for i in range(97)] + [
        _perm_fail(f"https://example.com/{i}") for i in range(97, 100)
    ]
    report = reconcile(disc, results, previous_report=prev)

    assert report.drop_alert is False


def test_drop_alert_does_not_fire_at_exact_threshold():
    # Previous run: 100; current: 95 → exactly 5% drop, threshold is STRICTLY greater
    prev = _prev_report(ok_count=100)
    disc = [_disc(f"https://example.com/{i}") for i in range(100)]
    results = [_ok(f"https://example.com/{i}") for i in range(95)] + [
        _perm_fail(f"https://example.com/{i}") for i in range(95, 100)
    ]
    report = reconcile(disc, results, previous_report=prev)

    assert report.drop_alert is False


def test_drop_alert_not_fired_on_first_run():
    # No previous report → no baseline, so never alert
    disc = [_disc(f"https://example.com/{i}") for i in range(5)]
    results = [_perm_fail(f"https://example.com/{i}") for i in range(5)]
    report = reconcile(disc, results, previous_report=None)

    assert report.drop_alert is False
    assert report.page_count_delta is None


def test_drop_alert_positive_delta_no_alert():
    # Page count increased — should not alert
    prev = _prev_report(ok_count=80)
    disc = [_disc(f"https://example.com/{i}") for i in range(90)]
    results = [_ok(f"https://example.com/{i}") for i in range(90)]
    report = reconcile(disc, results, previous_report=prev)

    assert report.drop_alert is False
    assert report.page_count_delta == 10


# ── save_report / load_latest_report ─────────────────────────────────────────


def test_save_report_creates_file(tmp_path):
    disc = [_disc("https://example.com/a")]
    results = [_ok("https://example.com/a")]
    report = reconcile(disc, results)

    path = save_report(report, staging_dir=str(tmp_path))
    assert path.exists()
    assert path.suffix == ".json"


def test_save_load_report_roundtrip(tmp_path):
    disc = [_disc("https://example.com/a"), _disc("https://example.com/b")]
    results = [_ok("https://example.com/a"), _perm_fail("https://example.com/b")]
    original = reconcile(disc, results)

    save_report(original, staging_dir=str(tmp_path))
    loaded = load_latest_report(staging_dir=str(tmp_path))

    assert loaded is not None
    assert loaded.run_id == original.run_id
    assert loaded.fetched == original.fetched
    assert loaded.failed_permanent == original.failed_permanent
    assert loaded.drop_alert == original.drop_alert
    assert len(loaded.url_records) == len(original.url_records)


def test_load_latest_report_returns_none_when_no_reports(tmp_path):
    result = load_latest_report(staging_dir=str(tmp_path))
    assert result is None


def test_load_latest_report_returns_most_recent(tmp_path):
    # Save two reports and verify we get the latest one
    disc = [_disc("https://example.com/a")]

    report1 = reconcile(disc, [_ok("https://example.com/a")])
    report1_id = "2026-01-01T00:00:00+00:00"
    report1.run_id = report1_id
    save_report(report1, staging_dir=str(tmp_path))

    report2 = reconcile(disc, [_perm_fail("https://example.com/a")])
    report2_id = "2026-06-01T00:00:00+00:00"
    report2.run_id = report2_id
    save_report(report2, staging_dir=str(tmp_path))

    latest = load_latest_report(staging_dir=str(tmp_path))
    assert latest is not None
    assert latest.run_id == report2_id


# ── save_failures / load_failures ────────────────────────────────────────────


def test_save_failures_creates_file(tmp_path):
    disc = [_disc("https://example.com/gone")]
    results = [_perm_fail("https://example.com/gone")]
    report = reconcile(disc, results)

    path = save_failures(report, staging_dir=str(tmp_path))
    assert path.exists()
    assert path.name == "failures.jsonl"


def test_save_load_failures_roundtrip(tmp_path):
    disc = [
        _disc("https://example.com/perm"),
        _disc("https://example.com/transient"),
        _disc("https://example.com/ok"),
    ]
    results = [
        _perm_fail("https://example.com/perm"),
        _transient("https://example.com/transient"),
        _ok("https://example.com/ok"),
    ]
    report = reconcile(disc, results)
    save_failures(report, staging_dir=str(tmp_path))

    loaded = load_failures(staging_dir=str(tmp_path))
    failure_urls = {e["url"] for e in loaded}

    assert "https://example.com/perm" in failure_urls
    assert "https://example.com/transient" in failure_urls
    assert "https://example.com/ok" not in failure_urls


def test_load_failures_empty_when_no_file(tmp_path):
    result = load_failures(staging_dir=str(tmp_path))
    assert result == []


def test_save_failures_overwrites_on_second_run(tmp_path):
    disc1 = [_disc("https://example.com/old-fail")]
    report1 = reconcile(disc1, [_perm_fail("https://example.com/old-fail")])
    save_failures(report1, staging_dir=str(tmp_path))

    disc2 = [_disc("https://example.com/new-fail")]
    report2 = reconcile(disc2, [_transient("https://example.com/new-fail")])
    save_failures(report2, staging_dir=str(tmp_path))

    loaded = load_failures(staging_dir=str(tmp_path))
    urls = {e["url"] for e in loaded}

    # Only the second run's failures remain
    assert "https://example.com/new-fail" in urls
    assert "https://example.com/old-fail" not in urls


def test_save_failures_not_attempted_included(tmp_path):
    disc = [
        _disc("https://example.com/a"),
        _disc("https://example.com/missing"),  # no fetch result
    ]
    results = [_ok("https://example.com/a")]
    report = reconcile(disc, results)
    save_failures(report, staging_dir=str(tmp_path))

    loaded = load_failures(staging_dir=str(tmp_path))
    urls = {e["url"] for e in loaded}
    assert "https://example.com/missing" in urls


# ── classify_crawl_cadence() ─────────────────────────────────────────────────


@pytest.mark.parametrize(
    "changefreq,expected",
    [
        ("always", "weekly"),
        ("hourly", "weekly"),
        ("daily", "weekly"),
        ("weekly", "weekly"),
        ("monthly", "monthly"),
        ("yearly", "yearly"),
        ("never", "yearly"),
        ("", "weekly"),  # empty → default
    ],
)
def test_classify_cadence_by_changefreq(changefreq, expected):
    url = _disc("https://example.com/page", changefreq=changefreq)
    assert classify_crawl_cadence(url) == expected


@pytest.mark.parametrize(
    "priority,expected",
    [
        (1.0, "weekly"),
        (0.8, "weekly"),
        (0.7, "weekly"),
        (0.6, "monthly"),
        (0.5, "monthly"),
        (0.4, "monthly"),
        (0.3, "yearly"),
        (0.1, "yearly"),
    ],
)
def test_classify_cadence_by_priority(priority, expected):
    url = _disc("https://example.com/page", priority=priority)
    assert classify_crawl_cadence(url) == expected


def test_classify_cadence_changefreq_takes_precedence_over_priority():
    # changefreq="yearly" + priority=1.0 → priority is ignored
    url = _disc("https://example.com/legal", changefreq="yearly", priority=1.0)
    assert classify_crawl_cadence(url) == "yearly"


def test_classify_cadence_default_when_no_data():
    url = _disc("https://example.com/page")
    assert classify_crawl_cadence(url) == "weekly"


# ── deindex_permanent_failures() ─────────────────────────────────────────────


def test_deindex_calls_delete_for_permanent_failures_only(tmp_path):
    mock_delete = MagicMock(return_value=3)

    disc = [
        _disc("https://example.com/gone"),
        _disc("https://example.com/ok"),
        _disc("https://example.com/slow"),
    ]
    results = [
        _perm_fail("https://example.com/gone", code=404),
        _ok("https://example.com/ok"),
        _transient("https://example.com/slow"),
    ]
    report = reconcile(disc, results)
    deindexed = deindex_permanent_failures(report, index_uri=str(tmp_path), _delete_fn=mock_delete)

    mock_delete.assert_called_once()
    called_urls = mock_delete.call_args[0][0]
    assert "https://example.com/gone" in called_urls
    assert "https://example.com/ok" not in called_urls
    assert "https://example.com/slow" not in called_urls

    assert deindexed == ["https://example.com/gone"]
    assert report.deindexed == 1

    gone_rec = next(r for r in report.url_records if r.url == "https://example.com/gone")
    assert gone_rec.action == "deindexed"


def test_deindex_no_permanent_failures_skips_delete():
    mock_delete = MagicMock()

    disc = [_disc("https://example.com/ok")]
    results = [_ok("https://example.com/ok")]
    report = reconcile(disc, results)
    deindexed = deindex_permanent_failures(
        report, index_uri="./lance_index", _delete_fn=mock_delete
    )

    mock_delete.assert_not_called()
    assert deindexed == []
    assert report.deindexed == 0


def test_deindex_transient_failure_not_deindexed():
    mock_delete = MagicMock(return_value=0)

    disc = [_disc("https://example.com/flaky")]
    results = [_transient("https://example.com/flaky")]
    report = reconcile(disc, results)
    deindex_permanent_failures(report, index_uri="./lance_index", _delete_fn=mock_delete)

    mock_delete.assert_not_called()


def test_deindex_updates_report_deindexed_count():
    mock_delete = MagicMock(return_value=2)

    disc = [
        _disc("https://example.com/a"),
        _disc("https://example.com/b"),
    ]
    results = [
        _perm_fail("https://example.com/a"),
        _perm_fail("https://example.com/b"),
    ]
    report = reconcile(disc, results)
    deindex_permanent_failures(report, index_uri="./lance_index", _delete_fn=mock_delete)

    assert report.deindexed == 2


# ── fetch_llms_txt_urls() ─────────────────────────────────────────────────────


def test_fetch_llms_txt_urls_extracts_https_links(httpx_mock):
    body = (
        "# Appther Overview\n"
        "See our services at https://www.appther.com/services/odoo\n"
        "More at https://www.appther.com/faq and https://www.appther.com/case-study/acme.\n"
    )
    httpx_mock.add_response(url="https://www.appther.com/llms-full.txt", text=body, status_code=200)

    import httpx as _httpx

    with _httpx.Client() as client:
        urls = fetch_llms_txt_urls(client=client)

    assert "https://www.appther.com/services/odoo" in urls
    assert "https://www.appther.com/faq" in urls
    # Trailing dot stripped from the case-study URL
    assert "https://www.appther.com/case-study/acme" in urls


def test_fetch_llms_txt_urls_returns_empty_on_failure(httpx_mock):
    httpx_mock.add_response(url="https://www.appther.com/llms-full.txt", status_code=404)
    httpx_mock.add_response(url="https://www.appther.com/llms.txt", status_code=404)

    import httpx as _httpx

    with _httpx.Client() as client:
        urls = fetch_llms_txt_urls(client=client)

    assert urls == []


def test_fetch_llms_txt_urls_deduplicates(httpx_mock):
    body = "https://www.appther.com/faq\nSee also https://www.appther.com/faq for details.\n"
    httpx_mock.add_response(url="https://www.appther.com/llms-full.txt", text=body, status_code=200)

    import httpx as _httpx

    with _httpx.Client() as client:
        urls = fetch_llms_txt_urls(client=client)

    faq_occurrences = urls.count("https://www.appther.com/faq")
    assert faq_occurrences == 1


# ── UrlRecord serialization ───────────────────────────────────────────────────


def test_url_record_to_from_dict_roundtrip():
    rec = UrlRecord(
        url="https://example.com/page",
        status="fetched",
        http_code=200,
        final_url=None,
        error=None,
        suggested_cadence="weekly",
        action=None,
    )
    restored = UrlRecord.from_dict(rec.to_dict())
    assert restored.url == rec.url
    assert restored.status == rec.status
    assert restored.http_code == rec.http_code
    assert restored.suggested_cadence == rec.suggested_cadence


# ── CrawlReport serialization ─────────────────────────────────────────────────


def test_crawl_report_to_from_dict_roundtrip():
    disc = [_disc("https://example.com/a"), _disc("https://example.com/b")]
    results = [_ok("https://example.com/a"), _perm_fail("https://example.com/b")]
    report = reconcile(disc, results)

    restored = CrawlReport.from_dict(report.to_dict())

    assert restored.run_id == report.run_id
    assert restored.fetched == report.fetched
    assert restored.failed_permanent == report.failed_permanent
    assert len(restored.url_records) == len(report.url_records)
    assert restored.ok_count == report.ok_count


def test_crawl_report_ok_count_property():
    disc = [_disc("https://example.com/a"), _disc("https://example.com/b")]
    results = [
        _ok("https://example.com/a"),
        _redirect("https://example.com/b", "https://example.com/b-new"),
    ]
    report = reconcile(disc, results)

    assert report.ok_count == 2


# ── C3: de-index key canonicalization + redirect alias map ───────────────────


def test_deindex_lowercases_mixed_case_path_key():
    """A page indexed under a case-folded path must be de-indexed under that same
    lower-cased key, not the raw mixed-case discovered URL (C3 path-case)."""
    mock_delete = MagicMock(return_value=3)
    disc = [_disc("https://appther.com/company-DelhiNCR")]
    results = [_perm_fail("https://appther.com/company-DelhiNCR", code=404)]
    report = reconcile(disc, results)

    keys = deindex_permanent_failures(report, index_uri="./idx", _delete_fn=mock_delete)

    called = mock_delete.call_args[0][0]
    assert called == ["https://appther.com/company-delhincr"]
    assert keys == ["https://appther.com/company-delhincr"]
    assert "https://appther.com/company-DelhiNCR" not in called
    gone_rec = report.url_records[0]
    assert gone_rec.action == "deindexed"
    assert report.deindexed == 1


def test_deindex_resolves_redirect_alias():
    """When a failed URL has a recorded redirect alias, de-index uses the target
    key it was actually stored under (C3 redirect-shift)."""
    mock_delete = MagicMock(return_value=3)
    aliases = {"https://appther.com/industries/odoo": "https://appther.com/industry/odoo"}
    disc = [_disc("https://appther.com/industries/odoo")]
    results = [_perm_fail("https://appther.com/industries/odoo", code=404)]
    report = reconcile(disc, results)

    keys = deindex_permanent_failures(
        report, index_uri="./idx", aliases=aliases, _delete_fn=mock_delete
    )

    assert mock_delete.call_args[0][0] == ["https://appther.com/industry/odoo"]
    assert keys == ["https://appther.com/industry/odoo"]


def test_deindex_plain_lowercase_url_unaffected():
    """Regression guard: an already-lower-case URL with no alias still de-indexes
    under its own key."""
    mock_delete = MagicMock(return_value=3)
    disc = [_disc("https://appther.com/old-page")]
    results = [_perm_fail("https://appther.com/old-page", code=410)]
    report = reconcile(disc, results)

    keys = deindex_permanent_failures(report, index_uri="./idx", _delete_fn=mock_delete)

    assert keys == ["https://appther.com/old-page"]
    assert mock_delete.call_args[0][0] == ["https://appther.com/old-page"]


def test_deindex_dedupes_mixed_case_duplicates():
    """Two mixed-case variants of the same page collapse to one canonical delete
    key, but both records are annotated."""
    mock_delete = MagicMock(return_value=6)
    disc = [
        _disc("https://appther.com/company-DelhiNCR"),
        _disc("https://appther.com/company-delhincr"),
    ]
    results = [
        _perm_fail("https://appther.com/company-DelhiNCR"),
        _perm_fail("https://appther.com/company-delhincr"),
    ]
    report = reconcile(disc, results)

    keys = deindex_permanent_failures(report, index_uri="./idx", _delete_fn=mock_delete)

    assert keys == ["https://appther.com/company-delhincr"]
    assert mock_delete.call_count == 1
    assert report.deindexed == 2
    assert all(r.action == "deindexed" for r in report.url_records)


def test_deindex_alias_takes_precedence_over_path_case():
    """An alias keyed on the canonicalized original wins even when the original
    also needs path lower-casing."""
    mock_delete = MagicMock(return_value=3)
    # alias key is the canonical (lower-cased) form of the discovered URL
    aliases = {"https://appther.com/industries/delhincr": "https://appther.com/industry/delhincr"}
    disc = [_disc("https://appther.com/industries/DelhiNCR")]
    results = [_perm_fail("https://appther.com/industries/DelhiNCR", code=404)]
    report = reconcile(disc, results)

    keys = deindex_permanent_failures(
        report, index_uri="./idx", aliases=aliases, _delete_fn=mock_delete
    )

    assert keys == ["https://appther.com/industry/delhincr"]


def test_update_and_load_url_aliases_roundtrip(tmp_path):
    """A redirected URL is persisted as origin-canonical → target-canonical and
    can be read back."""
    disc = [_disc("https://appther.com/industries/odoo")]
    results = [
        _redirect("https://appther.com/industries/odoo", "https://appther.com/industry/odoo")
    ]
    report = reconcile(disc, results)

    saved = update_url_aliases(report, staging_dir=str(tmp_path))
    assert saved["https://appther.com/industries/odoo"] == "https://appther.com/industry/odoo"

    loaded = load_url_aliases(staging_dir=str(tmp_path))
    assert loaded == {"https://appther.com/industries/odoo": "https://appther.com/industry/odoo"}


def test_load_url_aliases_empty_when_no_file(tmp_path):
    assert load_url_aliases(staging_dir=str(tmp_path)) == {}


def test_update_url_aliases_only_records_redirects(tmp_path):
    """fetched / failed URLs do not produce alias entries; only redirects do."""
    disc = [
        _disc("https://appther.com/fetched"),
        _disc("https://appther.com/gone"),
        _disc("https://appther.com/moved"),
    ]
    results = [
        _ok("https://appther.com/fetched"),
        _perm_fail("https://appther.com/gone"),
        _redirect("https://appther.com/moved", "https://appther.com/moved-new"),
    ]
    report = reconcile(disc, results)

    saved = update_url_aliases(report, staging_dir=str(tmp_path))
    assert saved == {"https://appther.com/moved": "https://appther.com/moved-new"}


def test_update_url_aliases_preserves_existing(tmp_path):
    """Older mappings survive across runs; newer runs only add to them."""
    report1 = reconcile(
        [_disc("https://appther.com/a-old")],
        [_redirect("https://appther.com/a-old", "https://appther.com/a-new")],
    )
    update_url_aliases(report1, staging_dir=str(tmp_path))

    report2 = reconcile(
        [_disc("https://appther.com/b-old")],
        [_redirect("https://appther.com/b-old", "https://appther.com/b-new")],
    )
    merged = update_url_aliases(report2, staging_dir=str(tmp_path))

    assert merged["https://appther.com/a-old"] == "https://appther.com/a-new"
    assert merged["https://appther.com/b-old"] == "https://appther.com/b-new"


def test_end_to_end_redirect_shift_deindex(tmp_path):
    """Full C3 redirect-shift flow: record the alias on the run the URL still
    redirects, then de-index it on a later run when the original 404s."""
    # Run 1 — the URL still redirects (HTTP 200).
    redirect_report = reconcile(
        [_disc("https://appther.com/industries/odoo")],
        [_redirect("https://appther.com/industries/odoo", "https://appther.com/industry/odoo")],
    )
    update_url_aliases(redirect_report, staging_dir=str(tmp_path))

    # Run 2 — the original now 404s.
    fail_report = reconcile(
        [_disc("https://appther.com/industries/odoo")],
        [_perm_fail("https://appther.com/industries/odoo", code=404)],
    )
    aliases = load_url_aliases(staging_dir=str(tmp_path))
    mock_delete = MagicMock(return_value=4)
    keys = deindex_permanent_failures(
        fail_report, index_uri="./idx", aliases=aliases, _delete_fn=mock_delete
    )
    assert keys == ["https://appther.com/industry/odoo"]
    assert mock_delete.call_args[0][0] == ["https://appther.com/industry/odoo"]


# ── C4: partial (targeted) runs never alert and never become baseline ──────────────


def _c4_report(ok_count: int, run_id: str, partial: bool) -> CrawlReport:
    """Build a full or partial report with `ok_count` fetched URLs."""
    disc = [_disc(f"https://www.appther.com/p{i}") for i in range(ok_count)]
    results = [_ok(f"https://www.appther.com/p{i}") for i in range(ok_count)]
    report = reconcile(disc, results, partial=partial)
    report.run_id = run_id
    return report


def test_partial_run_suppresses_drop_alert_vs_full_baseline():
    """A targeted re-run of a few URLs must not trip the drop alert just
    because its ok_count is tiny next to a full-crawl baseline."""
    prev = _prev_report(ok_count=100)
    disc = [_disc(f"https://www.appther.com/fail{i}") for i in range(3)]
    results = [_ok(f"https://www.appther.com/fail{i}") for i in range(3)]
    report = reconcile(disc, results, previous_report=prev, partial=True)
    assert report.partial is True
    assert report.drop_alert is False
    assert report.page_count_delta is None


def test_full_run_still_alerts_when_not_partial():
    """The same numbers on a FULL run must still fire the alert (no regression)."""
    prev = _prev_report(ok_count=100)
    disc = [_disc(f"https://www.appther.com/fail{i}") for i in range(3)]
    results = [_ok(f"https://www.appther.com/fail{i}") for i in range(3)]
    report = reconcile(disc, results, previous_report=prev, partial=False)
    assert report.partial is False
    assert report.drop_alert is True


def test_partial_report_excluded_from_baseline_selection(tmp_path):
    """A newer partial report must not be chosen as the baseline over an older
    full report."""
    full = _c4_report(ok_count=80, run_id="2026-01-01T00:00:00+00:00", partial=False)
    save_report(full, staging_dir=str(tmp_path))
    partial = _c4_report(ok_count=3, run_id="2026-06-01T00:00:00+00:00", partial=True)
    save_report(partial, staging_dir=str(tmp_path))

    latest = load_latest_report(staging_dir=str(tmp_path))
    assert latest is not None
    assert latest.partial is False
    assert latest.run_id == "2026-01-01T00:00:00+00:00"
    assert latest.ok_count == 80


def test_load_latest_report_none_when_only_partial(tmp_path):
    """If only partial reports exist there is no valid baseline."""
    partial = _c4_report(ok_count=3, run_id="2026-06-01T00:00:00+00:00", partial=True)
    save_report(partial, staging_dir=str(tmp_path))
    assert load_latest_report(staging_dir=str(tmp_path)) is None


def test_partial_report_roundtrip():
    """to_dict/from_dict must preserve the partial flag."""
    report = _c4_report(ok_count=2, run_id="2026-06-01T00:00:00+00:00", partial=True)
    restored = CrawlReport.from_dict(report.to_dict())
    assert restored.partial is True


def test_legacy_report_without_partial_field_defaults_false():
    """A report serialized before C4 (no `partial` key) loads as a full run."""
    report = _c4_report(ok_count=2, run_id="2026-06-01T00:00:00+00:00", partial=False)
    d = report.to_dict()
    d.pop("partial", None)
    restored = CrawlReport.from_dict(d)
    assert restored.partial is False


# ── H1: llms.txt coverage cross-check ──────────────────────────────────────────────


class _FakeRobots:
    """Minimal robots stand-in for merge_llms_urls tests (no network/parsing)."""

    def __init__(self, disallow: tuple[str, ...] = ()) -> None:
        self._disallow = disallow

    def is_allowed(self, url: str) -> bool:
        return not any(token in url for token in self._disallow)


def test_merge_llms_urls_adds_missing_pages():
    discovered = [_disc("https://www.appther.com/a")]
    llms = ["https://www.appther.com/a", "https://www.appther.com/b"]
    augmented, blocked = merge_llms_urls(discovered, llms, _FakeRobots())
    urls = [d.url for d in augmented]
    assert "https://www.appther.com/b" in urls
    assert blocked == []
    added = [d for d in augmented if d.url == "https://www.appther.com/b"]
    assert added[0].source == "llms"


def test_merge_llms_urls_dedupes_by_canonical_key():
    # Mixed-case path already discovered -> not added again (C3 canonical key).
    discovered = [_disc("https://www.appther.com/company-DelhiNCR")]
    llms = ["https://www.appther.com/company-delhincr"]
    augmented, blocked = merge_llms_urls(discovered, llms, _FakeRobots())
    assert len(augmented) == 1
    assert blocked == []


def test_merge_llms_urls_blocks_robots_disallowed():
    discovered: list[DiscoveredURL] = []
    llms = ["https://www.appther.com/vendor/secret"]
    augmented, blocked = merge_llms_urls(discovered, llms, _FakeRobots(disallow=("/vendor/",)))
    assert augmented == []
    assert blocked == ["https://www.appther.com/vendor/secret"]


def test_reconcile_flags_uncrawled_llms_urls():
    # /a crawled OK, /b attempted but 404, /c robots-blocked (never discovered).
    discovered = [_disc("https://www.appther.com/a"), _disc("https://www.appther.com/b")]
    results = [_ok("https://www.appther.com/a"), _perm_fail("https://www.appther.com/b")]
    llms = [
        "https://www.appther.com/a",
        "https://www.appther.com/b",
        "https://www.appther.com/c",
    ]
    report = reconcile(
        discovered,
        results,
        llms_urls=llms,
        llms_blocked=["https://www.appther.com/c"],
    )
    gaps = {g.url: g.reason for g in report.llms_uncrawled}
    assert "https://www.appther.com/a" not in gaps
    assert gaps["https://www.appther.com/b"] == "fetch_failed"
    assert gaps["https://www.appther.com/c"] == "robots_blocked"


def test_reconcile_llms_not_attempted_reason():
    # /d referenced + discovered but absent from fetch results -> not_attempted.
    discovered = [_disc("https://www.appther.com/d")]
    results: list[FetchResult] = []
    report = reconcile(
        discovered,
        results,
        llms_urls=["https://www.appther.com/d"],
    )
    gaps = {g.url: g.reason for g in report.llms_uncrawled}
    assert gaps["https://www.appther.com/d"] == "not_attempted"


def test_reconcile_without_llms_urls_leaves_empty():
    report = reconcile(
        [_disc("https://www.appther.com/a")],
        [_ok("https://www.appther.com/a")],
    )
    assert report.llms_uncrawled == []


def test_report_roundtrip_preserves_llms_uncrawled():
    report = reconcile(
        [_disc("https://www.appther.com/b")],
        [_perm_fail("https://www.appther.com/b")],
        llms_urls=["https://www.appther.com/b"],
    )
    restored = CrawlReport.from_dict(report.to_dict())
    assert [(g.url, g.reason) for g in restored.llms_uncrawled] == [
        (g.url, g.reason) for g in report.llms_uncrawled
    ]
    assert all(isinstance(g, LlmsCoverageGap) for g in restored.llms_uncrawled)


def test_legacy_report_without_llms_uncrawled_defaults_empty():
    report = reconcile(
        [_disc("https://www.appther.com/a")],
        [_ok("https://www.appther.com/a")],
    )
    d = report.to_dict()
    d.pop("llms_uncrawled", None)
    restored = CrawlReport.from_dict(d)
    assert restored.llms_uncrawled == []
