"""Shared fixtures for crawler tests."""

from __future__ import annotations

from pathlib import Path

import pytest

FIXTURES_DIR = Path(__file__).parent / "fixtures"


def fixture_text(name: str) -> str:
    return (FIXTURES_DIR / name).read_text(encoding="utf-8")


@pytest.fixture()
def sitemap_index_xml() -> str:
    return fixture_text("sitemap_index.xml")


@pytest.fixture()
def page_sitemap_xml() -> str:
    return fixture_text("page_sitemap.xml")


@pytest.fixture()
def blog_sitemap_xml() -> str:
    return fixture_text("blog_sitemap.xml")


@pytest.fixture()
def robots_txt() -> str:
    return fixture_text("robots.txt")


@pytest.fixture()
def homepage_html() -> str:
    return fixture_text("homepage.html")


@pytest.fixture()
def main_urlset_xml() -> str:
    """A <urlset> root sitemap matching the live appther.com structure (no sitemapindex)."""
    return fixture_text("main_urlset.xml")


@pytest.fixture()
def blog_subdomain_sitemap_xml() -> str:
    """A blog sitemap whose URLs live on the blog.appther.com subdomain."""
    return fixture_text("blog_subdomain_sitemap.xml")
