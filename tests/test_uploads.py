"""End-to-end tests for per-visitor uploads on the shared demo.

These drive the real app through Streamlit's AppTest. The properties under test are the
ones that matter on a multi-visitor host: an upload is visible only to the session that
made it, nothing reaches the server's disk, and the actions that would change shared state
stay disabled.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest
from streamlit.util import calc_hash

APP = Path(__file__).resolve().parents[1] / "streamlit_app.py"
DATA_DIR = APP.parent / "data"
UPLOAD_KEY = "uploaded_dataset"

# A Kobo-style export with personal fields - exactly what must never be shared.
SURVEY_CSV = (
    b"farmer_id,full_name,phone_number,state_region,crop,farm_size_ha,yield_tons\n"
    b"1,Ada Obi,08030000001,Oyo,Maize,1.2,2.4\n"
    b"2,Musa Bello,08030000002,Kano,Sorghum,2.0,1.9\n"
    b"3,Ngozi Eze,08030000003,Oyo,Cassava,0.8,6.1\n"
    b"4,Tunde Ade,08030000004,Kaduna,Maize,1.5,2.8\n"
)


def snapshot(root: Path) -> dict[str, tuple[int, int]]:
    return {str(p): (p.stat().st_size, p.stat().st_mtime_ns) for p in root.rglob("*") if p.is_file()}


# URL path of each st.Page registered in streamlit_app.py. The dashboard is the default
# page and is served at the root.
PAGE_PATHS = {
    "Dashboard": "",
    "Crop Insights": "crop-insights",
    "Data Quality": "data-quality",
    "Model Performance": "model-performance",
    "Data Explorer": "data-explorer",
    "Uploads": "uploads",
    "Settings": "settings",
    "About": "about",
}


def open_page(at: AppTest, page: str) -> AppTest:
    """Point the session at a page, as following its sidebar link would.

    AppTest.switch_page() only resolves file-based pages; the app's pages are functions,
    which st.navigation identifies by the hash of their URL path.
    """
    at._page_hash = calc_hash(PAGE_PATHS[page]) if PAGE_PATHS[page] else ""
    return at


def new_session(page: str = "Uploads") -> AppTest:
    return open_page(AppTest.from_file(str(APP), default_timeout=180), page)


def upload(at: AppTest, name: str, content: bytes) -> AppTest:
    at.file_uploader[0].set_value((name, content, "text/csv"))
    return at.run()


def button(at: AppTest, label: str):
    matches = [b for b in at.button if b.label == label]
    assert matches, f"no button labelled {label!r}"
    return matches[0]


@pytest.fixture(autouse=True)
def demo_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("AGROSYNAPSE_MODE", raising=False)
    monkeypatch.delenv("AGROSYNAPSE_ADMIN_PASSWORD", raising=False)


def test_upload_is_explorable_and_never_written_to_disk() -> None:
    before = snapshot(DATA_DIR)

    at = new_session().run()
    upload(at, "survey.csv", SURVEY_CSV)

    assert not at.exception
    stored = at.session_state[UPLOAD_KEY]
    assert stored["name"] == "survey.csv"
    assert stored["frame"].shape == (4, 7)
    assert any("Exploring your upload" in str(s.value) for s in at.success)

    for page in ["Dashboard", "Crop Insights", "Data Quality", "Data Explorer", "Model Performance"]:
        open_page(at, page).run()
        assert not at.exception, page

    assert snapshot(DATA_DIR) == before


def test_uploads_are_isolated_between_sessions() -> None:
    visitor_a = new_session().run()
    upload(visitor_a, "survey.csv", SURVEY_CSV)
    assert UPLOAD_KEY in visitor_a.session_state

    visitor_b = new_session(page="Data Explorer").run()
    assert UPLOAD_KEY not in visitor_b.session_state
    tab_labels = [t.label for t in visitor_b.tabs]
    assert "Your upload" not in tab_labels
    assert not any("Ada Obi" in str(df.value) for df in visitor_b.dataframe)


def test_shared_state_actions_are_disabled_in_demo_mode() -> None:
    at = new_session().run()
    upload(at, "survey.csv", SURVEY_CSV)

    assert button(at, "Run pipeline").disabled
    assert button(at, "Save uploaded CSV").disabled


def test_back_to_demo_data_does_not_reingest_the_same_file() -> None:
    at = new_session().run()
    upload(at, "survey.csv", SURVEY_CSV)
    assert UPLOAD_KEY in at.session_state

    button(at, "Back to demo data").click()
    at.run()

    assert not at.exception
    assert UPLOAD_KEY not in at.session_state
    # The uploader was rotated to a fresh, empty widget.
    assert at.file_uploader[0].value is None


@pytest.mark.parametrize(
    ("content", "message"),
    [
        (b"", "could not be read"),
        (b"only_a_header\n", "no data rows"),
    ],
)
def test_unusable_files_are_rejected_without_switching_data(content: bytes, message: str) -> None:
    at = new_session().run()
    upload(at, "bad.csv", content)

    assert UPLOAD_KEY not in at.session_state
    assert any(message in str(e.value) for e in at.error)


def test_semicolon_delimited_upload_is_parsed() -> None:
    at = new_session().run()
    upload(at, "eu.csv", b"Region;Crop;Yield (t/ha)\nNorth;Maize;2,4\nSouth;Rice;3,1\n")

    frame = at.session_state[UPLOAD_KEY]["frame"]
    assert list(frame.columns) == ["region", "crop", "yield_t_ha"]
    assert len(frame) == 2


def messy_survey_csv() -> bytes:
    rows = ["farmer_id,full_name,state,crop,season_year,farm_size_ha,yield_tons"]
    states = ["Oyo", "Kano", "Kaduna", "Benue"]
    crops = ["Maize", "Cassava", "Cowpea", "maize "]
    for i in range(48):
        size = "-1.0" if i == 5 else f"{1 + (i % 4) * 0.5:.1f}"
        # "2.1t" is the classic enumerator slip; pandas already reads "n/a" as missing.
        yield_value = "2.1t" if i == 7 else f"{2 + (i % 6) * 0.3:.1f}"
        rows.append(
            f"{i + 1},Farmer {i},{states[i % 4]},{crops[i % 4]},{2020 + i % 4},{size},{yield_value}"
        )
    return ("\n".join(rows) + "\n").encode()


def test_analysis_pages_work_on_an_uploaded_survey() -> None:
    before = snapshot(DATA_DIR)
    at = new_session().run()
    upload(at, "messy.csv", messy_survey_csv())
    assert UPLOAD_KEY in at.session_state

    open_page(at, "Data Quality").run()
    assert not at.exception
    findings = at.dataframe[0].value
    found = set(zip(findings["check"], findings["column"]))
    assert ("Impossible negative values", "farm_size_ha") in found
    assert ("Numbers stored as text", "yield_tons") in found
    assert ("Inconsistent labels", "crop") in found

    open_page(at, "Crop Insights").run()
    assert not at.exception
    cards = " ".join(m.value for m in at.markdown if "smart-insight-card" in m.value)
    assert "What the data covers" in cards
    # Personal fields are never offered as a grouping column.
    assert all("full_name" not in (s.options or []) for s in at.selectbox)

    assert snapshot(DATA_DIR) == before


@pytest.mark.parametrize("page", list(PAGE_PATHS))
def test_every_page_renders_on_the_demo_data(page: str) -> None:
    at = new_session(page).run()

    assert not at.exception, page
    # The page header is custom markup; its title identifies the page that rendered.
    headers = [m.value for m in at.markdown if '<h1 class="page-topbar-title">' in m.value]
    assert len(headers) == 1, page
