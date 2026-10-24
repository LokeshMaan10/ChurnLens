"""Phase 15: every Streamlit page renders without exceptions (needs pipeline outputs)."""

from pathlib import Path

import pytest

APP = Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py"
PAGES = ["Executive KPIs", "Risk × value", "Segments", "Drivers", "Customer lookup",
         "Budget simulator", "Agent chat", "Live scoring (April)"]


def _built() -> bool:
    try:
        from churnlens.tools import db
        return "customers" in db.schema()
    except Exception:
        return False


@pytest.mark.skipif(not _built(), reason="pipeline outputs not built")
@pytest.mark.parametrize("page", PAGES)
def test_page_renders(page: str) -> None:
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(APP), default_timeout=300)
    at.run()
    at.sidebar.radio[0].set_value(page).run()
    if page == "Agent chat":
        at.chat_input[0].set_value("What is the churn rate in snapshot A?").run()
        assert any("6.42%" in m.value for m in at.markdown)
    assert not at.exception
