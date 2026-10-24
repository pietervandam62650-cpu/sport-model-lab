"""Test de fumée de app.py avec un FAUX module streamlit.

Ce test exécute réellement app.py de bout en bout (chargement, évaluation, prévision,
enregistrement, registre) et attrape erreurs de nom, de signature ou de logique dans NOTRE code.
Il ne prouve PAS la compatibilité avec la vraie bibliothèque Streamlit (non installable dans
l'environnement où le projet a été développé) : lancer `streamlit run app.py` reste nécessaire."""
import contextlib
import os
import runpy
import sys
import tempfile
import types
import unittest
from datetime import date
from pathlib import Path

from sports_lab.registry import Registry

APP = Path(__file__).resolve().parent.parent / "app.py"


class Fake:
    def __init__(self, press=False, choices=None, state=None, submit=False):
        self.press, self.choices, self.submit = press, choices or {}, submit
        self.session_state = state if state is not None else {}
        self.secrets, self.calls, self.errors = {}, [], []

    def __enter__(self): return self
    def __exit__(self, *a): return False

    # structure
    def tabs(self, labels): return [self for _ in labels]
    def columns(self, spec): return [self for _ in range(spec if isinstance(spec, int) else len(spec))]
    def form(self, *a, **k): return self
    def expander(self, *a, **k): return self
    def cache_resource(self, f=None, **k): return f if f else (lambda g: g)
    def stop(self): raise SystemExit
    def rerun(self): self.calls.append("rerun")

    # widgets
    def button(self, label, *a, **k): return self.press
    def form_submit_button(self, label, *a, **k): return self.submit
    def checkbox(self, label, value=False, **k): return value
    def radio(self, label, options, **k): return self.choices.get(label, options[0])
    def selectbox(self, label, options, index=0, **k):
        options = list(options)
        if label in self.choices:
            return self.choices[label]
        return options[index] if options else None
    def text_input(self, label, value="", **k): return value
    def text_area(self, label, value="", **k): return value
    def number_input(self, label, min_value=None, max_value=None, value=0, step=None, **k): return value
    def date_input(self, label, value=None, **k): return value
    def file_uploader(self, *a, **k): return None
    def error(self, msg, *a, **k): self.errors.append(str(msg))
    def __getattr__(self, name):  # title, caption, dataframe, metric, warning, success...
        if name.startswith("__"):
            raise AttributeError(name)
        return lambda *a, **k: None


def run_app(fake):
    mod = types.ModuleType("streamlit")
    mod.__getattr__ = lambda name: getattr(fake, name)  # PEP 562 : toute fonction st.* est servie par le faux
    saved = sys.modules.get("streamlit")
    sys.modules["streamlit"] = mod
    try:
        with contextlib.suppress(SystemExit):
            runpy.run_path(str(APP), run_name="__main__")
    finally:
        if saved is None:
            sys.modules.pop("streamlit", None)
        else:
            sys.modules["streamlit"] = saved


class AppSmoke(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = str(Path(self.tmp.name) / "app.sqlite3")
        os.environ["SPORTS_LAB_DB"] = self.db
        os.environ.pop("APP_PASSWORD", None)

    def tearDown(self):
        os.environ.pop("SPORTS_LAB_DB", None)
        self.tmp.cleanup()

    def test_start_without_data(self):
        fake = Fake()
        run_app(fake)
        self.assertNotIn("data", fake.session_state)
        self.assertEqual(fake.errors, [])

    def test_full_flow_sample_data_and_forecast_saved(self):
        fake = Fake(press=True, submit=True, choices={"Source": "Exemple (démo)", "Sport": "football"})
        run_app(fake)
        self.assertEqual(fake.errors, [], fake.errors)
        self.assertIn("football", fake.session_state["analysis"])
        self.assertNotIn("error", fake.session_state["analysis"]["football"])
        df = Registry(self.db).list_forecasts()
        self.assertEqual(len(df), 1)
        self.assertEqual(df.loc[0, "sport"], "football")
        self.assertEqual(df.loc[0, "event_date"], date.today().isoformat())
        self.assertTrue(df.loc[0, "model_version"].startswith("elo-v"))
        self.assertEqual(Registry(self.db).verify_integrity(), [])

    def test_password_gate_blocks_until_correct(self):
        os.environ["APP_PASSWORD"] = "secret"
        try:
            fake = Fake()
            run_app(fake)
            self.assertNotIn("auth", fake.session_state)
            self.assertFalse(Path(self.db).exists() and Registry(self.db).list_forecasts().shape[0])
        finally:
            os.environ.pop("APP_PASSWORD", None)


if __name__ == "__main__":
    unittest.main()
