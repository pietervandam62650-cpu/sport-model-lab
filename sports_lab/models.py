"""Modèles de prévision. Chaque modèle est séquentiel : predict() n'utilise que
ce qui a été vu via update(), ce qui interdit toute fuite de données futures.

Modèles :
- UniformBaseline / FrequencyBaseline : références simples ;
- EloModel : Elo avec nul de type Davidson (football) ou à 2 issues (autres sports) ;
- PoissonModel : forces d'attaque/défense mises à jour en ligne (football).
  Poissons indépendants : tend à sous-estimer les nuls (pas de Dixon-Coles).
"""
from __future__ import annotations

import math

import numpy as np

from .config import sport_config


def outcome_of(home_score, away_score, has_draw: bool) -> str:
    if home_score > away_score:
        return "home"
    if home_score < away_score:
        return "away"
    if not has_draw:
        raise ValueError("Égalité impossible dans un marché sans nul.")
    return "draw"


class Model:
    name = "base"

    def __init__(self, has_draw: bool):
        self.has_draw = has_draw
        self.teams: set = set()

    def predict(self, home, away):  # -> (p_home, p_draw, p_away)
        raise NotImplementedError

    def update(self, home, away, hs, as_):
        self.teams.update((home, away))

    def params(self) -> dict:
        return {}


class UniformBaseline(Model):
    name = "baseline_uniforme"

    def predict(self, home, away):
        return (1 / 3, 1 / 3, 1 / 3) if self.has_draw else (0.5, 0.0, 0.5)


class FrequencyBaseline(Model):
    """Fréquences passées des issues (lissage de Laplace), sans info sur les équipes."""
    name = "baseline_frequences"

    def __init__(self, has_draw: bool):
        super().__init__(has_draw)
        self.counts = {"home": 0, "draw": 0, "away": 0}

    def predict(self, home, away):
        k = 3 if self.has_draw else 2
        n = sum(self.counts.values())
        ph = (self.counts["home"] + 1) / (n + k)
        pa = (self.counts["away"] + 1) / (n + k)
        pdr = (self.counts["draw"] + 1) / (n + k) if self.has_draw else 0.0
        return ph, pdr, pa

    def update(self, home, away, hs, as_):
        super().update(home, away, hs, as_)
        self.counts[outcome_of(hs, as_, self.has_draw)] += 1


class EloModel(Model):
    name = "elo"

    def __init__(self, has_draw, k=20.0, home_adv=55.0, scale=400.0, initial=1500.0,
                 draw_prior=0.25, draw_prior_weight=10.0):
        super().__init__(has_draw)
        self.k, self.home_adv, self.scale, self.initial = k, home_adv, scale, initial
        self.draw_prior, self.draw_w = draw_prior, draw_prior_weight
        self.ratings: dict = {}
        self.n = 0
        self.draws = 0

    def params(self):
        return {"k": self.k, "home_adv": self.home_adv}

    def draw_rate(self) -> float:
        r = (self.draws + self.draw_prior * self.draw_w) / (self.n + self.draw_w)
        return min(0.45, max(0.02, r))

    def predict(self, home, away):
        d = self.ratings.get(home, self.initial) + self.home_adv - self.ratings.get(away, self.initial)
        if not self.has_draw:
            ph = 1.0 / (1.0 + 10.0 ** (-d / self.scale))
            return ph, 0.0, 1.0 - ph
        r = self.draw_rate()
        nu = 2.0 * r / (1.0 - r)  # Davidson : P(nul) = r quand d = 0
        x = d / (2.0 * self.scale)
        a, b = 10.0 ** x, 10.0 ** (-x)
        z = a + b + nu
        return a / z, nu / z, b / z

    def update(self, home, away, hs, as_):
        ph, pdr, _ = self.predict(home, away)
        expected = ph + 0.5 * pdr
        actual = 1.0 if hs > as_ else (0.5 if hs == as_ else 0.0)
        delta = self.k * (actual - expected)
        self.ratings[home] = self.ratings.get(home, self.initial) + delta
        self.ratings[away] = self.ratings.get(away, self.initial) - delta
        self.n += 1
        self.draws += int(hs == as_)
        super().update(home, away, hs, as_)


_LOGFACT = np.array([math.lgamma(i + 1) for i in range(41)])


class PoissonModel(Model):
    name = "poisson_online"

    def __init__(self, lr=0.03, reg=0.002, max_goals=10, prior_home=1.45, prior_away=1.15, prior_weight=10.0):
        super().__init__(True)
        self.lr, self.reg, self.G = lr, reg, min(max_goals, 40)
        self.prior_home, self.prior_away, self.prior_w = prior_home, prior_away, prior_weight
        self.att: dict = {}
        self.dfn: dict = {}
        self.n, self.sum_h, self.sum_a = 0, 0.0, 0.0

    def params(self):
        return {"lr": self.lr, "reg": self.reg}

    def _base(self):
        w = self.prior_w
        return ((self.sum_h + self.prior_home * w) / (self.n + w),
                (self.sum_a + self.prior_away * w) / (self.n + w))

    def lambdas(self, home, away):
        bh, ba = self._base()
        lh = bh * math.exp(self.att.get(home, 0.0) - self.dfn.get(away, 0.0))
        la = ba * math.exp(self.att.get(away, 0.0) - self.dfn.get(home, 0.0))
        return lh, la

    def _pmf(self, lam):
        k = np.arange(self.G + 1)
        return np.exp(-lam + k * math.log(lam) - _LOGFACT[: self.G + 1])

    def grid(self, home, away):
        lh, la = self.lambdas(home, away)
        g = np.outer(self._pmf(lh), self._pmf(la))
        return g / g.sum()

    def predict(self, home, away):
        g = self.grid(home, away)
        return float(np.tril(g, -1).sum()), float(np.trace(g)), float(np.triu(g, 1).sum())

    def markets(self, home, away) -> dict:
        g = self.grid(home, away)
        ph, pdr, pa = self.predict(home, away)
        idx = np.add.outer(np.arange(g.shape[0]), np.arange(g.shape[1]))
        return {"home": ph, "draw": pdr, "away": pa,
                "over_2_5": float(g[idx >= 3].sum()), "btts": float(g[1:, 1:].sum())}

    def update(self, home, away, hs, as_):
        lh, la = self.lambdas(home, away)
        eh, ea = hs - lh, as_ - la
        for d, team, step in ((self.att, home, eh), (self.dfn, away, -eh),
                              (self.att, away, ea), (self.dfn, home, -ea)):
            v = d.get(team, 0.0) + self.lr * step
            d[team] = float(np.clip(v * (1 - self.reg), -1.5, 1.5))
        self.n += 1
        self.sum_h += hs
        self.sum_a += as_
        super().update(home, away, hs, as_)


def build_models(sport: str, has_draw: bool, elo_params: dict | None = None,
                 poisson_params: dict | None = None) -> dict:
    cfg = sport_config(sport)
    elo = {"k": cfg["k"], "home_adv": cfg["home_adv"], **(elo_params or {})}
    models = {
        "baseline_uniforme": UniformBaseline(has_draw),
        "baseline_frequences": FrequencyBaseline(has_draw),
        "elo": EloModel(has_draw, **elo),
    }
    if cfg["goals_model"] and has_draw:
        models["poisson_online"] = PoissonModel(**(poisson_params or {}))
    return models
