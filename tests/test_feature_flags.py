"""features.yaml: parsing, defaults, and what each switch does to a job."""

import tempfile
import unittest
from pathlib import Path

from feature_flags import FEATURES, FEATURES_FILE, load_feature_details, load_features, visible_event_types
from main import apply_name_switches, describe_event
from pipeline_config import PipelineConfig

DEFAULTS = {feature.key: feature.default for feature in FEATURES}


def _write(text):
    folder = tempfile.mkdtemp()
    path = Path(folder) / "features.yaml"
    path.write_text(text, encoding="utf-8")
    return path


class LoadFeaturesTests(unittest.TestCase):
    def test_the_shipped_file_parses_without_warnings_and_matches_todays_behaviour(self):
        details, warnings = load_feature_details(FEATURES_FILE)
        self.assertEqual(warnings, [])
        self.assertEqual({item["key"] for item in details}, set(DEFAULTS))
        flags = load_features(FEATURES_FILE, log=False)
        self.assertEqual(visible_event_types(flags), ["goal", "foul"])

    def test_yes_and_no_switch_features(self):
        path = _write("detect_passes:\n  enabled: yes\ndetect_fouls:\n  enabled: no\n")
        flags = load_features(path, log=False)
        self.assertTrue(flags["detect_passes"])
        self.assertFalse(flags["detect_fouls"])
        self.assertTrue(flags["detect_goals"])  # missing from the file: default

    def test_a_bad_value_falls_back_to_the_default_and_warns(self):
        path = _write("detect_cards:\n  enabled: maybe\nfoo:\n  enabled: yes\n")
        details, warnings = load_feature_details(path)
        cards = next(item for item in details if item["key"] == "detect_cards")
        self.assertEqual(cards["enabled"], DEFAULTS["detect_cards"])
        self.assertTrue(any("detect_cards" in warning for warning in warnings))
        self.assertTrue(any("foo" in warning and "not a known feature" in warning for warning in warnings))

    def test_a_broken_or_missing_file_uses_every_default(self):
        self.assertEqual(load_features(_write("detect_goals: [unclosed"), log=False), DEFAULTS)
        self.assertEqual(load_features(Path(tempfile.mkdtemp()) / "none.yaml", log=False), DEFAULTS)

    def test_rebuild_note_is_reported(self):
        path = _write("detect_cards:\n  enabled: yes\n  app_rebuild_needed: yes\n")
        cards = next(item for item in load_feature_details(path)[0] if item["key"] == "detect_cards")
        self.assertTrue(cards["app_rebuild_needed"])


class ProfileTests(unittest.TestCase):
    def test_switches_reach_the_job_profile(self):
        flags = dict(DEFAULTS, detect_cards=True, detect_passes=True, passer_name=True,
                     commentary_names=False, highlight_clips=False, detect_penalties=True)
        config = PipelineConfig.short_clip(features=flags)
        self.assertEqual(config.output_events, ("goal", "foul", "yellow_card", "red_card", "penalty", "pass"))
        self.assertTrue(config.detect_cards)
        self.assertTrue(config.detect_penalties)
        self.assertTrue(config.name_pass_players)
        self.assertFalse(config.use_commentary)
        self.assertFalse(config.highlight_clips)

    def test_passer_name_needs_passes(self):
        config = PipelineConfig.short_clip(features=dict(DEFAULTS, passer_name=True, detect_passes=False))
        self.assertFalse(config.name_pass_players)
        self.assertNotIn("pass", config.output_events)


def _goal():
    scorer = {"role": "scorer", "track_id": 4, "team_name": "Real Madrid", "jersey_number": 7,
              "player_name": "Vinicius Junior", "jersey_confidence": 0.9}
    assister = {"role": "assister", "track_id": 5, "team_name": "Real Madrid", "jersey_number": 10,
                "player_name": "Kylian Mbappe"}
    return {"event": "goal", "team_name": "Real Madrid", "scorer": dict(scorer), "actors": [scorer, assister]}


def _pass():
    return {"event": "pass", "team_name": "Real Madrid", "actors": [
        {"role": "passer", "track_id": 4, "jersey_number": 8, "player_name": "Luka Modric"},
        {"role": "receiver", "track_id": 5, "jersey_number": 10, "player_name": "Kylian Mbappe"},
    ]}


class NameSwitchTests(unittest.TestCase):
    def test_everything_on_keeps_names(self):
        events = [_goal(), _pass()]
        apply_name_switches(events, PipelineConfig.short_clip(features=dict(DEFAULTS, detect_passes=True, passer_name=True)))
        self.assertEqual(events[0]["scorer"]["player_name"], "Vinicius Junior")
        self.assertEqual(len(events[0]["actors"]), 2)
        self.assertIn("Modric", describe_event(events[1]))
        self.assertIn("Mbappe", describe_event(events[1]))

    def test_scorer_and_assister_switches(self):
        events = [_goal()]
        apply_name_switches(events, PipelineConfig.short_clip(
            features=dict(DEFAULTS, goal_scorer_name=False, assister_name=False)))
        goal = events[0]
        self.assertIsNone(goal["scorer"]["player_name"])
        self.assertIsNone(goal["scorer"]["jersey_number"])
        self.assertEqual(goal["scorer"]["team_name"], "Real Madrid")
        self.assertEqual([actor["role"] for actor in goal["actors"]], ["scorer"])

    def test_passer_name_off_keeps_the_pass_without_players(self):
        events = [_pass()]
        apply_name_switches(events, PipelineConfig.short_clip(features=dict(DEFAULTS, detect_passes=True)))
        self.assertTrue(all(actor["player_name"] is None for actor in events[0]["actors"]))
        self.assertEqual(describe_event(events[0]), "PASS — Real Madrid")


if __name__ == "__main__":
    unittest.main()
