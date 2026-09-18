import json

import pytest

from voxprint.cli import main

CONSENT = ["--consent", "synthetic voice, no real person", "--consent-by", "test fixture"]


def run(capsys, *argv):
    code = main(list(argv))
    captured = capsys.readouterr()
    return code, captured.out, captured.err


@pytest.fixture
def root(tmp_path):
    return str(tmp_path / "voices")


@pytest.fixture
def enrolled(root, clips, capsys):
    for name, files in clips.items():
        main(["--root", root, "enroll", *files[:3], "--id", name, *CONSENT])
    main(["--root", root, "calibrate"])
    capsys.readouterr()
    return root


def test_enroll_reports_progress(capsys, root, clips):
    code, out, _ = run(capsys, "--root", root, "enroll", *clips["omar"][:3], "--id", "omar", *CONSENT)
    assert code == 0
    assert "enrolled omar" in out
    assert "consistency" in out


def test_enroll_without_consent_explains_the_flag(capsys, root, clips):
    code, _, err = run(capsys, "--root", root, "enroll", clips["omar"][0], "--id", "omar")
    assert code == 1
    assert "--consent" in err and "--no-consent-check" in err


def test_no_consent_check_allows_enrolment(capsys, root, clips):
    code, out, _ = run(
        capsys, "--root", root, "--no-consent-check", "enroll", clips["omar"][0], "--id", "omar"
    )
    assert code == 0
    assert "enrolled omar" in out


def test_identify_returns_zero_on_a_match(capsys, enrolled, clips):
    code, out, _ = run(capsys, "--root", enrolled, "identify", clips["laila"][3])
    assert code == 0
    assert "MATCH: laila" in out


def test_identify_returns_two_for_an_unknown_voice(capsys, root, clips):
    for name in ("omar", "hana", "sami"):
        main(["--root", root, "enroll", *clips[name][:3], "--id", name, *CONSENT])
    main(["--root", root, "calibrate"])
    capsys.readouterr()
    code, out, _ = run(capsys, "--root", root, "identify", clips["khaled"][0])
    assert code == 2
    assert "UNKNOWN" in out


def test_identify_json_is_machine_readable(capsys, enrolled, clips):
    code, out, _ = run(capsys, "--root", enrolled, "--json", "identify", clips["sami"][3], "--top", "2")
    payload = json.loads(out)
    assert code == 0
    assert payload["decision"] == "match"
    assert payload["candidates"][0]["speaker_id"] == "sami"
    assert len(payload["candidates"]) == 2


def test_identify_on_an_empty_gallery(capsys, root, clips):
    code, out, err = run(capsys, "--root", root, "identify", clips["omar"][0])
    assert code == 1
    assert "empty" in (out + err)


def test_verify_exit_codes(capsys, enrolled, clips):
    assert run(capsys, "--root", enrolled, "verify", clips["omar"][3], "--id", "omar")[0] == 0
    assert run(capsys, "--root", enrolled, "verify", clips["omar"][3], "--id", "hana")[0] == 2


def test_list_shows_consent_state(capsys, enrolled):
    code, out, _ = run(capsys, "--root", enrolled, "list")
    assert code == 0
    assert "consent recorded" in out
    assert "5 enrolled speaker(s)" in out


def test_calibrate_reports_error_rates(capsys, root, clips):
    for name, files in clips.items():
        main(["--root", root, "enroll", *files[:3], "--id", name, *CONSENT])
    capsys.readouterr()
    code, out, _ = run(capsys, "--root", root, "calibrate")
    assert code == 0
    assert "equal error rate" in out
    assert "real-world error will be higher" in out


def test_calibrate_needs_two_speakers(capsys, root, clips):
    main(["--root", root, "enroll", *clips["omar"][:2], "--id", "omar", *CONSENT])
    capsys.readouterr()
    code, _, err = run(capsys, "--root", root, "calibrate")
    assert code == 1
    assert "two enrolled speakers" in err


def test_remove_deletes_the_speaker(capsys, enrolled):
    run(capsys, "--root", enrolled, "remove", "--id", "omar")
    _, out, _ = run(capsys, "--root", enrolled, "list")
    assert "omar" not in out


def test_convert_writes_a_file_and_reports_similarity(capsys, enrolled, clips, tmp_path):
    out_path = tmp_path / "converted.wav"
    code, out, _ = run(
        capsys, "--root", enrolled, "convert", clips["omar"][0], "--id", "hana", "-o", str(out_path)
    )
    assert code == 0
    assert out_path.exists()
    assert "similarity of the output" in out
    assert "would NOT be recognised" in out


def test_watermark_add_then_check(capsys, clips, tmp_path):
    marked = tmp_path / "marked.wav"
    assert run(capsys, "watermark", "add", clips["omar"][0], "-o", str(marked))[0] == 0
    code, out, _ = run(capsys, "watermark", "check", str(marked))
    assert code == 0 and "DETECTED" in out

    code, out, _ = run(capsys, "watermark", "check", clips["omar"][0])
    assert code == 2 and "not detected" in out


def test_watermark_add_requires_an_output(capsys, clips):
    code, _, err = run(capsys, "watermark", "add", clips["omar"][0])
    assert code == 1 and "--out" in err


def test_backends_lists_status_and_licence(capsys):
    code, out, _ = run(capsys, "backends")
    assert code == 0
    assert "dspvc" in out and "xtts" in out
    assert "CPML" in out


def test_info_summarises_the_gallery(capsys, enrolled):
    code, out, _ = run(capsys, "--root", enrolled, "info")
    assert code == 0
    assert "speakers       5" in out


def test_selftest_reports_measured_accuracy(capsys):
    code, out, _ = run(capsys, "--json", "selftest", "--speakers", "6", "--seconds", "2.5")
    payload = json.loads(out)
    assert code == 0
    assert payload["closed_set_accuracy"] >= 0.8
    assert payload["open_set_rejection"] >= 0.8


def test_missing_file_is_a_clean_error(capsys, enrolled, tmp_path):
    code, _, err = run(capsys, "--root", enrolled, "identify", str(tmp_path / "nope.wav"))
    assert code == 1
    assert "error:" in err
