"""Tek seferlik script: docs/forecasts.json'in git gecmisindeki TUM eski surumlerini
cikarip docs/history/<COIN>.jsonl arsivine ekler.

generate_forecasts.py yeni calismaya baslamadan ONCE de bu repo'da forecasts.json
dosyasi zaten periyodik olarak commit'leniyordu (bkz. `git log -- docs/forecasts.json`).
Bu script o commit'lerin her birindeki tam anlik goruntuyu okuyup, canli akisin
kullandigi AYNI build_archive_record/append_archive_record fonksiyonlariyla
(mantik kopyalanmadi) arsiv formatina donusturur.

Kullanim (repo kok dizininden, bir kere calistirilir):
    python scripts/backfill_from_git.py
"""
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from generate_forecasts import append_archive_record, build_archive_record  # noqa: E402

TARGET_PATH = "docs/forecasts.json"


def run_git(*args: str) -> str:
    result = subprocess.run(["git", *args], capture_output=True, text=True, check=True)
    return result.stdout


def list_commits() -> list:
    """TARGET_PATH'i degistiren tum commit'ler, eskiden yeniye siralanmis."""
    out = run_git("log", "--follow", "--reverse", "--format=%H", "--", TARGET_PATH)
    return [line.strip() for line in out.splitlines() if line.strip()]


def load_snapshot(commit_hash: str) -> dict | None:
    try:
        raw = run_git("show", f"{commit_hash}:{TARGET_PATH}")
    except subprocess.CalledProcessError:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None


def main() -> int:
    commits = list_commits()
    print(f"{len(commits)} commit bulundu ({TARGET_PATH} icin).")

    written = 0
    skipped = 0
    for commit_hash in commits:
        snapshot = load_snapshot(commit_hash)
        if not snapshot or "coins" not in snapshot:
            skipped += 1
            continue

        generated_at = snapshot.get("generated_at")
        for symbol, coin in snapshot["coins"].items():
            history = coin.get("history") or []
            forecast_points = coin.get("forecast") or []
            if not history or not forecast_points or not generated_at:
                continue

            entry_ts = history[-1]["ts"]
            entry_price = history[-1]["close"]
            signal = coin.get("signal")

            record = build_archive_record(
                symbol, coin.get("inst_id", ""), generated_at, entry_ts,
                entry_price, signal, forecast_points,
            )
            # Backfill sirasinda budama yapma (butun gecmis onemli); canli akis
            # zaten kendi ARCHIVE_MAX_RECORDS'una gore ileride budayacak.
            append_archive_record(symbol, record, max_records=100_000)
            written += 1

    print(f"{written} kayit arsivlendi, {skipped} commit atlandi (eski/bozuk format).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
