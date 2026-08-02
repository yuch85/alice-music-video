#!/usr/bin/env python3
"""Burn JP/EN subtitles onto v2 video, then append end credits.

Steps:
  1. Load transcript for segment timing.
  2. Write hardcoded JP + EN subtitle text to temp files.
  3. Build ffmpeg drawtext filter chain (textfile= approach).
  4. Burn subtitles into v2 video -> v3.
  5. Append credits sequence -> final.
"""
from __future__ import annotations

import json
import logging
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _mv_credits import build_final_with_credits  # noqa: E402

# --- Paths ---
TRANSCRIPT_PATH = Path(
    os.environ.get("MV_TRANSCRIPT", "transcript.json")
)
INPUT_VIDEO = Path(
    os.environ.get("MV_OUTPUT_DIR", "output/")
    "project-name-17seg-v2.mp4"
)
OUTPUT_DIR = INPUT_VIDEO.parent
V3_VIDEO = OUTPUT_DIR / "project-name-17seg-v3.mp4"
FINAL_VIDEO = OUTPUT_DIR / "project-name-17seg-final.mp4"
DOWNLOADS_VIDEO = Path(os.environ.get("MV_INPUT_VIDEO", "input.mp4"))

# --- Fonts ---
JP_FONT = Path(os.environ.get("MV_JP_FONT", "/path/to/NotoSansCJKjp-Regular.otf"))
EN_FONT = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")

# --- Subtitle styles ---
JP_FONT_SIZE = 30
EN_FONT_SIZE = 22
JP_Y_OFFSET = 100  # pixels from bottom (above EN)
EN_Y_OFFSET = 65   # pixels from bottom
OUTLINE_WIDTH = 2

# --- Hardcoded subtitle text (segment index -> (JP, EN)) ---
# Only singer segments 0-13 get subtitles. Broll 14-16 are skipped.
SUBTITLE_TEXT: dict[int, tuple[str, str]] = {
    0: (
        "山の朝はまだ夜の続き 冷たい川がどこかへ剥がれてゆく",
        "The mountain morning is still night's continuation / A cold river peels away somewhere",
    ),
    1: (
        "君がいなくなって感想もしたけど 声は届かない 呼ぶことしかできない",
        "You're gone — I had my thoughts, but my voice can't reach you / All I can do is call out",
    ),
    2: (
        "それ以上のことが僕にはない どこかで息をしているはずなら 戻",
        "I have nothing more to give / If you're still breathing somewhere, come back",
    ),
    3: (
        "戻っておいで 戻っておいで この場所にまだいるから 君が",
        "Come back to me / I'm still here, in this place, for you",
    ),
    4: (
        "見えた景色を それだけを願っていた",
        "The scenery you saw / That's all I wished for",
    ),
    5: (
        "君は戻ってきた 静かな顔で何かを決めたいよ 柔らかい目で山の",
        "You came back / With a quiet face, wanting to decide something / With gentle eyes, the mountains —",
    ),
    6: (
        "の空気が二人の間を通り抜けた うまく聞けなかった 何があったのか",
        "The mountain air passed between us / I couldn't quite ask / what happened",
    ),
    7: (
        "分かろうとしている 分かれなくも いつか分かる 君がここに立っている",
        "I'm trying to understand / Not because we're parting — someday I'll know / You're standing here",
    ),
    8: (
        "よかったよかった この場所にまだいてくれた 君が",
        "How good, how good / That you're still here, in this place — you,",
    ),
    9: (
        "見えた景色を それだけが今はいいいい いつか",
        "The scenery you saw / That's all that's good right now / Someday",
    ),
    10: (
        "君が全部話してくれたら うまく答えられないかもしれない",
        "When you tell me everything / I might not know how to answer",
    ),
    11: (
        "でも隣に座っていられる それだけはたぶんできる 戻",
        "But I can sit beside you / That's the one thing I probably can / Come back",
    ),
    12: (
        "戻っておいで 戻っておいで 僕はここにいつもいる 民の",
        "Come back to me / I'm always here",
    ),
    13: (
        "意味なんていらない 今夜はここで一緒にいよう",
        "We don't need meaning / Tonight, let's just be together here",
    ),
}

log = logging.getLogger(__name__)


def _write_text_file(dir: Path, name: str, text: str) -> Path:
    """Write subtitle text to a temp file for drawtext textfile= param."""
    p = dir / name
    p.write_text(text, encoding="utf-8")
    return p


def _make_drawtext(
    *,
    text_file: Path,
    font: Path,
    fontsize: int,
    fontcolor: str,
    y_offset: int,
    start: float,
    end: float,
  # unused, kept for signature compat
) -> str:
    """Build a single drawtext filter string."""
    parts = [
        f"textfile={text_file}",
        f"fontfile={font}",
        f"fontsize={fontsize}",
        f"fontcolor={fontcolor}",
        f"bordercolor=black",
        f"borderw={OUTLINE_WIDTH}",
        f"x=(w-tw)/2",
        f"y=h-th-{y_offset}",
        f"enable=between(t\\,{start}\\,{end})",
    ]
    return "drawtext=" + ":".join(parts)


def build_subtitle_filter(segments: list[dict]) -> tuple[str, str]:
    """Build the full drawtext filter chain for all singer segments.

    Returns (filter_string, temp_dir_path). The caller must clean up temp_dir.
    """
    tmp_dir = tempfile.mkdtemp(prefix="mv_subs_")
    tmp = Path(tmp_dir)
    filters: list[str] = []

    for seg in segments:
        idx = seg["index"]
        if idx not in SUBTITLE_TEXT:
            continue  # skip broll / non-singer segments
        jp_text, en_text = SUBTITLE_TEXT[idx]
        start = seg["start"]
        end = seg["end"]

        jp_file = _write_text_file(tmp, f"jp_{idx:03d}.txt", jp_text)
        en_file = _write_text_file(tmp, f"en_{idx:03d}.txt", en_text)

        filters.append(
            _make_drawtext(
                text_file=jp_file,
                font=JP_FONT,
                fontsize=JP_FONT_SIZE,
                fontcolor="white",
                y_offset=JP_Y_OFFSET,
                start=start,
                end=end,
            )
        )
        filters.append(
            _make_drawtext(
                text_file=en_file,
                font=EN_FONT,
                fontsize=EN_FONT_SIZE,
                fontcolor="white",
                y_offset=EN_Y_OFFSET,
                start=start,
                end=end,
            )
        )

    return ",".join(filters), tmp_dir


def burn_subtitles(input_video: Path, output_video: Path) -> Path:
    """Burn JP/EN subtitles into video using NVENC encode."""
    transcript = json.loads(TRANSCRIPT_PATH.read_text(encoding="utf-8"))
    segments = transcript["segments"]
    singer_segments = [s for s in segments if s["index"] in SUBTITLE_TEXT]
    log.info("Burning subtitles for %d singer segments", len(singer_segments))

    filter_chain, tmp_dir = build_subtitle_filter(segments)
    try:
        filter_complex = f"[0:v]{filter_chain}[vout]"
        cmd = [
            "ffmpeg", "-y",
            "-i", str(input_video),
            "-filter_complex", filter_complex,
            "-map", "[vout]",
            "-map", "0:a",
            "-c:v", "h264_nvenc", "-cq", "23", "-b:v", "0",
            "-c:a", "aac", "-b:a", "192k",
            "-movflags", "+faststart",
            str(output_video),
        ]
        log.info("Running ffmpeg subtitle burn (%d drawtext filters)", len(singer_segments) * 2)
        t0 = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
        if result.returncode != 0:
            log.error("Subtitle burn failed: %s", result.stderr[-2000:])
            raise RuntimeError(f"Subtitle burn failed (rc={result.returncode})")
        log.info("Subtitle burn done in %.0fs", time.time() - t0)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    return output_video


def add_credits(sub_video: Path, final_path: Path) -> Path:
    """Append end credits sequence to subtitle-burned video."""
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(sub_video)],
        capture_output=True, text=True, check=True,
    )
    duration = float(result.stdout.strip())
    log.info("Subtitle video duration: %.2fs", duration)

    return build_final_with_credits(
        concat_path=sub_video,
        final_path=final_path,
        total_duration=duration,
    )


def main() -> None:
    """Entry point: burn subtitles, add credits, copy to downloads."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if not JP_FONT.exists():
        log.error("Japanese font not found: %s", JP_FONT)
        sys.exit(1)
    if not EN_FONT.exists():
        log.error("English font not found: %s", EN_FONT)
        sys.exit(1)
    if not INPUT_VIDEO.exists():
        log.error("Input video not found: %s", INPUT_VIDEO)
        sys.exit(1)

    # Step 1: Burn subtitles -> v3
    burn_subtitles(INPUT_VIDEO, V3_VIDEO)

    # Step 2: Add credits -> final
    add_credits(V3_VIDEO, FINAL_VIDEO)

    # Step 3: Copy final to downloads
    DOWNLOADS_VIDEO.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(FINAL_VIDEO, DOWNLOADS_VIDEO)
    log.info("Copied final video to %s", DOWNLOADS_VIDEO)

    # Step 4: Verify
    for label, path in [("v3 (subs)", V3_VIDEO), ("final (+credits)", FINAL_VIDEO)]:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries",
             "format=duration,size", "-of", "default=noprint_wrappers=1",
             str(path)],
            capture_output=True, text=True, check=True,
        )
        log.info("%s: %s (%s)", label, path.name, result.stdout.strip())


if __name__ == "__main__":
    main()
