#!/usr/bin/env python3
"""
compare_pitch_detection.py

用 Basic Pitch 检测录音，和参考谱面对比算准确率。
同时支持小提琴的 Basic Pitch vs librosa pyin 对比。

安装依赖：
    pip install basic-pitch librosa soundfile numpy

用法：
    # 钢琴
    python compare_pitch_detection.py piano.wav --instrument piano --reference piano_ref.json

    # 小提琴（Basic Pitch vs pyin 对比）
    python compare_pitch_detection.py violin.wav --instrument violin --reference violin_ref.json

    # 如果用户不是从 t=0 开始弹，加 --auto-offset 自动归零
    python compare_pitch_detection.py violin.wav --instrument violin --reference violin_ref.json --auto-offset
"""

import argparse
import json
import sys
import numpy as np
from pathlib import Path

try:
    import librosa
    import soundfile as sf
except ImportError:
    print("缺少依赖：pip install librosa soundfile numpy")
    sys.exit(1)

try:
    from basic_pitch.inference import predict
    from basic_pitch import ICASSP_2022_MODEL_PATH
    BASIC_PITCH_AVAILABLE = True
except ImportError:
    print("⚠️  Basic Pitch 未安装：pip install basic-pitch")
    BASIC_PITCH_AVAILABLE = False

NOTE_NAMES = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']

def midi_to_name(midi: int) -> str:
    return f"{NOTE_NAMES[midi % 12]}{(midi // 12) - 1}"

def freq_to_midi(freq: float) -> int:
    if freq <= 0: return 0
    return round(69 + 12 * np.log2(freq / 440.0))

def apply_offset(notes: list[dict], auto_offset: bool) -> list[dict]:
    """把检测结果的时间归零，对齐参考谱面。"""
    if not notes or not auto_offset:
        return notes
    offset = notes[0]['time_sec']
    if offset > 0.5:  # 只有偏移超过 0.5 秒才归零，避免误操作
        print(f"  时间归零：第一个音符在 t={offset}s，所有时间减去 {offset}s")
        for n in notes:
            n['time_sec'] = round(n['time_sec'] - offset, 3)
        notes = [n for n in notes if n['time_sec'] >= 0]
    return notes


# ── Basic Pitch 多音检测 ───────────────────────────────────────────────────────
def run_basic_pitch(audio_path: str, instrument: str,
                    auto_offset: bool = False) -> list[dict]:
    if not BASIC_PITCH_AVAILABLE:
        return []

    print("\n🎹 [Basic Pitch] 多音检测中...")

    if instrument == 'piano':
        fmin, fmax = 27.5, 4186.0
    else:
        fmin, fmax = 196.0, 2637.0

    _, _, note_events = predict(
        audio_path,
        ICASSP_2022_MODEL_PATH,
        onset_threshold=0.8,
        frame_threshold=0.6,
        minimum_note_length=0.15,
        minimum_frequency=fmin,
        maximum_frequency=fmax,
    )

    notes = sorted([{
        'time_sec':   round(float(n[0]), 3),
        'end_sec':    round(float(n[1]), 3),
        'pitch_midi': int(n[2]),
        'note_name':  midi_to_name(int(n[2])),
        'confidence': round(float(n[3]), 3),
    } for n in note_events], key=lambda x: x['time_sec'])

    notes = apply_offset(notes, auto_offset)
    print(f"   检测到 {len(notes)} 个音符")
    return notes


# ── librosa pyin 单音检测 ─────────────────────────────────────────────────────
def run_pyin(audio_path: str, instrument: str,
             auto_offset: bool = False) -> list[dict]:
    print("\n🎻 [librosa pyin] 单音检测中（模拟 pitch_detector_dart）...")

    y, sr = librosa.load(audio_path, sr=44100)

    if instrument == 'violin':
        fmin = librosa.note_to_hz('G3')
        fmax = librosa.note_to_hz('E7')
    else:
        fmin = librosa.note_to_hz('A0')
        fmax = librosa.note_to_hz('C8')

    f0, voiced_flag, _ = librosa.pyin(
        y, fmin=fmin, fmax=fmax, sr=sr,
        frame_length=2048, hop_length=512,
    )
    times = librosa.frames_to_time(np.arange(len(f0)), sr=sr, hop_length=512)

    notes = []
    in_note = False
    start_t = 0.0
    freqs = []

    for t, freq, voiced in zip(times, f0, voiced_flag):
        if voiced and freq and not np.isnan(freq) and 200 <= freq <= 1500:
            if not in_note:
                in_note = True
                start_t = t
                freqs = [freq]
            else:
                freqs.append(freq)
        else:
            if in_note and len(freqs) >= 2:
                avg = float(np.median(freqs))
                notes.append({
                    'time_sec':   round(start_t, 3),
                    'pitch_midi': freq_to_midi(avg),
                    'note_name':  midi_to_name(freq_to_midi(avg)),
                    'freq_hz':    round(avg, 1),
                })
            in_note = False
            freqs = []

    if in_note and len(freqs) >= 2:
        avg = float(np.median(freqs))
        notes.append({
            'time_sec':   round(start_t, 3),
            'pitch_midi': freq_to_midi(avg),
            'note_name':  midi_to_name(freq_to_midi(avg)),
            'freq_hz':    round(avg, 1),
        })

    notes = apply_offset(notes, auto_offset)
    print(f"   检测到 {len(notes)} 个音符")
    return notes


# ── 准确率评估 ────────────────────────────────────────────────────────────────
def evaluate(detected: list[dict], reference: list[dict],
             time_tolerance: float = 0.5,
             pitch_tolerance: int = 1) -> dict:
    if not reference:
        return {'error': '没有参考谱面'}

    used = set()
    hits = 0

    for ref in reference:
        best_idx = None
        best_diff = float('inf')
        for i, det in enumerate(detected):
            if i in used: continue
            time_diff = abs(det['time_sec'] - ref['time_sec'])
            pitch_diff = abs(det['pitch_midi'] - ref['pitch_midi'])
            if time_diff <= time_tolerance and pitch_diff <= pitch_tolerance:
                if time_diff < best_diff:
                    best_diff = time_diff
                    best_idx = i
        if best_idx is not None:
            used.add(best_idx)
            hits += 1

    miss = len(reference) - hits
    fp = len(detected) - hits
    precision = hits / (hits + fp) if (hits + fp) > 0 else 0
    recall = hits / (hits + miss) if (hits + miss) > 0 else 0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0)

    return {
        'total_reference': len(reference),
        'total_detected':  len(detected),
        'hit':             hits,
        'miss':            miss,
        'false_positive':  fp,
        'precision':       round(precision * 100, 1),
        'recall':          round(recall * 100, 1),
        'f1_score':        round(f1 * 100, 1),
    }


def print_report(label: str, stats: dict):
    print(f"\n{'─' * 50}")
    print(f"📊 {label}")
    print(f"{'─' * 50}")
    if 'error' in stats:
        print(f"   {stats['error']}")
        return
    print(f"  参考音符数:    {stats['total_reference']}")
    print(f"  检测音符数:    {stats['total_detected']}")
    print(f"  ✅ 正确匹配:   {stats['hit']}")
    print(f"  ❌ 漏检:       {stats['miss']}")
    print(f"  ⚠️  误检:       {stats['false_positive']}")
    print(f"  查准率:        {stats['precision']}%")
    print(f"  查全率:        {stats['recall']}%")
    print(f"  F1 综合得分:   {stats['f1_score']}%")


def print_conclusion(instrument: str, bp_stats: dict,
                     pyin_stats: dict | None):
    print(f"\n{'═' * 60}")
    print("🔍 结论")
    print(f"{'═' * 60}")

    if instrument == 'piano':
        f1 = bp_stats.get('f1_score', 0)
        if f1 >= 75:
            print(f"  钢琴：Basic Pitch F1={f1}% ✅ 准确率足够，推荐用 server-side Basic Pitch")
        elif f1 >= 55:
            print(f"  钢琴：Basic Pitch F1={f1}% ⚠️  中等，可用但需要继续调参")
        else:
            print(f"  钢琴：Basic Pitch F1={f1}% ❌ 准确率偏低，需要分析原因")
    else:
        bp_f1   = bp_stats.get('f1_score', 0) if bp_stats else 0
        pyin_f1 = pyin_stats.get('f1_score', 0) if pyin_stats else 0
        print(f"  小提琴：Basic Pitch F1={bp_f1}%  vs  pyin F1={pyin_f1}%")
        if bp_f1 >= pyin_f1 + 10:
            print("  → Basic Pitch 明显更准，可以统一用 Basic Pitch 处理两种乐器")
        elif pyin_f1 >= bp_f1 + 10:
            print("  → 本地方案更准，小提琴继续用 pitch_detector_dart，钢琴用 Basic Pitch")
        else:
            print("  → 两者相差不大，小提琴用本地方案节省服务器成本")

    print(f"{'═' * 60}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('audio', help='录音文件路径（.wav）')
    parser.add_argument('--instrument', choices=['piano', 'violin'],
                        default='piano')
    parser.add_argument('--reference', default=None, help='参考谱面 JSON')
    parser.add_argument('--auto-offset', action='store_true', default=False,
                        help='自动把第一个音符时间归零（用户不是从 t=0 开始弹时使用）')
    args = parser.parse_args()

    print(f"{'═' * 60}")
    print(f"🎵 文件：{args.audio}")
    print(f"🎸 乐器：{args.instrument}")
    if args.auto_offset:
        print(f"⏱  时间归零：开启")
    print(f"{'═' * 60}")

    reference = []
    if args.reference:
        with open(args.reference, encoding='utf-8') as f:
            reference = json.load(f)
        print(f"📄 参考谱面：{len(reference)} 个音符")

    bp_notes = run_basic_pitch(args.audio, args.instrument, args.auto_offset)
    bp_stats = evaluate(bp_notes, reference) if reference else {}

    pyin_notes = None
    pyin_stats = None
    if args.instrument == 'violin':
        pyin_notes = run_pyin(args.audio, args.instrument, args.auto_offset)
        pyin_stats = evaluate(pyin_notes, reference) if reference else {}

    if reference:
        print_report(f"Basic Pitch 准确率（{args.instrument}）", bp_stats)
        if pyin_stats:
            print_report("librosa pyin 准确率（小提琴）", pyin_stats)
        print_conclusion(args.instrument, bp_stats, pyin_stats)

    output = {
        'audio': args.audio,
        'instrument': args.instrument,
        'basic_pitch': {'notes': bp_notes, 'accuracy': bp_stats},
    }
    if pyin_notes is not None:
        output['pyin'] = {'notes': pyin_notes, 'accuracy': pyin_stats}

    out_path = Path(args.audio).stem + '_result.json'
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\n✅ 详细结果已保存：{out_path}")


if __name__ == '__main__':
    main()
