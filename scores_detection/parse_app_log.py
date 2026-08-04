#!/usr/bin/env python3
"""
parse_app_log.py

把 Flutter app 日志里的起始事件解析成 JSON，
然后和参考谱面对比，算出 pitch_detector_dart 的真实准确率。

用法：
    python parse_app_log.py app_log.txt --reference jingle_bells_violin_reference.json

    # 如果录音不是从 t=0 开始弹（比如 4.3 秒后才开始），加 --auto-offset 自动归零：
    python parse_app_log.py app_log.txt --reference jingle_bells_violin_reference.json --auto-offset
"""

import argparse
import json
import re
import sys
import numpy as np

NOTE_NAMES = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']

def freq_to_midi(freq: float) -> int:
    if freq <= 0: return 0
    return round(69 + 12 * np.log2(freq / 440.0))

def midi_to_name(midi: int) -> str:
    return f"{NOTE_NAMES[midi % 12]}{(midi // 12) - 1}"

def parse_log(log_path: str, auto_offset: bool = True,
              manual_offset: float = 0.0) -> list[dict]:
    """
    解析 Flutter 日志里的起始事件，格式：
    [起始事件] t=0.231s  freq=293.7Hz  ≈D4
    """
    notes = []
    pattern = re.compile(r't=(\d+\.?\d*)s\s+freq=(\d+\.?\d*)Hz')

    with open(log_path, encoding='utf-8') as f:
        for line in f:
            if '起始事件' not in line and 'onset' not in line.lower():
                continue
            m = pattern.search(line)
            if m:
                t = float(m.group(1))
                freq = float(m.group(2))
                midi = freq_to_midi(freq)
                notes.append({
                    'time_sec':   round(t, 3),
                    'pitch_midi': midi,
                    'note_name':  midi_to_name(midi),
                    'freq_hz':    round(freq, 1),
                })

    notes.sort(key=lambda n: n['time_sec'])

    # 时间归零：节拍器启动和录音启动之间有延迟，
    # 用户实际开始弹的时间不是 t=0，需要把时间偏移掉才能和参考谱面对齐。
    if auto_offset and notes:
        offset = notes[0]['time_sec']
        print(f"  自动时间归零：第一个起始事件在 t={offset}s，"
              f"所有时间减去 {offset}s")
        for n in notes:
            n['time_sec'] = round(n['time_sec'] - offset, 3)
    elif manual_offset != 0.0:
        print(f"  手动时间偏移：所有时间减去 {manual_offset}s")
        for n in notes:
            n['time_sec'] = round(n['time_sec'] - manual_offset, 3)
        # 过滤掉偏移后时间为负的事件
        notes = [n for n in notes if n['time_sec'] >= 0]

    return notes


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('log', help='Flutter 日志文件（.txt）')
    parser.add_argument('--reference', required=True, help='参考谱面 JSON')
    parser.add_argument('--auto-offset', action='store_true', default=True,
                        help='自动把第一个起始事件的时间归零（默认开启）')
    parser.add_argument('--no-auto-offset', dest='auto_offset',
                        action='store_false',
                        help='关闭自动归零')
    parser.add_argument('--time-offset', type=float, default=0.0,
                        help='手动指定时间偏移（秒），和 --no-auto-offset 一起用')
    args = parser.parse_args()

    print(f"📄 解析日志：{args.log}")
    notes = parse_log(args.log, args.auto_offset, args.time_offset)

    if not notes:
        print("❌ 没有解析到任何起始事件")
        print("   期望格式：[起始事件] t=0.231s  freq=293.7Hz  ≈D4")
        sys.exit(1)

    print(f"✅ 解析到 {len(notes)} 个起始事件")
    print("\n前 10 个（归零后）：")
    for n in notes[:10]:
        print(f"  t={n['time_sec']}s  {n['note_name']}  "
              f"MIDI={n['pitch_midi']}  freq={n['freq_hz']}Hz")

    with open(args.reference, encoding='utf-8') as f:
        reference = json.load(f)

    stats = evaluate(notes, reference)

    print(f"\n{'─' * 50}")
    print(f"📊 pitch_detector_dart 真实准确率")
    print(f"{'─' * 50}")
    print(f"  参考音符数:    {stats['total_reference']}")
    print(f"  检测音符数:    {stats['total_detected']}")
    print(f"  ✅ 正确匹配:   {stats['hit']}")
    print(f"  ❌ 漏检:       {stats['miss']}")
    print(f"  ⚠️  误检:       {stats['false_positive']}")
    print(f"  查准率:        {stats['precision']}%")
    print(f"  查全率:        {stats['recall']}%")
    print(f"  F1 综合得分:   {stats['f1_score']}%")

    out_path = args.log.replace('.txt', '_result.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump({'detected': notes, 'accuracy': stats},
                  f, ensure_ascii=False, indent=2)
    print(f"\n✅ 结果已保存：{out_path}")


if __name__ == '__main__':
    main()