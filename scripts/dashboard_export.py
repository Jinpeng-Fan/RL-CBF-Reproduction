"""Export only defense metrics from CSV; watch mode follows complete appended rows.

Run with the managed TensorBoard interpreter. Original CSV and event logs are read-only.
"""
import argparse
import csv
import io
import json
import math
import time
from pathlib import Path

from tensorboard.compat.proto import event_pb2, summary_pb2
from tensorboard.summary.writer.event_file_writer import EventFileWriter


class Run:
    def __init__(self, entry):
        self.entry = entry
        self.writer = EventFileWriter(entry['out'], flush_secs=1)
        self.rows = 0
        self.last_step = 0
        self.unsafe = self.collisions = 0

    def update(self):
        path = Path(self.entry['csv'])
        if not path.exists():
            return
        # A writer can be midway through its final row. Retry that row next poll.
        content = path.read_text(encoding='utf-8-sig')
        content = content[:content.rfind('\n') + 1]
        rows = list(csv.DictReader(io.StringIO(content)))
        if len(rows) < self.rows:
            raise RuntimeError('Source CSV was truncated: ' + str(path))
        for row in rows[self.rows:]:
            step = int(row['episode'])
            if step <= self.last_step:
                raise ValueError('Episodes must increase: ' + str(path))
            if self.entry['task'] == 'pendulum':
                angle = float(row['max_angle'])
                self.unsafe += int(angle > 1 + 1e-6)
                values = {'reward/episode_return': float(row['reward']),
                          'safety/max_abs_angle_rad': angle,
                          'safety/unsafe_episodes_cumulative': self.unsafe}
                correction = row.get('correction_rms', '')
            else:
                gap = float(row['min_headway'])
                self.collisions += int(gap <= 0)
                self.unsafe += int(gap < 2)
                values = {'reward/episode_return': float(row['return']),
                          'safety/min_headway_m': gap,
                          'safety/collision_episodes_cumulative': self.collisions,
                          'safety/unsafe_episodes_cumulative': self.unsafe}
                correction = row.get('cbf_rms', '')
            if self.entry['name'].endswith('-CBF') and correction != '':
                values['cbf/correction_rms'] = float(correction)
            if not all(math.isfinite(value) for value in values.values()):
                raise ValueError('Nonfinite metric in ' + str(path))
            summary = summary_pb2.Summary(value=[
                summary_pb2.Summary.Value(tag=tag, simple_value=value)
                for tag, value in values.items()])
            self.writer.add_event(event_pb2.Event(wall_time=time.time(), step=step, summary=summary))
            self.rows += 1
            self.last_step = step
        self.writer.flush()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--manifest', type=Path, required=True)
    ap.add_argument('--watch', action='store_true')
    ap.add_argument('--ready', type=Path, required=True)
    args = ap.parse_args()
    runs = [Run(entry) for entry in json.loads(args.manifest.read_text(encoding='utf-8'))]
    try:
        while True:
            for run in runs:
                run.update()
            if not args.ready.exists():
                args.ready.write_text('ready', encoding='ascii')
            if not args.watch:
                break
            time.sleep(2)
    finally:
        for run in runs:
            run.writer.close()


if __name__ == '__main__':
    main()
