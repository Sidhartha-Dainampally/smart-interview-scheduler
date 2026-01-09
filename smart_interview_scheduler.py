"""
Smart Interview Scheduler

"""

from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional
import heapq
import csv
import argparse
import random
import sys


@dataclass(order=True)
class HeapInterviewer:
    available_time: int
    interviewer_id: int = field(compare=False)


@dataclass
class Candidate:
    cid: str
    arrival: int
    duration: int


@dataclass
class Assignment:
    candidate_id: str
    interviewer_id: int
    start_time: int
    end_time: int
    waiting_time: int


class Scheduler:
    """Scheduler provides different strategies for assigning candidates to interviewers.

    Methods
    -------
    schedule_greedy(candidates, num_interviewers)
        Use a min-heap keyed by next-available time to always assign the candidate to the earliest available interviewer.

    schedule_naive(candidates, num_interviewers)
        Assign candidates sequentially to interviewers in round-robin order (tracks each interviewer's last end time but does not globally select the earliest).
    """

    @staticmethod
    def schedule_greedy(candidates: List[Candidate], num_interviewers: int) -> Tuple[List[Assignment], Dict[int, int]]:
        """Greedy scheduling using a min-heap of (available_time, interviewer_id).

        Returns:
            assignments: List[Assignment]
            idle_times: Dict[interviewer_id -> total idle time]
        """
        if num_interviewers <= 0:
            raise ValueError("Number of interviewers must be >= 1")

        # Sort candidates by arrival time (stable)
        candidates_sorted = sorted(candidates, key=lambda c: c.arrival)

        # Initialize heap of interviewers: all available at time 0
        heap: List[HeapInterviewer] = [HeapInterviewer(0, i) for i in range(num_interviewers)]
        heapq.heapify(heap)

        # Track last finish time per interviewer (for idle time calc)
        last_finish: Dict[int, int] = {i: 0 for i in range(num_interviewers)}
        idle_times: Dict[int, int] = {i: 0 for i in range(num_interviewers)}

        assignments: List[Assignment] = []

        for c in candidates_sorted:
            # pop the earliest available interviewer
            interviewer = heapq.heappop(heap)
            start_time = max(c.arrival, interviewer.available_time)
            waiting = start_time - c.arrival
            end_time = start_time + c.duration

            # accumulate idle time for this interviewer
            if start_time > last_finish[interviewer.interviewer_id]:
                idle_times[interviewer.interviewer_id] += start_time - last_finish[interviewer.interviewer_id]

            last_finish[interviewer.interviewer_id] = end_time

            # record assignment
            assignments.append(Assignment(c.cid, interviewer.interviewer_id, start_time, end_time, waiting))

            # push interviewer back with updated available_time
            heapq.heappush(heap, HeapInterviewer(end_time, interviewer.interviewer_id))

        return assignments, idle_times

    @staticmethod
    def schedule_naive(candidates: List[Candidate], num_interviewers: int) -> Tuple[List[Assignment], Dict[int, int]]:
        """Naive round-robin scheduler. Keeps a list of last finish times per interviewer and assigns next candidate to (i % m).

        This intentionally does not pick the globally earliest interviewer, so it's typically worse than the greedy method.
        """
        if num_interviewers <= 0:
            raise ValueError("Number of interviewers must be >= 1")

        candidates_sorted = sorted(candidates, key=lambda c: c.arrival)

        last_finish = [0] * num_interviewers
        idle_times = {i: 0 for i in range(num_interviewers)}
        assignments: List[Assignment] = []

        idx = 0
        for c in candidates_sorted:
            i = idx % num_interviewers
            start_time = max(c.arrival, last_finish[i])
            waiting = start_time - c.arrival
            end_time = start_time + c.duration

            if start_time > last_finish[i]:
                idle_times[i] += start_time - last_finish[i]

            last_finish[i] = end_time
            assignments.append(Assignment(c.cid, i, start_time, end_time, waiting))
            idx += 1

        return assignments, idle_times


# Utility functions for metrics and pretty printing

def compute_metrics(assignments: List[Assignment], idle_times: Dict[int, int], candidates: List[Candidate]) -> Dict[str, float]:
    total_wait = sum(a.waiting_time for a in assignments)
    avg_wait = total_wait / len(assignments) if assignments else 0
    makespan = max((a.end_time for a in assignments), default=0) - min((c.arrival for c in candidates), default=0)
    total_idle = sum(idle_times.values())
    return {
        "total_waiting_time": total_wait,
        "average_waiting_time": avg_wait,
        "makespan_total_time": makespan,
        "total_idle_time": total_idle,
    }


def print_schedule(assignments: List[Assignment]):
    print("Assignments:")
    print(f"{'Candidate':10} {'Interviewer':10} {'Start':>6} {'End':>6} {'Wait':>6}")
    for a in assignments:
        print(f"{a.candidate_id:10} {a.interviewer_id:^10} {a.start_time:6} {a.end_time:6} {a.waiting_time:6}")


# Demo functions

def generate_random_candidates(n: int, max_arrival: int = 120, max_duration: int = 60, seed: Optional[int] = 1) -> List[Candidate]:
    random.seed(seed)
    candidates = []
    for i in range(1, n + 1):
        arrival = random.randint(0, max_arrival)
        duration = random.randint(5, max_duration)
        candidates.append(Candidate(f"C{i}", arrival, duration))
    return candidates


def read_candidates_from_csv(path: str) -> List[Candidate]:
    candidates = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or len(row) < 3:
                continue
            cid = row[0].strip()
            arrival = int(row[1])
            duration = int(row[2])
            candidates.append(Candidate(cid, arrival, duration))
    return candidates


def run_demo(num_candidates: int = 12, num_interviewers: int = 3):
    print("\nSmart Interview Scheduler — Demo\n")
    candidates = generate_random_candidates(num_candidates, max_arrival=40, max_duration=25, seed=42)
    print("Candidates (unsorted):")
    for c in candidates:
        print(f"{c.cid:3} arrival={c.arrival:3} dur={c.duration:3}")

    # Greedy
    greedy_assignments, greedy_idle = Scheduler.schedule_greedy(candidates, num_interviewers)
    print("\n--- Greedy Scheduling ---")
    print_schedule(greedy_assignments)
    greedy_metrics = compute_metrics(greedy_assignments, greedy_idle, candidates)
    print("Metrics:", greedy_metrics)

    # Naive
    naive_assignments, naive_idle = Scheduler.schedule_naive(candidates, num_interviewers)
    print("\n--- Naive Scheduling (round-robin) ---")
    print_schedule(naive_assignments)
    naive_metrics = compute_metrics(naive_assignments, naive_idle, candidates)
    print("Metrics:", naive_metrics)

    # Compare
    print("\n--- Comparison ---")
    def pct_improve(a, b):
        # improvement from b -> a (positive means improvement)
        if b == 0:
            return float('inf') if a != 0 else 0
        return 100 * (b - a) / b

    for key in ["total_waiting_time", "average_waiting_time", "makespan_total_time", "total_idle_time"]:
        g = greedy_metrics[key]
        n = naive_metrics[key]
        print(f"{key}: greedy={g:.2f}  naive={n:.2f}  improvement={pct_improve(g, n):.2f}%")


# Basic CLI
def main(argv=None):
    parser = argparse.ArgumentParser(description="Smart Interview Scheduler — Demo and CLI")
    parser.add_argument("--csv", help="CSV file with candidates: id,arrival,duration", default=None)
    parser.add_argument("--interviewers", type=int, default=3, help="Number of interviewers")
    parser.add_argument("--demo-candidates", type=int, default=12, help="Number of random demo candidates (if --csv not given)")

    args = parser.parse_args(argv)

    if args.csv:
        candidates = read_candidates_from_csv(args.csv)
        if not candidates:
            print("No candidates read from CSV. Exiting.")
            return
        greedy_assignments, greedy_idle = Scheduler.schedule_greedy(candidates, args.interviewers)
        print("\nGreedy scheduling results:")
        print_schedule(greedy_assignments)
        print("Metrics:", compute_metrics(greedy_assignments, greedy_idle, candidates))
    else:
        run_demo(num_candidates=args.demo_candidates, num_interviewers=args.interviewers)


if __name__ == "__main__":
    main()
