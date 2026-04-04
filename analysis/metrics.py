import numpy as np


def calculate_average_performance(students):
    return sum(s.competence * s.motivation for s in students) / len(students)


def calculate_collaboration(students):
    return sum(s.interactions for s in students)


def calculate_engagement(students):
    return sum(s.actions for s in students) / len(students)


def calculate_equity(students):
    workloads = [s.charge_travail for s in students]
    if not workloads:
        return 0
    return max(workloads) - min(workloads)


def calculate_std_performance(students):
    scores = [s.competence * s.motivation for s in students]
    return np.std(scores)

def calculate_completion_rate(groups):
    total = 0
    success = 0

    for g in groups:
        if g.task:
            total += 1
            if g.task.completed:
                success += 1

    return success / total if total > 0 else 0
def calculate_interventions(tutor):
    return tutor.interventions