"""
Module de calcul des métriques pour la simulation multi-agents.
Inclut les indicateurs descriptifs et les fonctions de validation
des hypothèses de recherche (H1 à H5).
"""

import numpy as np
from scipy import stats


# ============================================================
# MÉTRIQUES DE BASE (descriptives)
# ============================================================

def calculate_average_performance(students):
    """Performance moyenne du groupe = moyenne(competence * motivation)."""
    if not students:
        return 0
    return sum(s.competence * s.motivation for s in students) / len(students)


def calculate_collaboration(students):
    """
    Nombre TOTAL d'interactions entre étudiants (somme brute).
    À utiliser comme indicateur global du scénario.
    """
    return sum(s.interactions for s in students)


def calculate_collaboration_per_student(students):
    """
    Nombre MOYEN d'interactions par étudiant.
    Plus rigoureux pour comparer des scénarios avec un nombre
    différent d'agents.
    """
    if not students:
        return 0
    return np.mean([s.interactions for s in students])


def calculate_engagement(students):
    """Engagement moyen = moyenne du nombre d'actions par étudiant."""
    if not students:
        return 0
    return sum(s.actions for s in students) / len(students)


def calculate_std_performance(students):
    """Écart-type des performances individuelles."""
    if not students:
        return 0
    scores = [s.competence * s.motivation for s in students]
    return float(np.std(scores))


def calculate_completion_rate(groups):
    """Taux de complétion des tâches."""
    total = 0
    success = 0
    for g in groups:
        if g.task:
            total += 1
            if g.task.completed:
                success += 1
    return success / total if total > 0 else 0


def calculate_interventions(tutor):
    """Nombre d'interventions du tuteur."""
    return tutor.interventions


# ============================================================
# MÉTRIQUES D'ÉQUITÉ (CORRIGÉES)
# ============================================================

def calculate_equity(students):
    """
    Équité de la charge de travail mesurée par l'écart-type.
    INTERPRÉTATION : plus σ est faible, plus la répartition est équitable.
    
    NOTE : remplace l'ancienne version (max - min) qui était instable
    et sensible aux outliers.
    """
    if not students:
        return 0
    workloads = [s.charge_travail for s in students]
    return float(np.std(workloads))


def calculate_equity_index(students):
    """
    Indice d'équité normalisé entre 0 et 1.
    1 = parfaitement équitable | 0 = très inégal.
    
    Basé sur le coefficient de Gini inversé (1 - Gini).
    Utile pour H4 : permet une interprétation plus intuitive.
    """
    if not students:
        return 1.0
    workloads = np.array([s.charge_travail for s in students], dtype=float)
    n = len(workloads)
    total = workloads.sum()
    if total == 0 or n == 0:
        return 1.0  # personne n'a travaillé → équité triviale
    workloads_sorted = np.sort(workloads)
    index = np.arange(1, n + 1)
    gini = (2 * np.sum(index * workloads_sorted)) / (n * total) - (n + 1) / n
    return float(max(0.0, min(1.0, 1 - gini)))


# ============================================================
# FONCTIONS POUR LA VALIDATION STATISTIQUE DES HYPOTHÈSES
# ============================================================

def correlation_interactions_performance(students):
    """
    [Pour H1] Corrélation de Pearson entre interactions et performance
    au sein d'un groupe d'étudiants.
    
    H1 est validée si r > 0 et p < 0.05.
    
    Returns:
        tuple: (r, p_value) — r dans [-1, 1], p dans [0, 1]
    """
    if len(students) < 3:
        return 0.0, 1.0
    interactions = np.array([s.interactions for s in students], dtype=float)
    performances = np.array([s.competence * s.motivation for s in students], dtype=float)
    
    if np.std(interactions) == 0 or np.std(performances) == 0:
        return 0.0, 1.0
    
    r, p = stats.pearsonr(interactions, performances)
    return float(r), float(p)


def correlation_equity_engagement(equity_values, engagement_values):
    """
    [Pour H4] Corrélation entre équité (indice ∈ [0,1]) et engagement
    sur plusieurs runs.
    
    H4 est validée si r > 0 et p < 0.05.
    
    Args:
        equity_values: liste/array d'indices d'équité (un par run)
        engagement_values: liste/array d'engagements (un par run)
    
    Returns:
        tuple: (r, p_value)
    """
    if len(equity_values) < 3 or len(equity_values) != len(engagement_values):
        return 0.0, 1.0
    eq = np.array(equity_values, dtype=float)
    en = np.array(engagement_values, dtype=float)
    if np.std(eq) == 0 or np.std(en) == 0:
        return 0.0, 1.0
    r, p = stats.pearsonr(eq, en)
    return float(r), float(p)


def confidence_interval(data, confidence=0.95):
    """
    Intervalle de confiance à 95% (par défaut) avec distribution t.
    Plus rigoureux que l'écart-type pour une présentation scientifique.
    
    Returns:
        tuple: (mean, half_width) — l'IC est [mean - hw, mean + hw]
    """
    data = np.array(data, dtype=float)
    n = len(data)
    if n < 2:
        return float(np.mean(data)) if n == 1 else 0.0, 0.0
    mean = np.mean(data)
    sem = stats.sem(data)
    hw = sem * stats.t.ppf((1 + confidence) / 2, n - 1)
    return float(mean), float(hw)


def cohens_d(group_a, group_b):
    """
    Taille d'effet de Cohen (d) entre deux groupes.
    
    Interprétation :
        |d| < 0.2  → effet négligeable
        0.2 - 0.5  → petit effet
        0.5 - 0.8  → effet moyen
        |d| > 0.8  → grand effet
    
    Returns:
        float: la valeur de d
    """
    a = np.array(group_a, dtype=float)
    b = np.array(group_b, dtype=float)
    if len(a) < 2 or len(b) < 2:
        return 0.0
    var_a = np.var(a, ddof=1)
    var_b = np.var(b, ddof=1)
    pooled = np.sqrt(((len(a) - 1) * var_a + (len(b) - 1) * var_b) / (len(a) + len(b) - 2))
    if pooled == 0:
        return 0.0
    return float((np.mean(a) - np.mean(b)) / pooled)


def welch_t_test(group_a, group_b):
    """
    Test t de Welch (ne suppose pas l'égalité des variances).
    Plus robuste que le t-test classique pour comparer deux scénarios.
    
    Returns:
        tuple: (t_stat, p_value)
    """
    if len(group_a) < 2 or len(group_b) < 2:
        return 0.0, 1.0
    t, p = stats.ttest_ind(group_a, group_b, equal_var=False)
    return float(t), float(p)


def anova_test(*groups):
    """
    ANOVA à un facteur pour comparer plusieurs scénarios.
    Test global : y a-t-il au moins une différence significative ?
    
    Returns:
        tuple: (F_stat, p_value)
    """
    valid = [g for g in groups if len(g) >= 2]
    if len(valid) < 2:
        return 0.0, 1.0
    f, p = stats.f_oneway(*valid)
    return float(f), float(p)