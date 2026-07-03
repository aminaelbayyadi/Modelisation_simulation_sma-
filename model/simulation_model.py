from mesa import Model
from mesa.time import RandomActivation

from agents.student_agent import StudentAgent
from agents.tutor_agent import TutorAgent
from agents.teacher_agent import TeacherAgent
from agents.system_agent import SystemAgent

from ml.kmeans_model import cluster_students

from analysis.metrics import (
    calculate_average_performance,
    calculate_collaboration,
    calculate_engagement,
    calculate_equity,
    calculate_std_performance,
)


class SimulationModel(Model):

    def __init__(self, num_students, use_tutor=True,
                 smart_grouping=False, auto_learning=False,
                 verbose=False, seed=None, peer_help=True, help_scope="classe",
                 ml_model=None):
        super().__init__()

        if seed is not None:
            import random as _random
            self.random = _random.Random(seed)

        self.num_students = num_students
        self.use_tutor = use_tutor
        self.smart_grouping = smart_grouping
        self.auto_learning = auto_learning
        self.verbose = verbose

        # Random Forest pre-entraine, utilise par le tuteur pour predire le risque.
        # S'il vaut None, le tuteur se rabat sur une heuristique.
        self.ml_model = ml_model

        # Portee de l'entraide : "classe" (toute la classe) ou "groupe" (coequipiers)
        self.help_scope = help_scope

        # H1 : drapeau d'entraide entre pairs.
        #   peer_help=True  -> scenario COLLABORATIF (les agents s'entraident)
        #   peer_help=False -> scenario INDIVIDUEL  (travail sans entraide)
        self.peer_help = peer_help

        self.schedule = RandomActivation(self)
        self.students = []

        self.performance_history = []
        self.collaboration_history = []
        self.engagement_history = []
        self.equity_history = []
        self.std_history = []

        for i in range(self.num_students):
            student = StudentAgent(i, self)
            self.schedule.add(student)
            self.students.append(student)

        if self.use_tutor:
            self.tutor = TutorAgent("Tutor", self)
            self.schedule.add(self.tutor)
        else:
            self.tutor = None

        self.teacher = TeacherAgent("Teacher", self)
        self.schedule.add(self.teacher)

        self.system = SystemAgent("System", self)
        self.schedule.add(self.system)

    def step(self):
        # 1) DEBUT DE TOUR : former les groupes, attribuer UNE tache commune
        #    par groupe, et donner a chaque etudiant ses coequipiers + cette tache.
        self.system.form_groups()
        self.system.assign_tasks()
        for g in self.system.groups:
            for s in g.members:
                s.groupmates = [m for m in g.members if m is not s]
                s.current_task = g.task   # meme tache pour tout le groupe

        # 2) Les agents agissent (etudiants travaillent ensemble + s'entraident
        #    dans leur groupe ; tuteur et enseignant interviennent).
        self.schedule.step()

        if self.auto_learning:
            try:
                labels = cluster_students(self.students)
            except Exception:
                labels = [0] * len(self.students)
            for student, label in zip(self.students, labels):
                if label == 0:
                    student.motivation += 0.030
                    student.competence += 0.010
                elif label == 1:
                    student.motivation += 0.015
                    student.competence += 0.005
                else:
                    student.motivation += 0.005
                    student.competence += 0.002
                student.motivation -= 0.010
                student.motivation = max(StudentAgent.MOTIVATION_MIN,
                                         min(student.motivation, StudentAgent.MOTIVATION_MAX))
                student.competence = max(StudentAgent.COMPETENCE_MIN,
                                         min(student.competence, StudentAgent.COMPETENCE_MAX))

        avg_perf = calculate_average_performance(self.students)
        collab = calculate_collaboration(self.students)
        engage = calculate_engagement(self.students)
        equity = calculate_equity(self.students)
        std_perf = calculate_std_performance(self.students)

        self.performance_history.append(avg_perf)
        self.collaboration_history.append(collab)
        self.engagement_history.append(engage)
        self.equity_history.append(equity)
        self.std_history.append(std_perf)

        if self.verbose:
            print(f"Perf: {avg_perf:.3f} | Collab: {collab} | "
                  f"Engage: {engage:.2f} | Equity(sigma): {equity:.3f}")