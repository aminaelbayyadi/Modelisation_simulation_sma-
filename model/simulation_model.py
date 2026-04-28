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
    calculate_std_performance
)


class SimulationModel(Model):

    def __init__(self, num_students, use_tutor=True, smart_grouping=False, auto_learning=False):
        super().__init__()

        self.random.seed(42)

        self.num_students = num_students
        self.use_tutor = use_tutor
        self.smart_grouping = smart_grouping
        self.auto_learning = auto_learning

        #  scheduler réaliste
        self.schedule = RandomActivation(self)

        self.students = []

        # historiques
        self.performance_history = []
        self.collaboration_history = []
        self.engagement_history = []
        self.equity_history = []
        self.std_history = []

        # ==============================
        # CRÉATION AGENTS
        # ==============================
        for i in range(self.num_students):
            student = StudentAgent(i, self)
            self.schedule.add(student)
            self.students.append(student)

        if self.use_tutor:
            self.tutor = TutorAgent("Tutor", self)
            self.schedule.add(self.tutor)

        self.teacher = TeacherAgent("Teacher", self)
        self.schedule.add(self.teacher)

        self.system = SystemAgent("System", self)
        self.schedule.add(self.system)

    # ==============================
    # STEP GLOBAL
    # ==============================
    def step(self):

        # 1. exécution agents
        self.schedule.step()

        # ==============================
        # 🔴 S2 : BOOST TUTEUR 
        # ==============================
        if self.use_tutor and not self.auto_learning and not self.smart_grouping:           
            for student in self.students:
               student.competence += 0.01
               student.motivation += 0.005

            # normalisation
               student.competence = min(student.competence, 1.0)
               student.motivation = min(student.motivation, 1.0)
        # ==============================
        # 🔴 S4 : AUTO-APPRENTISSAGE
        # ==============================
        if self.auto_learning:

            labels = cluster_students(self.students)

            for student, label in zip(self.students, labels):

                if label == 0:
                  student.motivation += 0.03
                  student.competence += 0.01

                elif label == 1:
                    student.motivation += 0.015
                    student.competence += 0.005
                else:
                    student.motivation += 0.005
                    student.competence += 0.002

                # fatigue
                student.motivation -= 0.01

                # normalisation
                student.motivation = max(0, min(student.motivation, 1.0))
                student.competence = max(0, min(student.competence, 1.0))

        # ==============================
        # CALCUL METRICS
        # ==============================
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

        print("Perf:", avg_perf, "| Collab:", collab, "| Engage:", engage)

        if self.use_tutor:
            print("Interventions tuteur :", self.tutor.interventions)
            print("Std performance :", std_perf)