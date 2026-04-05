from mesa import Model
from mesa.time import RandomActivation
from analysis.metrics import calculate_std_performance #
from agents.student_agent import StudentAgent
from agents.tutor_agent import TutorAgent
from agents.teacher_agent import TeacherAgent
from agents.system_agent import SystemAgent
from analysis.metrics import (
    calculate_average_performance,
    calculate_collaboration,
    calculate_engagement,
    calculate_equity
)

class SimulationModel(Model):

    def __init__(self, num_students, use_tutor=True, smart_grouping=False):
        self.num_students = num_students
        self.use_tutor = use_tutor
        self.smart_grouping = smart_grouping

        self.schedule = RandomActivation(self)

        self.students = []

        # historiques
        self.performance_history = []
        self.collaboration_history = []
        self.engagement_history = []
        self.equity_history = []
        self.std_history = []
        # création agents
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

    def step(self):
        self.schedule.step()

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