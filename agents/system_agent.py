from mesa import Agent
import random
from environment.groups import Group
from environment.tasks import Task


class SystemAgent(Agent):

    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)

        self.groups = []

    def form_groups(self):
        students = self.model.students.copy()

        self.groups = []
        group_size = 3

        if self.model.smart_grouping:
            #  GROUPES INTELLIGENTS

            sorted_students = sorted(students, key=lambda s: s.competence)

            while len(sorted_students) >= group_size:
                group_members = [
                    sorted_students.pop(0),  # faible
                    sorted_students.pop(len(sorted_students)//2),  # moyen
                    sorted_students.pop(-1)  # fort
                ]

                group = Group(group_members)
                self.groups.append(group)

        else:
            # groupes aléatoires
            random.shuffle(students)

            for i in range(0, len(students), group_size):
                group_members = students[i:i + group_size]
                group = Group(group_members)
                self.groups.append(group)

    def assign_tasks(self):
        difficulties = ["facile", "moyenne", "difficile"]

        for group in self.groups:
            difficulty = random.choice(difficulties)
            task = Task(difficulty)
            group.assign_task(task)

    def execute_groups(self):
        for group in self.groups:
            results = group.work()
            # print(results)

    def calculate_metrics(self):
        performances = []

        for student in self.model.students:
            perf = student.competence * student.motivation
            performances.append(perf)

        if performances:
            avg = sum(performances) / len(performances)
            print("Performance moyenne :", avg)

    def step(self):
        self.form_groups()
        self.assign_tasks()
        self.execute_groups()
        self.calculate_metrics()