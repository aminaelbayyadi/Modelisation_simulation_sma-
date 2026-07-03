from mesa import Agent


class StudentAgent(Agent):

    MOTIVATION_MIN = 0.1
    MOTIVATION_MAX = 1.0
    COMPETENCE_MIN = 0.0
    COMPETENCE_MAX = 1.0
    CHARGE_MAX = 5.0

    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.competence = self.random.uniform(0.3, 1.0)
        self.motivation = self.random.uniform(0.4, 1.0)
        self.charge_travail = 0
        self.satisfaction = 0.5
        self.interactions = 0
        self.actions = 0
        self.current_task = None
        self.groupmates = []   # coequipiers (definis a chaque tour par le modele)

    def work(self):
        self.actions += 1
        self.charge_travail += 0.3
        performance = self.competence * self.motivation
        if self.current_task is not None:
            self.current_task.attempt(self)
        return performance

    def request_help(self):
        return self.competence < 0.5

    def help(self, other_student):
        
        self.actions += 1
        self.interactions += 1
        other_student.interactions += 1

        if self.competence > other_student.competence:
            # Benefice de l'aide pour l'aide
            other_student.competence = min(
                other_student.competence + 0.03, self.COMPETENCE_MAX
            )
            # Effet protege : benefice (plus faible) pour l'aidant
            if getattr(self.model, "protege_effect", True):
                self.competence = min(self.competence + 0.01, self.COMPETENCE_MAX)
                self.motivation = min(self.motivation + 0.01, self.MOTIVATION_MAX)

    def update_state(self, success):
        if success:
            self.motivation += 0.04
            self.satisfaction += 0.04
            self.charge_travail *= 0.9
        else:
            self.motivation -= 0.02
            self.satisfaction -= 0.01
            self.charge_travail *= 1.05

        self.motivation += 0.005
        self.competence += 0.002

        self.charge_travail = max(0, min(self.charge_travail, self.CHARGE_MAX))
        self.motivation = max(self.MOTIVATION_MIN, min(self.motivation, self.MOTIVATION_MAX))
        self.competence = max(self.COMPETENCE_MIN, min(self.competence, self.COMPETENCE_MAX))
        self.satisfaction = max(0, min(self.satisfaction, 1))

    def step(self):
        performance = self.work()

        # PORTEE DE L'ENTRAIDE selon le scenario :
        #   help_scope="classe" -> aide de n'importe quel etudiant de la classe (S1, S2, S4)
        #   help_scope="groupe" -> aide uniquement d'un coequipier du groupe (S3)
        if getattr(self.model, "peer_help", True) and self.request_help():
            scope = getattr(self.model, "help_scope", "classe")
            if scope == "groupe":
                pool = [s for s in self.groupmates if s.competence > self.competence]
            else:
                pool = [s for s in self.model.students
                        if s.competence > self.competence and s is not self]
            if pool:
                helper = self.random.choice(pool)
                helper.help(self)

        success = self.random.random() < performance
        self.update_state(success)