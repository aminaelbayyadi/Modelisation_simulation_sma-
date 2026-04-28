from mesa import Agent

class StudentAgent(Agent):
    
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        
        # États internes
        self.competence = self.random.uniform(0.3, 1.0)
        self.motivation = self.random.uniform(0.4, 1.0)
        self.charge_travail = 0
        self.satisfaction = 0.5
        self.interactions = 0
        self.actions = 0

    # ==============================
    # TRAVAIL
    # ==============================
    def work(self):
        self.actions += 1
        
        #  charge contrôlée (important)
        self.charge_travail += 0.3

        performance = self.competence * self.motivation
        return performance

    # ==============================
    # DEMANDE D’AIDE
    # ==============================
    def request_help(self):
        return self.competence < 0.5

    # ==============================
    # AIDE ENTRE ÉTUDIANTS
    # ==============================
    def help(self, other_student):
        self.actions += 1
        self.interactions += 1
        
        if self.competence > other_student.competence:
            other_student.competence += 0.03
            other_student.competence = min(other_student.competence, 1.0)

    # ==============================
    # MISE À JOUR
    # ==============================
    def update_state(self, success):

        if success:
            self.motivation += 0.04
            self.satisfaction += 0.04

            #  réduit la charge si réussite
            self.charge_travail *= 0.9

        else:
            self.motivation -= 0.02

            # légère augmentation charge
            self.charge_travail *= 1.05

        #  limites importantes
        self.charge_travail = max(0, min(self.charge_travail, 5))
        self.motivation = max(0, min(self.motivation, 1))
        self.competence = max(0, min(self.competence, 1))
        self.satisfaction = max(0, min(self.satisfaction, 1))

    # ==============================
    # STEP
    # ==============================
    def step(self):

        performance = self.work()

        #  demande d’aide intelligente
        if self.request_help():

            better_students = [
                s for s in self.model.students
                if s.competence > self.competence and s != self
            ]

            if better_students:
                helper = self.random.choice(better_students)
                helper.help(self)

        success = performance > 0.5

        self.update_state(success)