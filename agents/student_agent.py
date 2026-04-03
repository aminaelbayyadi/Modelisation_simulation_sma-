from mesa import Agent
import random

class StudentAgent(Agent):
    
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        
        # États internes
        self.competence = random.uniform(0.3, 1.0)
        self.motivation = random.uniform(0.4, 1.0)
        self.charge_travail = 0
        self.satisfaction = 0.5
        self.interactions = 0
        self.actions = 0

    def work(self):
        """
        L'étudiant travaille
        """
        self.actions += 1
        
        performance = self.competence * self.motivation
        return performance
    
    def request_help(self):
        """
        Demande d'aide si faible
        """
        return self.competence < 0.5
    
    def help(self, other_student):
        """
        Aider un autre étudiant
        """
        self.actions += 1
        self.interactions += 1
        
        if self.competence > other_student.competence:
            other_student.competence += 0.05
            
            #  Limiter
            other_student.competence = min(other_student.competence, 1.0)
    
    def update_state(self, success):
        """
        Mise à jour de l'état
        """
        if success:
            self.motivation += 0.05
            self.satisfaction += 0.05
        else:
            self.motivation -= 0.05
        
        #  NORMALISATION 
        self.motivation = max(0, min(self.motivation, 1))
        self.competence = max(0, min(self.competence, 1))
        self.satisfaction = max(0, min(self.satisfaction, 1))
    
    def step(self):
        """
        Étape de simulation
        """
        performance = self.work()
        
        # demande aide
        if self.request_help():
            other = random.choice(self.model.students)
            if other != self:
                other.help(self)
        
        # succès
        success = performance > 0.5
        
        # mise à jour
        self.update_state(success)