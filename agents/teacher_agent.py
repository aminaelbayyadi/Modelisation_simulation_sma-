from mesa import Agent

class TeacherAgent(Agent):
    
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        
        # consignes pédagogiques
        self.instructions = "Collaborer pour résoudre les tâches"

    def give_instructions(self):
        """
        L'enseignant donne les consignes aux étudiants
        """
        print("Consigne :", self.instructions)

    def evaluate_students(self):
        """
        L'enseignant évalue la performance des étudiants
        """
        results = []

        for student in self.model.students:
            performance = student.competence * student.motivation
            results.append(performance)

        return results

    def step(self):
        """
        Action de l'enseignant dans la simulation
        """
        self.give_instructions()
        results = self.evaluate_students()
        print("Performances :", results)    