from django.db import models

# Create your models here.
class chatbot(models.Model):
    userInput = models.TextField()
    intentClassified = models.TextField()
    response = models.TextField()

    def __str__(self):
        return self.userInput+ ' - '+ self.intentClassified

