from rest_framework import serializers
from .models import chatbot

class chatbotSerializer(serializers.ModelSerializer):
    class Meta:
        model = chatbot
        fields = ['id', 'userInput', 'intentClassified', 'response']

# class botSerializer(serializers.Serializer):
#     userInput = serializers.TextField()
#     intentClassified = serializers.TextField()
#     response = serializers.TextField()

#     def create(self, validated_data):
#         return chatbot.objects.create(validated_data)

