# 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

import os
import openai
api_key = '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'
openai.api_key = api_key

response = openai.Completion.create(
  model="text-davinci-003",
  prompt="Classify the sentiment in these tweets:\n\n1. \"I can't stand homework\"\n2. \"This sucks. I'm bored  \"\n3. \"I can't wait for Halloween!!!\"\n4. \"My cat is adorable \"\n5. \"I hate chocolate\"\n\nTweet sentiment ratings:",
  temperature=0,
  max_tokens=60,
  top_p=1.0,
  frequency_penalty=0.0,
  presence_penalty=0.0
)