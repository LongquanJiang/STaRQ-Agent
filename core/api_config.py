import os
# set your OPENAI_API_BASE, OPENAI_API_KEY here!
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

import openai
openai.api_key = OPENAI_API_KEY

MODEL_NAME = 'gpt-4.5-preview'