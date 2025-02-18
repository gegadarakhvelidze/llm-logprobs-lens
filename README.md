## LLM uncertainty visualization (CLI)
This is a simple CLI tool to visualize the logprobs and uncertainty in the LLM responses.

![Example](./assets/Token%20probabilities%20example.png)

### Installation
Create a virtual environment and install the requirements:
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` file with the following content:
```bash
OPENAI_API_KEY="<your-openai-api-key>"
OPENAI_BASE_URL="https://api.openai.com/v1"
OPENAI_MODEL_NAME="<model-name>"
```

### Usage
```bash
python app.py
```
