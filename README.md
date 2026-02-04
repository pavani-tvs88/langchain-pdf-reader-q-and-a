# langchain-pdf-reader-q-and-a
PDF Document reader Q&A bot using Gradio and OpenAI

## Configuration 🔧

This project requires an **OpenAI API key** set in the `OPENAI_API_KEY` environment variable.

Options to set it:

- **Codespaces secret (recommended)**: use the GitHub CLI to set a secret in your Codespace:

  ```bash
  gh codespace secret set -n OPENAI_API_KEY -b <your_key>
  ```

- **Local .env (development)**: create a `.env` file at the project root with:

  ```bash
  OPENAI_API_KEY=your_key
  ```

- **Shell export**:

  ```bash
  export OPENAI_API_KEY=your_key
  ```

Do not commit API keys to source control. To verify the variable is available in your environment:

```bash
echo $OPENAI_API_KEY
# or in python
python -c "import os; print(bool(os.getenv('OPENAI_API_KEY')))"
```
