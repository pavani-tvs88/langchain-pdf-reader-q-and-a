# langchain-pdf-reader-q-and-a
PDF Document reader Q and A bot using Grado and Gemini API

## Configuration 🔧

This project requires a Google API key (Gemini) set in the `GOOGLE_API_KEY` environment variable.

Options to set it:

- **Codespaces secret (recommended)**: use the GitHub CLI to set a secret in your Codespace:

  ```bash
  gh codespace secret set -n GOOGLE_API_KEY -b <your_key>
  ```

- **Local .env (development)**: create a `.env` file at the project root with:

  ```bash
  GOOGLE_API_KEY=your_key
  ```

- **Shell export**:

  ```bash
  export GOOGLE_API_KEY=your_key
  ```

Do not commit API keys to source control. To verify the variable is available in your environment:

```bash
echo $GOOGLE_API_KEY
# or
python -c "import os; print(bool(os.getenv('GOOGLE_API_KEY')))"
```
