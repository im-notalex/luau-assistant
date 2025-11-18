[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org)
[![Local RAG](https://img.shields.io/badge/RAG-Local-green?logo=serverless&logoColor=white)]()
[![LuaU](https://img.shields.io/badge/LuaU-Language-orange?logo=lua&logoColor=white)](https://luau.org)
[![Roblox](https://img.shields.io/badge/Roblox-Platform-red?logo=roblox&logoColor=white)](https://create.roblox.com)
[![MIT License](https://img.shields.io/badge/License-MIT-yellow?logo=open-source-initiative&logoColor=white)](https://opensource.org/licenses/MIT)
[![OpenRouter](https://img.shields.io/badge/OpenRouter-API-blue?logo=serverless&logoColor=white)](https://openrouter.ai)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-API-blue?logo=serverless&logoColor=white)](https://huggingface.co)
[![OpenAI](https://img.shields.io/badge/OpenAI-API-blue?logo=serverless&logoColor=white)](https://openai.com)
[![GitHub](https://img.shields.io/badge/GitHub-Repository-black?logo=github&logoColor=white)](https://github.com/im-notalex/luau-assistant)

# ***LuaU RAG Assistant***

- This project is a lightweight local assistant designed for Roblox LuaU scripting and API reference. It uses locally stored documentation and an embedding-based search system to provide instant lookups, explanations, and examples without relying on cloud models.

- Yes, this was made with the 'assistance' of AI. AI for AI? i dunno, but i did make a good portion. :]

# ***-- Requirements --***

**Make sure you have Python 3.10 or newer installed and the required packages.**

> Install everything with:

**pip install -r requirements.txt**

# ***-- Setup --***

**Before running the assistant, run the setup scripts in order:**

> python setup_verify.py
> python api_validator.py

**After both scripts finish, start the web interface:**

> python web_ui.py=

**This opens the assistant interface in your browser.**

# ***-- Usage --***

**The project is designed to be used inside VS Code or any IDE that supports Python. It indexes your local LuaU and Roblox documentation files, then uses them for fast retrieval during coding. (to be used inside i mean run it through your IDE terminal, so its cleaner.)**

**Make sure your markdown and documentation files are inside the Docs/ directory before running the setup scripts.**

**In the web UI Settings panel you can choose between Local, Ollama, OpenAI, OpenRouter, or any OpenAI-compatible endpoint (provide base URL, key, and model). External endpoints automatically enforce the 204k-token context cap, while Ollama/local models keep the full 12M-token window.**

# ***-- Features --***

**Inline LuaU and Roblox API reference**

**Local retrieval system (no external dependencies)**

**Fast lookup using embeddings**

**Works with local LLMs, OpenRouter, and any OpenAI-compatible endpoint (Groq, Anthropic, Google, etc.)**
**12M-token context for Ollama/local models (external endpoints capped at 204k tokens for safety)**

**Tooltip-friendly data for development**

**Runs offline except for optional model endpoints**

# ***-- Troubleshooting --***

**Missing Dependencies**

**If setup scripts complain about modules, run:**

> pip install -r requirements.txt

**Check that you're using the correct Python environment.**

**Port Already in Use**

**If web_ui.py reports that the port is taken, edit the script and change the port number.**

***Documentation Not Found***

**If searches or tooltips return empty results, confirm that your .md files are in Docs/markdown and that embeddings were generated correctly.**

***Local Model Errors***

**If using a local model, confirm the model path and format match what your runner supports. Some formats require extra runtimes.**

# ***-- Notes --***

**This project is intended for personal or educational use. I didn't make this to replace developers, i made it to teach and make it easier for people to learn. Don't use it to replace developers. TLDR; don't be a dirtbag, you dingus.**

**This project is not affiliated with Roblox or any other company. This is a personal project. :]**

**Internet is only required for initial dependency installation or remote model endpoints.**

# ***-- License --***

**This project is released under the MIT License. You can use it, modify it, and include it in your own work.**
**There is no warranty of any kind, and the authors are not responsible for any issues that come from using the code.**

**If you share or publish anything built from this project, leaving credit is appreciated but not required.**
