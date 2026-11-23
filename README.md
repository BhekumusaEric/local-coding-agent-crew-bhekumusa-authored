# Local Ollama coding crew

`crew.py` runs a sequential Planner → Researcher → Builder → Tester → Reviewer
workflow using Ollama's local `/api/chat` tool calling. These are separate
role-specific turns with the same local model, not independent parallel
models. `agent.py` remains available as the simpler single-agent runner.
Neither runner uses OpenAI's Responses API or changes Continue's setup.

## Prerequisites

- Python 3.11 or newer
- Ollama running locally with `qwen2.5-coder:3b` available
- Node.js/npm for the filesystem MCP server
- Node.js/npm and a Playwright browser runtime for browser-based research
- `uv` and Git for Windows for Git status/diff tools and public-page fetching

The runner's Python dependency is installed in its own virtual environment:

```powershell
Set-Location C:\path\to\local-coding-agent-crew
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

`npx` and `uvx` fetch their configured MCP packages on first launch. Review
each package's upstream source before trusting it. Research can fetch known
public URLs and use Playwright to search/read public pages; it is not an
unrestricted crawler.
After installing Node.js or `uv`, open a new terminal (or restart VS Code) so
the updated `PATH` is visible to the runner.

## Run

Point `--workspace` at the project the crew should work in:

```powershell
$agent = 'C:\path\to\local-coding-agent-crew'
$project = 'C:\path\to\your\project'
& "$agent\.venv\Scripts\python.exe" "$agent\crew.py" `
  --workspace $project `
  "Describe the feature to build or the issue to fix."
```

The MCP configuration is in `mcp-servers.json`; `${workspaceFolder}` is
expanded to the `--workspace` path. Set `test_command` there to the project's
fixed test command, or remove it if no test runner is configured.

## Safety and loop behavior

- The crew is sequential and bounded to 20 model turns, 30 tool calls, and one
  Builder → Tester → Reviewer repair cycle.
- The Researcher receives Fetch plus a limited Playwright tool set. Public
  navigation and browser clicks require approval. Direct navigation blocks
  literal private/reserved IPs and local/internal hostnames; approved clicks
  and server redirects are not independently URL-validated. Web content is
  treated as untrusted and findings should include source URLs.
- Filesystem access is scoped to the selected workspace. Writes and other MCP
  actions require interactive approval unless a server explicitly marks them
  read-only.
- Tests and every project command require approval. Commands run as an argv
  array with `shell=False` and the workspace as the working directory. They
  can still execute project code, so review each command before approving.
- Local commits and remote pushes are never automatic. The Builder must request
  commands such as `git add`, `git commit`, or `git push`; each requires a
  separate confirmation. A remote must be configured before pushing.
- A local progress journal is written under `local_coding_agent/run_history/`.
  It records the planner's measurable acceptance checks and checkpoints each role
  before and after execution; tester evidence updates each check only when its
  exact text and evidence are reported.
