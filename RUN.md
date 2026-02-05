Run Commands

Command List (short)
- intent new/get/list: log and browse intents
- run: generate output for an intent
- feedback: rate and add edits
- refine: improve the latest prompt
- history: prompt timeline for an intent
- prompt get/list: inspect stored prompts
- feedback-list: list feedback for a prompt
- export/import: backup or restore JSON
- watch start|stop|status: reminders
- sync push|pull|status: sync CLI and desktop databases

Setup
- Python 3 required. Database file defaults to ./usrintent.sqlite.
- Override database path with --db /absolute/path/to/usrintent.sqlite
- Optional env file: create ./.usrintent.env with API keys and defaults.
  - Optional desktop DB path: USRINTENT_DESKTOP_DB=~/Library/Application Support/usrintent/usrintent.sqlite

Create an intent
- python3 usrintent_cli.py intent new "Add CSV parser with error handling" --type=code --constraints="no deps,keep public API" --done-criteria="tests pass,no lint errors"

Get an intent
- python3 usrintent_cli.py intent get intent_XXXXXXXX

List intents
- python3 usrintent_cli.py intent list

Run a prompt for an intent
- python3 usrintent_cli.py run intent_XXXXXXXX
  - add --output-file=output.txt to save the mock output to a file.
  - add --provider=openai --model=gpt-4o-mini to use OpenAI.
  - add --provider=anthropic --model=claude-3-5-sonnet-latest to use Anthropic.
  - judge scoring: --judge-provider=openai --judge-model=gpt-4o-mini

Submit feedback for a prompt
- python3 usrintent_cli.py feedback prompt_XXXXXXXX --rating=4 --edit-file=diff.patch
  - edit-file is optional; a warning is shown if the file is missing.

Refine the latest prompt for an intent
- python3 usrintent_cli.py refine intent_XXXXXXXX

Show prompt history for an intent
- python3 usrintent_cli.py history intent_XXXXXXXX

Get a prompt
- python3 usrintent_cli.py prompt get prompt_XXXXXXXX

List prompts for an intent
- python3 usrintent_cli.py prompt list intent_XXXXXXXX

List feedback for a prompt
- python3 usrintent_cli.py feedback-list prompt_XXXXXXXX

Export data
- python3 usrintent_cli.py export --file=usrintent_export.json
- python3 usrintent_cli.py export --intent-id=intent_XXXXXXXX --file=intent_export.json

Import data
- python3 usrintent_cli.py import usrintent_export.json

Watch
- python3 usrintent_cli.py watch start
- python3 usrintent_cli.py watch stop
- python3 usrintent_cli.py watch status

Sync
- python3 usrintent_cli.py sync status
- python3 usrintent_cli.py sync push
- python3 usrintent_cli.py sync pull
- python3 usrintent_cli.py sync push --interactive

Env file example (.usrintent.env)
OPENAI_API_KEY=your_key_here
ANTHROPIC_API_KEY=your_key_here
USRINTENT_PROVIDER=openai
USRINTENT_DESKTOP_DB=/Users/you/Library/Application Support/usrintent/usrintent.sqlite
