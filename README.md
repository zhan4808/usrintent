## usrintent

Intent refinement CLI with feedback loops, scoring, prompt versioning, and prompt sharing between CLI and Desktop.

### What Works Today
- Intent lifecycle: create, run, rate, refine, history.
- Real model adapters: OpenAI + Anthropic (via env file).
- Local persistence: SQLite.
- Share (v0): refined prompts/ratings via a shared JSON file.
- Sync (v0): database copy between CLI and desktop paths.

### Quick Start
1) Create `.usrintent.env` in the repo:
   OPENAI_API_KEY=your_key_here  
   ANTHROPIC_API_KEY=your_key_here  
   USRINTENT_PROVIDER=openai  
   USRINTENT_DESKTOP_DB=/Users/you/Library/Application Support/usrintent/usrintent.sqlite  
   USRINTENT_SHARE_DIR=/Users/you/Library/Application Support/usrintent/share  

2) Run:
   ./usrintent intent new "Add CSV parser with error handling" --type=code --constraints="no deps,keep public API" --done-criteria="tests pass,no lint errors"  
   ./usrintent run intent_XXXXXXXX --provider=openai --model=gpt-4o-mini  
   ./usrintent feedback prompt_XXXXXXXX --rating=4 --edit-file=diff.patch  
   ./usrintent refine intent_XXXXXXXX  
   ./usrintent history intent_XXXXXXXX  

### Commands
| Command | Purpose |
| --- | --- |
| `intent new/get/list` | Create and browse intents |
| `run` | Generate output for an intent |
| `feedback` | Rate output and add edits |
| `refine` | Improve the latest prompt |
| `history` | Prompt timeline for an intent |
| `prompt get/list` | Inspect stored prompts |
| `feedback-list` | List feedback for a prompt |
| `export/import` | Backup or restore JSON |
| `watch start/stop/status` | Reminders |
| `share push/pull/status` | Share refined prompts/ratings |
| `sync push/pull/status` | Mirror CLI and desktop DB |

### Prompt Sharing (v0)
Share uses a single JSON file in `USRINTENT_SHARE_DIR`.  
- `share push` exports prompts + ratings  
- `share pull` imports new prompts into the target DB  

### Current Sync (v0)
This is a direct database copy between CLI and Desktop paths.  
- Push: CLI → Desktop  
- Pull: Desktop → CLI  

Example:
./usrintent sync status  
./usrintent sync push --interactive  

### Why the current sync is limited
It copies the CLI database only. It does not:
- transform Claude Code skills/plugins into Desktop DXT extensions
- sync MCP server configs
- resolve conflicts via diffs/checkboxes
- do registry lookups
- support rollback UI

### Target Sync (Claude-sync style)
Goal: full bidirectional sync with transformation and diff review.

Feature checklist
- Bidirectional sync: Code → Desktop or Desktop → Code
- Format transformation: Skills/Plugins ↔ DXT Extensions
- MCP server sync: Claude Code settings ↔ Desktop config
- Registry lookup: use official DXT when available
- Interactive UI: checkboxes + diff preview
- Auto-backup + rollback

Implementation barriers to address
1) Inventory + paths
   - Claude Code: ~/.claude/skills, ~/.claude/plugins, ~/.claude/settings.json
   - Claude Desktop: ~/Library/Application Support/Claude/Claude Extensions, claude_desktop_config.json

2) Diff + conflict handling
   - File hashing, manifest state, and per-item diff
   - Selection UI with checkboxes

3) Rollback
   - Timestamped backups + restore selection

### Context sharing (Desktop ↔ CLI)
Likely not possible for full conversation context due to Claude Desktop’s internal storage and context handling. Practical options:
- One-way sharing: export prompt artifacts from CLI to Desktop (prompts/skills) so they can be used there.
- If Desktop exposes a local data store or export API in the future, we can add import for history.

### Example Use Cases
1) Developer prompt refinement
   - Capture intent, run against model, rate output, refine prompt versions.

2) Team prompt sharing
   - Share refined prompts/ratings via `share push/pull`.

3) Code/Desktop sync (future)
   - Sync skills + MCP servers with diff previews and rollback.

### Proposed Sync Flow (future)
Scan → Diff → Select → Backup → Transform → Sync → Update Manifest

### Mini Diagram (future)
CLI (skills/plugins/settings) --> Transform --> Desktop (DXT/extensions/config)  
Desktop (DXT/extensions/config) --> Transform --> CLI (skills/plugins/settings)

### Notes
- The sync v0 is safe for CLI database mirroring only.
- The full Claude-sync feature set is planned as a separate module/CLI with interactive UI.
