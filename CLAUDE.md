# CLAUDE.md

> **AI Assistant Guide for claude-code-web-test2**
> Last Updated: 2025-12-01

## Overview

This repository (`claude-code-web-test2`) is a test environment for Claude Code web functionality. It serves as a sandbox for testing AI-assisted development workflows, git operations, and feature development with Claude.

## Repository Structure

```
claude-code-web-test2/
├── .git/                 # Git repository data
├── README.md             # Basic repository information
└── CLAUDE.md            # This file - AI assistant guide
```

### Current State

This is a minimal repository currently containing:
- **README.md**: Basic repository identifier
- **CLAUDE.md**: Comprehensive guide for AI assistants (this document)

The repository is in its initial state and serves as a clean slate for testing various development scenarios.

## Development Workflows

### Branch Strategy

This repository uses a **feature branch workflow** with Claude-specific naming conventions:

#### Branch Naming Convention
- **Pattern**: `claude/<session-prefix>-<session-id>`
- **Current branch**: `claude/claude-md-mimlgu2yaipu47hh-01WZgmjuTA48niG1rujTeTfW`
- **Critical**: All branch names MUST start with `claude/` and end with the matching session ID for push operations to succeed

#### Working with Branches

1. **Development Flow**:
   ```bash
   # All development happens on designated Claude branches
   # Check current branch
   git status

   # Create new branch if needed (follows naming convention)
   git checkout -b claude/<prefix>-<session-id>
   ```

2. **Committing Changes**:
   ```bash
   # Stage changes
   git add <files>

   # Commit with descriptive message
   git commit -m "Clear, descriptive commit message"
   ```

3. **Pushing Changes**:
   ```bash
   # ALWAYS use -u flag for first push
   git push -u origin <branch-name>

   # Branch name MUST match pattern: claude/*-<session-id>
   # Otherwise push will fail with 403 error
   ```

### Git Operation Best Practices

#### Push Operations
- **Always** use `git push -u origin <branch-name>` for tracking
- Branch name **MUST** follow `claude/<prefix>-<session-id>` pattern
- **Retry policy**: If push fails due to network errors, retry up to 4 times with exponential backoff (2s, 4s, 8s, 16s)
- **Never** force push to main/master branches
- **Never** skip hooks without explicit user permission

#### Fetch/Pull Operations
- **Prefer** fetching specific branches: `git fetch origin <branch-name>`
- For pulls: `git pull origin <branch-name>`
- **Retry policy**: Same as push (up to 4 retries with exponential backoff)

#### Commit Practices
- Write clear, concise commit messages that explain the "why" not just the "what"
- Follow existing commit message style (check `git log` for patterns)
- **Never** amend commits unless:
  1. User explicitly requests it, OR
  2. Adding fixes from pre-commit hooks
- Before amending, **always** check authorship: `git log -1 --format='%an %ae'`
- **Never** commit files with secrets (.env, credentials.json, etc.)

## Key Conventions for AI Assistants

### File Operations

1. **Reading Before Writing**:
   - **ALWAYS** read a file before modifying it
   - Never propose changes to code you haven't examined
   - Use `Read` tool for file reading, not bash commands

2. **Prefer Editing Over Creating**:
   - **ALWAYS** prefer editing existing files over creating new ones
   - Only create new files when absolutely necessary
   - This applies to all file types, including markdown

3. **Tool Usage**:
   - Use specialized tools (`Read`, `Edit`, `Write`) over bash commands for file operations
   - Use `Glob` for file pattern matching
   - Use `Grep` for content searching
   - Reserve bash for actual terminal operations only

### Code Quality Standards

1. **Security First**:
   - Never introduce security vulnerabilities (command injection, XSS, SQL injection, etc.)
   - Follow OWASP top 10 guidelines
   - If insecure code is written, immediately fix it

2. **Avoid Over-Engineering**:
   - Make only requested or clearly necessary changes
   - Keep solutions simple and focused
   - Don't add unrequested features, refactoring, or "improvements"
   - Don't add docstrings, comments, or type annotations to unchanged code
   - Only add comments where logic isn't self-evident

3. **No Premature Abstraction**:
   - Don't create helpers/utilities for one-time operations
   - Don't design for hypothetical future requirements
   - Three similar lines > premature abstraction

4. **Error Handling**:
   - Only add error handling for scenarios that can actually occur
   - Trust internal code and framework guarantees
   - Only validate at system boundaries (user input, external APIs)

5. **Clean Deletions**:
   - Avoid backwards-compatibility hacks
   - Don't rename unused variables with `_` prefix
   - Don't add `// removed` comments
   - If code is unused, delete it completely

### Task Management

1. **Use TodoWrite Tool**:
   - Use for any complex multi-step task (3+ steps)
   - Plan tasks before executing
   - Track progress in real-time

2. **Task States**:
   - `pending`: Not yet started
   - `in_progress`: Currently working (ONLY ONE at a time)
   - `completed`: Finished successfully

3. **Completion Requirements**:
   - Mark tasks completed **immediately** after finishing
   - **ONLY** mark complete when fully accomplished
   - Keep as `in_progress` if errors/blockers encountered
   - Never batch completions

### Communication Style

1. **Concise and Clear**:
   - Output is displayed in CLI with monospace font
   - Be short and concise
   - Use GitHub-flavored markdown for formatting

2. **No Unnecessary Emojis**:
   - Only use emojis if user explicitly requests
   - Avoid emojis in communication by default

3. **Direct Output**:
   - Output text directly to communicate with users
   - Never use bash echo or comments to communicate
   - Tools are for tasks, not communication

4. **Code References**:
   - Use pattern `file_path:line_number` when referencing code
   - Example: "Error handling occurs in src/main.js:42"

### Exploration and Search

1. **Use Task Tool for Exploration**:
   - When exploring codebase for context, use `Task` tool with `subagent_type=Explore`
   - Don't use `Glob` or `Grep` directly for open-ended exploration
   - Examples:
     - "Where are errors handled?" → Use Explore agent
     - "What is the codebase structure?" → Use Explore agent

2. **Parallel Operations**:
   - Make independent tool calls in parallel when possible
   - Use sequential calls only when dependencies exist
   - Example: Can read multiple files in parallel

## GitHub Integration

### Important Limitations

- **GitHub CLI (`gh`) is NOT available** in this environment
- For GitHub issues/PRs, ask user to provide information directly
- Cannot automatically create PRs or interact with GitHub API

### Pull Request Creation (When Available)

If PR creation is requested:

1. Understand full commit history from branch divergence
2. Analyze ALL commits (not just latest)
3. Draft comprehensive PR summary with:
   - Summary section (1-3 bullet points)
   - Test plan (markdown checklist)
4. Use HEREDOC for PR body formatting

## Testing and Validation

### Before Committing

1. **Review Changes**:
   ```bash
   git status          # See untracked files
   git diff            # See unstaged changes
   git diff --staged   # See staged changes
   git log             # Check recent commit style
   ```

2. **Analyze Changes**:
   - Ensure changes match intent
   - Verify no secrets are included
   - Check commit message accuracy

3. **Handle Pre-commit Hooks**:
   - If commit fails due to hook changes, retry ONCE
   - Verify safe to amend (check authorship, not pushed)
   - If safe: amend commit
   - If not: create NEW commit

## Project-Specific Patterns

### Initial Repository State

This repository is currently minimal and serves as a test environment. When developing:

1. **Establish structure gradually**: Add directories/files as needed
2. **Document decisions**: Update this CLAUDE.md when adding patterns
3. **Test workflows**: Use this as a sandbox for git/development workflows
4. **Keep it simple**: Only add what's necessary for testing

### Future Development

As this repository grows, update this document with:

- **New directory structures**: Document purpose of each directory
- **Build systems**: If adding package.json, Makefile, etc.
- **Testing frameworks**: Document how to run tests
- **Coding standards**: Language-specific conventions
- **Dependencies**: How to install and manage
- **Deployment**: If applicable

## Quick Reference

### Essential Commands

```bash
# Check status
git status

# Create feature branch
git checkout -b claude/<prefix>-<session-id>

# Stage and commit
git add <files>
git commit -m "Description of changes"

# Push (with retry logic for network errors)
git push -u origin <branch-name>

# View file structure
ls -la

# Check git history
git log --oneline -10
```

### File Operations Preferences

| Task | Use This | NOT This |
|------|----------|----------|
| Read files | `Read` tool | `cat`, `head`, `tail` |
| Edit files | `Edit` tool | `sed`, `awk` |
| Write files | `Write` tool | `echo >`, `cat <<EOF` |
| Find files | `Glob` tool | `find`, `ls` |
| Search content | `Grep` tool | `grep`, `rg` |
| Explore code | `Task` (Explore) | Multiple `Grep` calls |

## Maintenance

### Keeping CLAUDE.md Updated

This document should be updated when:

- Repository structure changes significantly
- New development patterns are established
- Build/test systems are added
- New coding conventions are adopted
- Deployment workflows are introduced

**Last structural change**: Initial creation (2025-12-01)

---

## AI Assistant Checklist

Before starting work, ensure you:

- ✅ Read this CLAUDE.md thoroughly
- ✅ Check current branch matches session ID pattern
- ✅ Read existing files before modifying
- ✅ Use TodoWrite for multi-step tasks
- ✅ Follow git best practices (proper branch naming, clear commits)
- ✅ Prefer simple solutions over complex ones
- ✅ Only create files when absolutely necessary

During work:

- ✅ Mark todos in_progress before starting
- ✅ Complete todos immediately after finishing
- ✅ Use specialized tools over bash for file operations
- ✅ Keep communication concise and clear
- ✅ Reference code with file_path:line_number pattern

Before committing:

- ✅ Review all changes (git status, git diff)
- ✅ Write clear commit message explaining "why"
- ✅ Verify no secrets in commit
- ✅ Check branch name follows claude/* pattern

When pushing:

- ✅ Use `git push -u origin <branch-name>`
- ✅ Implement retry logic for network failures
- ✅ Verify branch name matches session ID
