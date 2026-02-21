$ErrorActionPreference = 'Stop'

$workspace = 'D:/CS-OmniMamba'
$cacheDir = Join-Path $env:USERPROFILE '.cache/superpowers'
$repoUrl = 'https://github.com/obra/superpowers'

$skills = @(
    @{src='writing-plans'; cmd='write-plan'; desc='Create a detailed implementation plan (Superpowers)'},
    @{src='executing-plans'; cmd='execute-plan'; desc='Execute an implementation plan with checkpoints'},
    @{src='brainstorming'; cmd='brainstorm'; desc='Generate creative solutions and explore ideas'},
    @{src='test-driven-development'; cmd='tdd'; desc='Implement code using strict TDD cycles'},
    @{src='systematic-debugging'; cmd='investigate'; desc='Perform systematic root-cause analysis'},
    @{src='verification-before-completion'; cmd='verify'; desc='Ensure fixes work before claiming success'},
    @{src='using-git-worktrees'; cmd='worktree'; desc='Create isolated workspace for parallel development'},
    @{src='finishing-a-development-branch'; cmd='finish-branch'; desc='Merge, PR, or discard completed work'},
    @{src='requesting-code-review'; cmd='review'; desc='Request a self-correction code review'},
    @{src='receiving-code-review'; cmd='receive-review'; desc='Respond to code review feedback'},
    @{src='subagent-driven-development'; cmd='subagent-dev'; desc='Dispatch subagents for task-by-task development'},
    @{src='dispatching-parallel-agents'; cmd='dispatch-agents'; desc='Run concurrent subagent workflows'},
    @{src='writing-skills'; cmd='write-skill'; desc='Create new skills following TDD best practices'},
    @{src='using-superpowers'; cmd='superpowers'; desc='Learn about the Superpowers capabilities'}
)

if (Test-Path $cacheDir) {
    git -C $cacheDir pull
} else {
    git clone $repoUrl $cacheDir
}

$workspaceSuperpowers = Join-Path $workspace '.superpowers'
if (Test-Path $workspaceSuperpowers) {
    Remove-Item $workspaceSuperpowers -Recurse -Force
}
New-Item -ItemType Directory -Path $workspaceSuperpowers -Force | Out-Null
Copy-Item -Path (Join-Path $cacheDir 'skills') -Destination $workspaceSuperpowers -Recurse -Force

$githubDir = Join-Path $workspace '.github'
$promptsDir = Join-Path $githubDir 'prompts'
New-Item -ItemType Directory -Path $promptsDir -Force | Out-Null

$instructions = @'
<!-- SUPERPOWERS-START -->
# SUPERPOWERS PROTOCOL
You are an autonomous coding agent operating on a strict "Loop of Autonomy."

## CORE DIRECTIVE: The Loop
For every request, you must execute the following cycle:
1. **PERCEIVE**: Read `plan.md`. Do not act without checking the plan.
2. **ACT**: Execute the next unchecked step in the plan.
3. **UPDATE**: Check off the step in `plan.md` when verified.
4. **LOOP**: If the task is large, do not stop. Continue to the next step.

## YOUR SKILLS (Slash Commands)
VS Code reserved commands are replaced with these Superpowers equivalents:

- **Use `/write-plan`** (instead of /plan) to interview me and build `plan.md`.
- **Use `/investigate`** (instead of /fix) when tests fail to run a systematic analysis.
- **Use `/tdd`** to write code. NEVER write code without a failing test.

## RULES
- If `plan.md` does not exist, your ONLY valid action is to ask to run `/write-plan`.
- Do not guess. If stuck, write a theory in `scratchpad.md`.

## AVAILABLE SKILLS

All skill definitions are available at `./.superpowers/skills/` (workspace-resident).
This path keeps all Superpowers content within your workspace, preventing permission prompts.
<!-- SUPERPOWERS-END -->
'@
Set-Content -Path (Join-Path $githubDir 'copilot-instructions.md') -Value $instructions -Encoding UTF8

foreach ($skill in $skills) {
    $srcPath = Join-Path $workspaceSuperpowers ("skills/{0}/SKILL.md" -f $skill.src)
    if (-not (Test-Path $srcPath)) {
        Write-Warning "Missing skill: $($skill.src)"
        continue
    }
    $body = Get-Content -Raw $srcPath
    $prompt = "---`nname: $($skill.cmd)`ndescription: $($skill.desc)`n---`n# Skill: $($skill.src)`n$body"
    Set-Content -Path (Join-Path $promptsDir ("{0}.prompt.md" -f $skill.cmd)) -Value $prompt -Encoding UTF8
}

Write-Host 'Superpowers fallback installation completed.'
