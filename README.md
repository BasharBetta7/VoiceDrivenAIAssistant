# AI Voice Assistant Project :
## GOAL : Design a user-specific voice assistant agent that can respond to audio commands.

## Baseline Pipeline: 
```mermaid
flowchart TD
    A[Microphone audio] --> B[Detect and capture speech]
    B --> C{Speaker verification\n<b>ECAPA-TDNN</b>}
    C -->|Rejected| D[Ignore command]
    C -->|Accepted| E[ASR: audio to text \n<b>Qwen3-ASR-0.6B</b>]
    E --> F[LLM: select action and arguments \n<b>Qwen3.5:4b</b>]
    G[Available script descriptions] --> F
    F --> H{Validate structured output}
    H -->|Valid action| I[Execute registered script]
    H -->|Ambiguous request| J[Ask for clarification]
    H -->|Unsupported or invalid| K[Explain why no action ran]
    I --> L[Report execution result]
```
## Main Modules:
1. <b>Speaker Verification</b> : pre-trained model to enroll the user's voice and identify it for authorization.
2. <b>ASR Model</b>: pre-trained model to convert human speech to text.
3. <b>LLM</b> with structured outputs, tools, and guardrails to respond to the commands as needed.
4. <b>Evaluation Module </b>: to measure the performance of the overall assistant (pre-written tasks), the speaker verification (False Acceptance Rate, and False Rejection Rate), the ASR Module (Word Error Rate), Responsiveness (latency), and Resources.


## Checkpoints : 
1. Specify the main modules. - [x] 
2. Run the full pipeline on local machine successfully (no bugs). [x]
3. The pipeline can now run a simple script from command. [x]
4. Incorporate the Speaker Verification and test it. [x]
5. The pipeline can add new tasks and respond to them. 
6. Run the full evals to measure and optimize the performance of the baseline.
7. Provide the UI for the pipeline.
8. Dockerize the project.
9. CI/CD integration 
10. Publish. 


## Downsides:
1. The speaker verfication contains high False Positives. And is prone to synthetic speaker imitation.
<b>FIX</b>: sweep different thresholds and test it on available datasets.

2. The model records audio for fixed duration (10 seconds) and therefore is not dynamic.
<b>FIX</b>: Make dynamic-duration audio recording based on voice activity instead of fixed duration.


## LLM Agent:
### Capabilities:
1. Answer simple questions from memory without hellucination
2. Execute python/shell scripts
3. Write new scripts and add them to the tools list for future use
4. Remember user preferences and history among sessions.
5. Ask for clarification until the request is fully understood

### Limitations and guardrails:
1. Can't execute scripts that contain certain dangerous commands 
2. Can't access recent knowledge (offline)
3. 

## TODO:
1. Enhance the CLI : remove non-necessary WARNINGS and INFO logs from the stream.
2. Fix the problem with LD_PRELOAD falling back to another version when not specified explicitly
3. Create a secret phrase to launch the model, and sleep after long silence.
4. Meausre the metrics : latency, correctness. And traces : token usage, prices .. etc.s


