# track1_nottingDuck

## Team members

We are a team called *NottingDuck* with following three members:

- [Shuli Wang](https://github.com/NingShuZhu)
- [Yujie Yao](https://github.com/JackyYao1021)
- [Ningbo Wei](https://github.com/NingboWEI)

## Project description

### 1. 📌 Project Overview

**Samantha** is a natural language powered assistant that helps you interact with your system more easily.  
It can be used in your terminal on **openEuler** (and other Linux systems), which will translate natural language into executable commands, and execute them.

### 2. 📖 User Guide

Please go to [how_to_use_Samantha](readme_help/how_to_use_Samantha.md) for detailed guidance :)

### 3. 🏗️ System Architecture

Our system follows a multi-agent architecture that processes user natural language requests step by step until successful execution. The workflow ensures clarity, safety, and self-correction throughout the process.

![Overall system design](readme_help/Group_10.png)

- **Agent-0 – Request Completeness Checker:**  
  Validates whether the user’s natural language request is clear and executable. If key information is missing, it asks clarifying questions.

- **Agent-1 – User Intent Parser:**  
  Breaks the clarified request into structured steps in natural language, clearly defining what actions need to be performed step by step.

- **Agent-2 – Shell Command Generator:**  
  Converts these steps into robust and executable shell commands tailored for the openEuler environment.

- **Agent-3 – Code Confirmation Agent:**  
  Summarizes the commands in plain language and asks for user confirmation, especially for risky operations and permission-related actions, before execution.

- **Agent-e – Error Correction Agent:**  
  Monitors execution results. If a command fails, it analyzes the error, adjusts the plan, and sends the updated result back to Agent-2 to automatically regenerate commands and retry execution.

In addition, we designed an **efficient interaction mechanism**: after setting up _(simply using `source /work/setup.sh`)_, users can directly invoke **Samantha** in the terminal simply by typing `samantha` followed by their request, without needing to manually run a script each time.

![efficient interaction](readme_help/simple.png)

### 3. ✅ Current Implementation & Results

Our current version of **Samantha** successfully implements all core features from **Tier 1** and **Tier 2**, as well as several features in **Tier 3**, providing a robust and intelligent natural language interface for the openEuler terminal.

![efficient interaction](readme_help/overall_progress.png)

#### 🧰 Tier 1 – Basic File Operations  
- ✅ **Navigation:** change directories and list files using natural language.  
- ✅ **Creation:** create new files and directories.  
- ✅ **Basic manipulation:** copy, move, and rename files or directories.  
- ✅ **Simple search:** find files or directories by exact name.  
- ✅ **User feedback:** provide clear confirmation of actions or error messages.  
- ✅ **Safety mechanism:** confirmation required for destructive operations (e.g., delete).
![Safety mechanism](readme_help/t1_safty.png)  
*Figure：Safety mechanism - Ask user if to permanently delete a file.*

#### ⚙️ Tier 2 – Enhanced Intelligence  

- ✅ **Advanced search and filtering** by type, extension, size, or modification date.  
![efficient interaction](readme_help/t2_example.png)  
*Figure：example of Advance search - file filtering by size.*
- ✅ **Context awareness:** understands relative paths and context implicitly.  
- ✅ **Basic multi-step operations:** handles sequences of actions in a single request.  
- ✅ **Robust error handling:** gracefully handles invalid paths, permission issues, and missing files.
![Robust error handling](readme_help/t2_error_handling.png)  
*Figure：example of Robust error handling - create a file in non-exist folder.*


#### 🚧 Tier 3 – More advanced  

- ✅ **Content-Aware Search:**  able to search for files ontaining specific keywords. 
![keyword search](readme_help/t3_example1.png)
*Figure：example of keyword search - find all the files include "openai".*
- ✅ **Self-Correction:** when an error is encounter, Samantha will try different approaches to achieve the goal. (Up to _three_ self-correction attempts will be tried)
![keyword search](readme_help/t3_example_sc.png)
*Figure：example of self correction.*
- ✅ **Natural Conversation Flow:** Samantha can handle follow-up questions and clarifications. In case the request is not clear, she will ask for clarification until she exactly get your point. If during the conversion you don't want her to execute the request anymore, she will simply do nothing and exit.
![keyword search](readme_help/t3_example2.png)
*Figure：example of keyword search - find all the files include "openai".*

Due to time constraints, there are Tier 3 features (**agentic capabilities**, **organizational 
Intelligence**) that have not yet fully been implemented. However, we have already reserved internal pipelines for future Tier 3 development, including logging mechanisms and other extensions. More details are discussed in the next section.

### 3. 🚀 Future Work

Although our current implementation focuses on Tier 1 and Tier 2 functionality, we have laid a solid foundation for future development towards these Tier 3 and more advanced intelligent capabilities.

To support these future features, we have already made several key architectural preparations, some have already achieved:

![Future Architecture](readme_help/Group_12.png)
*Figure：Architecture for the future development*

- **Logging System:** A complete logging mechanism has been implemented to record user interactions, execution results, and error information. This provides the groundwork for context-aware decision-making, adaptive learning, and self-improvement.
- **Vector Space Database:** The architecture includes a reserved vector space database interface, enabling future integration of semantic search for PDF files and images.
- **Tool Integration Pathways:** The system design anticipates the inclusion of external tools, allowing the assistant to extend its capabilities beyond basic shell commands. This system will be achieved by using LangGraph.

