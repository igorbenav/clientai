# ClientAI Tutorial: Building a Local Task Planner

In this tutorial, we'll create a local task planning system using ClientAI and Ollama. Our planner will break down goals into actionable tasks, create realistic timelines, and manage resources - all running on your local machine.

## Table of Contents

1. [Introduction](#1-introduction)
2. [Setting Up the Project](#2-setting-up-the-project)
3. [Building the Task Planner](#3-building-the-task-planner)
4. [Creating the Tools](#4-creating-the-tools)
5. [Implementing the Interface](#5-implementing-the-interface)
6. [Running the Planner](#6-running-the-planner)
7. [Further Improvements](#7-further-improvements)

## 1. Introduction

ClientAI with Ollama allows us to run AI models locally, making it perfect for tools like task planners. Our implementation will demonstrate:
- Local AI model management with OllamaManager
- Tool-based task decomposition
- Realistic timeline generation with error handling
- Structured plan formatting

The end result will be a practical planning tool that runs entirely on your machine, with no need for external API calls.

## 2. Setting Up the Project

First, create a new directory for your project:

```bash
mkdir local_task_planner
cd local_task_planner
```

Install ClientAI with Ollama support:

```bash
pip install "clientai[ollama]"
```

Make sure you have Ollama installed on your system. You can get it from [Ollama's website](https://ollama.ai).

## 3. Building the Task Planner

Let's start by importing our required modules and setting up logging:

```python
from datetime import datetime, timedelta
from typing import Dict, List
import logging

from clientai import ClientAI
from clientai.agent import create_agent, tool
from clientai.ollama import OllamaManager

logger = logging.getLogger(__name__)
```

Now, let's create our TaskPlanner class. This will be the core of our application:

```python
class TaskPlanner:
    """A local task planning system using Ollama."""

    def __init__(self):
        """Initialize the task planner with Ollama."""
        # Create an instance of OllamaManager to handle the local AI server
        self.manager = OllamaManager()
        # Placeholders for the AI client and planner agent
        self.client = None  # Will hold the ClientAI instance
        self.planner = None  # Will hold the planning agent

    def start(self):
        """Start the Ollama server and initialize the client."""
        # Start the local Ollama server
        self.manager.start()
        # Create a ClientAI instance connected to the local Ollama server
        self.client = ClientAI("ollama", host="http://localhost:11434")

        # Create a planning agent with specific capabilities
        self.planner = create_agent(
            client=self.client,
            role="task planner",  # Define the agent's role
            system_prompt="""You are a practical task planner. Break down goals into
            specific, actionable tasks with realistic time estimates and resource needs.
            Use the tools provided to validate timelines and format plans properly.""",
            model="llama3",  # Specify which local model to use
            step="think",    # Set the agent to use thinking steps
            tools=[validate_timeline, format_plan],  # Provide planning tools
            tool_confidence=0.8,  # Set minimum confidence for tool usage
            stream=True,  # Enable real-time response streaming
        )

    def stop(self):
        """Stop the Ollama server."""
        # Safely shut down the Ollama server if it's running
        if self.manager:
            self.manager.stop()
```

The initialization is straightforward - we create an OllamaManager instance and prepare placeholders for our client and planner. The actual initialization happens in the start method, which:

- Starts the Ollama server
- Creates a ClientAI instance connected to the server
- Initializes the planning agent with the necessary tools

## 4. Creating the Tools

Our planner needs two main tools: one for validating timelines and another for formatting plans:

```python
@tool(name="validate_timeline")  # Register this as a tool for the agent to use
def validate_timeline(tasks: Dict[str, int]) -> Dict[str, dict]:
    """
    Validate time estimates and create a realistic timeline.

    Args:
        tasks: Dictionary of task names and estimated hours

    Returns:
        Dictionary with start dates and deadlines
    """
    try:
        # Initialize timeline calculation with current date as starting point
        current_date = datetime.now()
        timeline = {}
        accumulated_hours = 0  # Track total hours for sequential scheduling

        for task, hours in tasks.items():
            try:
                # Convert hours to integer safely, handling various input types
                # str() handles potential non-string inputs
                # float() handles decimal numbers
                # int() converts to final integer form
                hours_int = int(float(str(hours)))

                # Skip tasks with zero or negative hours
                if hours_int <= 0:
                    logger.warning(f"Skipping task {task}: Invalid hours value {hours}")
                    continue
                
                # Calculate working days needed (assuming 6 productive hours per day)
                days_needed = hours_int / 6

                # Calculate start date based on accumulated hours from previous tasks
                start_date = current_date + timedelta(hours=accumulated_hours)
                # Calculate end date based on working days needed
                end_date = start_date + timedelta(days=days_needed)

                # Store task timeline information
                timeline[task] = {
                    "start": start_date.strftime("%Y-%m-%d"),  # Format dates as strings
                    "end": end_date.strftime("%Y-%m-%d"),
                    "hours": hours_int,
                }

                # Add this task's hours to accumulated total for next task's scheduling
                accumulated_hours += hours_int
                
            except (ValueError, TypeError) as e:
                # Handle invalid hour values (non-numeric strings, invalid types, etc.)
                logger.warning(f"Skipping task {task}: Invalid hours value {hours} - {e}")
                continue

        return timeline
    except Exception as e:
        # Catch any unexpected errors and return empty timeline rather than failing
        logger.error(f"Error validating timeline: {str(e)}")
        return {}

@tool(name="format_plan")  # Register this as a named tool for the agent
def format_plan(
    tasks: List[str],          # List of task names
    timeline: Dict[str, dict], # Timeline data from validate_timeline tool
    resources: List[str]       # List of required resources
) -> str:
    """
    Format the plan in a clear, structured way.

    Args:
        tasks: List of tasks
        timeline: Timeline from validate_timeline
        resources: List of required resources

    Returns:
        Formatted plan as a string
    """
    try:
        # Create header for the plan
        plan = "== Project Plan ==\n\n"

        # Start tasks section with timeline details
        plan += "Tasks and Timeline:\n"
        for i, task in enumerate(tasks, 1):  # Enumerate from 1 for natural numbering
            if task in timeline:  # Only include tasks that have timeline data
                t = timeline[task]
                # Format each task with indented details
                plan += f"\n{i}. {task}\n"                      # Task name with number
                plan += f"   Start: {t['start']}\n"            # Start date indented
                plan += f"   End: {t['end']}\n"                # End date indented
                plan += f"   Estimated Hours: {t['hours']}\n"  # Hours indented

        # Add resources section
        plan += "\nRequired Resources:\n"
        for resource in resources:
            plan += f"- {resource}\n"  # Bullet points for resources

        return plan
    except Exception as e:
        # Log any formatting errors and return error message
        logger.error(f"Error formatting plan: {str(e)}")
        return "Error: Unable to format plan"
```

## 5. Implementing the Interface

Now let's add the main interface method and command-line interface:

```python
def get_plan(self, goal: str) -> str:
    """
    Generate a plan for the given goal.

    Args:
        goal: The goal to plan for

    Returns:
        A formatted plan string
    """
    if not self.planner:
        raise RuntimeError("Planner not initialized. Call start() first.")

    return self.planner.run(goal)

def main():
    # Create an instance of our TaskPlanner
    planner = TaskPlanner()

    try:
        # Display welcome messages and instructions
        print("Task Planner (Local AI)")
        print(
            "Enter your goal, and I'll create a practical, timeline-based plan."
        )
        print("Type 'quit' to exit.")

        # Initialize the planner and start the Ollama server
        planner.start()

        # Main interaction loop
        while True:
            # Visual separator for better readability
            print("\n" + "=" * 50 + "\n")
            # Get user input
            goal = input("Enter your goal: ")

            # Check for quit command
            if goal.lower() == "quit":
                break

            try:
                # Get the plan from the planner
                plan = planner.get_plan(goal)
                print("\nYour Plan:\n")
                # Stream the response chunk by chunk for real-time output
                for chunk in plan:
                    print(chunk, end="", flush=True)  # flush=True ensures immediate display
            except Exception as e:
                # Handle any errors during plan generation
                print(f"Error: {str(e)}")

    finally:
        # Ensure the Ollama server is properly shut down
        # This runs even if there's an error or user quits
        planner.stop()

if __name__ == "__main__":
    main()
```

The interface includes:

- Proper error handling for uninitialized planner
- Graceful shutdown in the finally block
- Streaming output support with flush
- Clear user instructions
- Clean exit handling

## 6. Running the Planner

To use the planner, simply run:

```bash
python task_planner.py
```

Example interaction:

```
Task Planner (Local AI)
Enter your goal, and I'll create a practical, timeline-based plan.
Type 'quit' to exit.

==================================================

Enter your goal: Create a personal portfolio website

Your Plan:

== Project Plan ==

Tasks and Timeline:
1. Requirements Analysis and Planning
   Start: 2024-12-08
   End: 2024-12-09
   Estimated Hours: 6

2. Design and Wireframing
   Start: 2024-12-09
   End: 2024-12-11
   Estimated Hours: 12

3. Content Creation
   Start: 2024-12-11
   End: 2024-12-12
   Estimated Hours: 8

4. Development
   Start: 2024-12-12
   End: 2024-12-15
   Estimated Hours: 20

Required Resources:
- Design software (e.g., Figma)
- Text editor or IDE
- Web hosting service
- Version control system
```

## Understanding the Implementation

Our task planner has several key components working together:

1. **Timeline Validator**:
    - Converts all time estimates to integers safely
    - Assumes 6 productive hours per day for realistic scheduling
    - Handles invalid inputs gracefully
    - Provides detailed logging for debugging

2. **Plan Formatter**:
    - Creates consistent, readable output
    - Includes error handling for malformed data
    - Clearly separates tasks, timelines, and resources

3. **OllamaManager Integration**:
    - Handles server lifecycle automatically
    - Provides clean startup and shutdown
    - Manages the connection to the local AI model

4. **Error Handling**:
    - Comprehensive try/except blocks
    - Detailed logging
    - User-friendly error messages
    - Graceful degradation on failures

## 7. Further Improvements

Consider these enhancements to make the planner even more robust:

- Add dependency tracking between tasks
- Include cost calculations for resources
- Save plans to files or project management tools
- Track progress against the original plan
- Add validation for resource availability
- Implement parallel task scheduling
- Add support for recurring tasks
- Include priority levels for tasks

Remember that performance will depend on your chosen Ollama model. Experiment with different models to find the right balance between speed and plan quality for your needs.