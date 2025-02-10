import requests
import json
from typing import Dict, List
import time
from src.prompt_evaluator import PromptEvaluator

def main():
    evaluator = PromptEvaluator()

    base_prompt = (
        "Create an actor system that processes messages containing numerical operations"
    )
    base_system_prompt = "You are a code assistant specialized in Akka and Scala. Provide only code in your response, without explanations or markdown formatting. Focus on actor-based solutions."

    variations = [
        {
            "name": "Zero shot",
            "prompt": base_prompt,
            "system_prompt": base_system_prompt,
        },
        {
            "name": "Chain of Thought",
            "prompt": f"""Let's design an Akka actor system step by step:
1. What message types do we need for numerical operations?
2. How should we structure the actor hierarchy?
3. How will we handle state and processing?
4. What error handling do we need?
Now implement: {base_prompt}""",
            "system_prompt": base_system_prompt,
        },
        {
            "name": "Few-Shot Learning",
            "prompt": """Here's an example of a basic Akka actor:

case class Increment(value: Int)
case object GetCount

class CounterActor extends Actor {
  var count = 0
  def receive = {
    case Increment(n) => count += n
    case GetCount => sender() ! count
  }
}

Based on this example, implement: {base_prompt}""",
            "system_prompt": base_system_prompt,
        },
        {
            "name": "Tree of Thoughts",
            "prompt": f"""Three Akka experts are designing a solution:

Expert A (Actor Design):
Step 1: Define message protocol for operations
Step 2: Design actor hierarchy

Expert B (State Management):
Step 1: Plan state handling approach
Step 2: Implement processing logic

Expert C (Error Handling):
Step 1: Add supervision strategy
Step 2: Implement recovery mechanisms

Synthesize their approaches and implement: {base_prompt}""",
            "system_prompt": base_system_prompt,
        },
        {
            "name": "ReAct",
            "prompt": f"""Task: {base_prompt}

Thought 1: Need to define message protocol
Action 1: Create case classes for operations
Output 1: Mathematical operation messages defined

Thought 2: Design actor structure
Action 2: Create main actor class
Output 2: Actor with message handling

Thought 3: Add processing logic
Action 3: Implement numerical operations
Output 3: Complete implementation with error handling

Final Implementation:""",
            "system_prompt": base_system_prompt,
        },
    ]

    results = evaluator.evaluate_prompt_variations(base_prompt, variations)

    for result in results:
        print("\n" + "=" * 50)
        print(f"Variation: {result['variation_name']}")
        print(f"Time taken: {result['time_taken']:.2f} seconds")
        print("\nResponse:")
        print(result["response"])


if __name__ == "__main__":
    main()
