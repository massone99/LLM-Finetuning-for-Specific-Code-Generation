import requests
import json
from typing import Dict, List
import time


class PromptEvaluator:
    def __init__(self, model_name: str = "llama3.2", base_system_prompt: str = None):
        self.base_url = "http://localhost:11434/api"
        self.model_name = model_name
        self.base_system_prompt = (
            base_system_prompt
            or "You are an expert code assistant specializing in Akka and Scala. Your responses must consist solely of syntactically correct, runnable Scala code—no explanations, comments, or markdown formatting are allowed."
        )

    def generate_response(self, prompt: str, system_prompt: str = None) -> Dict:
        payload = {"model": self.model_name, "prompt": prompt, "stream": False}

        if system_prompt:
            payload["system"] = system_prompt

        response = requests.post(f"{self.base_url}/generate", json=payload)
        return response.json()

    def evaluate_prompt_variations(
        self, base_prompt: str, variations: List[Dict]
    ) -> List[Dict]:
        results = []

        for variant in variations:
            print(f"\nTesting prompt variation: {variant['name']}")

            start_time = time.time()
            response = self.generate_response(
                variant["prompt"], variant.get("system_prompt")
            )
            end_time = time.time()

            results.append(
                {
                    "variation_name": variant["name"],
                    "prompt": variant["prompt"],
                    "system_prompt": variant.get("system_prompt"),
                    "response": response["response"],
                    "time_taken": end_time - start_time,
                }
            )

        return results

    def create_variations(self, base_prompt: str) -> List[Dict]:
        return [
            {
                "name": "Zero shot",
                "prompt": base_prompt,
                "system_prompt": self.base_system_prompt,
            },
            {
                "name": "Chain of Thought",
                "prompt": f"""Let's design the Akka/Scala solution step by step:
1. What message types and case classes do we need?
2. How should we structure the actor hierarchy?
3. What behaviors should each actor implement?
4. How should we handle communication and supervision?

Now implement: {base_prompt}""",
                "system_prompt": self.base_system_prompt,
            },
            {
                "name": "Few-Shot Learning",
                "prompt": f"""Here's an example of a basic Akka/Scala pattern:

// Message Protocol
sealed trait Message
case class Request(data: String) extends Message
case class Response(result: String) extends Message

// Actor Implementation
class ExampleActor extends Actor {{
  def receive: Receive = {{
    case Request(data) => 
      // Process request
      sender() ! Response(s"Processed $data")
    case _ => // Handle unknown messages
  }}
}}

Based on this pattern, implement: {base_prompt}""",
                "system_prompt": self.base_system_prompt,
            },
            {
                "name": "Tree of Thoughts",
                "prompt": f"""Three Akka/Scala experts are designing a solution:

Expert A (Protocol Design):
Step 1: Define message hierarchy and types
Step 2: Plan actor communication patterns

Expert B (Actor Implementation):
Step 1: Design actor behaviors and states
Step 2: Implement message handling logic

Expert C (System Architecture):
Step 1: Plan actor hierarchy and supervision
Step 2: Define system configuration and setup

Synthesize their approaches and implement: {base_prompt}""",
                "system_prompt": self.base_system_prompt,
            },
            {
                "name": "ReAct",
                "prompt": f"""Task: {base_prompt}

Thought 1: Message Protocol Design
Action 1: Define messages and case classes
Output 1: Message types defined

Thought 2: Actor Implementation
Action 2: Create actor class with behaviors
Output 2: Actor implementation complete

Thought 3: System Integration
Action 3: Setup actor system and communication
Output 3: Complete working implementation

Final Implementation:""",
                "system_prompt": self.base_system_prompt,
            },
        ]
