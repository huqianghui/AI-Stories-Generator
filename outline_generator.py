"""Generate book outlines using AutoGen agents with improved error handling"""
import re
from typing import Dict, List

import autogen


class OutlineGenerator:
    def __init__(self, agents: Dict[str, autogen.ConversableAgent], agent_config: Dict):
        self.agents = agents
        self.agent_config = agent_config

    def generate_outline(self, initial_prompt: str, num_stories: int = 25) -> List[Dict]:
        """Generate a stories outline based on initial prompt"""
        print("\nGenerating outline...")

        
        groupchat = autogen.GroupChat(
            agents=[
                self.agents["user_proxy"],
                self.agents["story_planner"],
                self.agents["world_builder"],
                self.agents["outline_creator"]
            ],
            messages=[],
            max_round=4,
            speaker_selection_method="round_robin"
        )
        
        manager = autogen.GroupChatManager(groupchat=groupchat, llm_config=self.agent_config)

        outline_prompt = f"""Let's create a {num_stories}-story outline for a book with the following premise:

{initial_prompt}

Process:
1. Story Planner: Create a series high-level stories arc and major plot points
2. World Builder: Suggest key settings and world elements needed
3. Outline Creator: Generate a detailed outline with each stories titles and prompts. They are a series of stories that can also be read independently and different. 
4. Each story should have a clear beginning, middle, and end, with a focus on character development and plot progression.
5. Please write the story in Chinese
6. The stories must be independent of each other, and the content must not be repeated
7. There is no character dialogue in the story, only narration.

Start with story 1 and number stories sequentially.

Make sure there are at least 3 scenes in each story.

[Continue with remaining stories]

Please output all stories, do not leave out any stories. Think through every story carefully, none should be to be determined later
It is of utmost importance that you detail out every story.
There should be clear content for each story. There should be a total of {num_stories} stories of the series.

End the outline with 'END OF OUTLINE'"""

        try:
            # Initiate the chat
            self.agents["user_proxy"].initiate_chat(
                manager,
                message=outline_prompt
            )

            # Extract the outline from the chat messages
            return self._process_outline_results(groupchat.messages, num_stories)
            
        except Exception as e:
            print(f"Error generating outline: {str(e)}")
            # Try to salvage any outline content we can find
            return self._emergency_outline_processing(groupchat.messages, num_stories)

    def _get_sender(self, msg: Dict) -> str:
        """Helper to get sender from message regardless of format"""
        return msg.get("sender") or msg.get("name", "")

    def _extract_outline_content(self, messages: List[Dict]) -> str:
        """Extract outline content from messages with better error handling"""
        print("Searching for outline content in messages...")
        
        # Look for content between "OUTLINE:" and "END OF OUTLINE"
        for msg in reversed(messages):
            content = msg.get("content", "")
            if "OUTLINE:" in content:
                # Extract content between OUTLINE: and END OF OUTLINE
                start_idx = content.find("OUTLINE:")
                end_idx = content.find("END OF OUTLINE")
                
                if start_idx != -1:
                    if end_idx != -1:
                        return content[start_idx:end_idx].strip()
                    else:
                        # If no END OF OUTLINE marker, take everything after OUTLINE:
                        return content[start_idx:].strip()
                        
        # Fallback: look for content with story markers
        for msg in reversed(messages):
            content = msg.get("content", "")
            if "Story 1:" in content or "**Story 1:**" in content:
                return content

        return ""

    def _process_outline_results(self, messages: List[Dict], num_stories: int) -> List[Dict]:
        """Extract and process the outline with strict format requirements"""
        outline_content = self._extract_outline_content(messages)
        
        if not outline_content:
            print("No structured outline found, attempting emergency processing...")
            return self._emergency_outline_processing(messages, num_stories)

        stories = []
        story_sections = re.split(r'Story \d+:', outline_content)
        
        for i, section in enumerate(story_sections[1:], 1):  # Skip first empty section
            try:
                    # Extract required components
                title_match = re.search(r'\*?\*?Title:\*?\*?\s*(.+?)(?=\n|$)', section, re.IGNORECASE)
                events_match = re.search(r'\*?\*?Key Events:\*?\*?\s*(.*?)(?=\*?\*?Character Developments:|$)', section, re.DOTALL | re.IGNORECASE)
                setting_match = re.search(r'\*?\*?Setting:\*?\*?\s*(.*?)(?=\*?\*?Tone:|$)', section, re.DOTALL | re.IGNORECASE)
                tone_match = re.search(r'\*?\*?Tone:\*?\*?\s*(.*?)(?=\*?\*?Story \d+:|$)', section, re.DOTALL | re.IGNORECASE)

                # If no explicit title match, try to get it from the story header
                if not title_match:
                    title_match = re.search(r'\*?\*?Story \d+:\s*(.+?)(?=\n|$)', section)

                # Verify all components exist
                if not all([title_match, events_match, setting_match, tone_match]):
                    print(f"Missing required components in Story {i}")
                    missing = []
                    if not title_match: missing.append("Title")
                    if not events_match: missing.append("Key Events")
                    if not setting_match: missing.append("Setting")
                    if not tone_match: missing.append("Tone")
                    print(f"  Missing: {', '.join(missing)}")
                    continue

                # Format story content
                story_info = {
                    "story_number": i,
                    "title": title_match.group(1).strip(),
                    "prompt": "\n".join([
                        f"- Key Events: {events_match.group(1).strip()}",
                        f"- Setting: {setting_match.group(1).strip()}",
                        f"- Tone: {tone_match.group(1).strip()}"
                    ])
                }
                
                # Verify events (at least 3)
                events = re.findall(r'-\s*(.+?)(?=\n|$)', events_match.group(1))
                if len(events) < 3:
                    print(f"Story {i} has fewer than 3 events")
                    continue

                stories.append(story_info)

            except Exception as e:
                print(f"Error processing Story {i}: {str(e)}")
                continue

        # If we don't have enough valid stories, raise error to trigger retry
        if len(stories) < num_stories:
            raise ValueError(f"Only processed {len(stories)} valid stories out of {num_stories} required")

        return stories

    def _verify_story_sequence(self, stories: List[Dict], num_stories: int) -> List[Dict]:
        """Verify and fix story numbering"""
        # Sort stories by their current number
        stories.sort(key=lambda x: x['story_number'])
        
        # Renumber stories sequentially starting from 1
        for i, story in enumerate(stories, 1):
            story['story_number'] = i
        
        # Add placeholder stories if needed
        while len(stories) < num_stories:
            next_num = len(stories) + 1
            stories.append({
                'story_number': next_num,
                'title': f'Story {next_num}',
                'prompt': '- Key events: [To be determined]\n- Setting: [To be determined]\n- Tone: [To be determined]'
            })
        
        # Trim excess stories if needed
        stories = stories[:num_stories]
        
        return stories

    def _emergency_outline_processing(self, messages: List[Dict], num_stories: int) -> List[Dict]:
        """Emergency processing when normal outline extraction fails"""
        print("Attempting emergency outline processing...")
        
        stories = []
        current_story = None
        
        # Look through all messages for any story content
        for msg in messages:
            content = msg.get("content", "")
            lines = content.split('\n')
            
            for line in lines:
                # Look for story markers
                story_match = re.search(r'Story (\d+)', line)
                if story_match and "Key events:" in content:
                    if current_story:
                        stories.append(current_story)
                    
                    current_story = {
                        'story_number': int(story_match.group(1)),
                        'title': line.split(':')[-1].strip() if ':' in line else f"Story {story_match.group(1)}",
                        'prompt': []
                    }
                
                # Collect bullet points
                if current_story and line.strip().startswith('-'):
                    current_story['prompt'].append(line.strip())
            
            # Add the last story if it exists
            if current_story and current_story.get('prompt'):
                current_story['prompt'] = '\n'.join(current_story['prompt'])
                stories.append(current_story)
                current_story = None
        
        if not stories:
            print("Emergency processing failed to find any stories")
            # Create a basic outline structure
            stories = [
                {
                    'story_number': i,
                    'title': f'Story {i}',
                    'prompt': '- Key events: [To be determined]\n- Setting: [To be determined]\n- Tone: [To be determined]'
                }
                for i in range(1, num_stories + 1)
            ]
        
        # Ensure proper sequence and number of stories
        return self._verify_story_sequence(stories, num_stories)