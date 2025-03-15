"""Main class for generating stories using AutoGen with improved iteration control"""
import os
import re
import time
from typing import Dict, List, Optional

import autogen


class StoryGenerator:
    def __init__(self, agents: Dict[str, autogen.ConversableAgent], agent_config: Dict, outline: List[Dict]):
        """Initialize with outline to maintain story count context"""
        self.agents = agents
        self.agent_config = agent_config
        self.output_dir = "story_output"
        self.stories_memory = []  # Store stories summaries
        self.max_iterations = 3  # Limit editor-writer iterations
        self.outline = outline  # Store the outline
        os.makedirs(self.output_dir, exist_ok=True)

    def _clean_story_content(self, content: str) -> str:
        """Clean up story content by removing artifacts and story numbers"""
        # Remove story number references
        content = re.sub(r'\*?\s*\(Story \d+.*?\)', '', content)
        content = re.sub(r'\*?\s*Story \d+.*?\n', '', content, count=1)
        
        # Clean up any remaining markdown artifacts
        content = content.replace('*', '')
        content = content.strip()
        
        return content
    
    def initiate_group_chat(self) -> autogen.GroupChat:
        """Create a new group chat for the agents with improved speaking order"""
        outline_context = "\n".join([
            f"\nStory {ch['story_number']}: {ch['title']}\n{ch['prompt']}"
            for ch in sorted(self.outline, key=lambda x: x['story_number'])
        ])

        messages = [{
            "role": "system",
            "content": f"Complete Book Outline:\n{outline_context}"
        }]

        writer_final = autogen.AssistantAgent(
            name="writer_final",
            system_message=self.agents["writer"].system_message,
            llm_config=self.agent_config
        )
        
        return autogen.GroupChat(
            agents=[
                self.agents["user_proxy"],
                self.agents["memory_keeper"],
                self.agents["writer"],
                self.agents["editor"],
                writer_final
            ],
            messages=messages,
            max_round=5,
            speaker_selection_method="round_robin"
        )

    def _get_sender(self, msg: Dict) -> str:
        """Helper to get sender from message regardless of format"""
        return msg.get("sender") or msg.get("name", "")

    def _verify_story_complete(self, messages: List[Dict]) -> bool:
        """Verify story completion by analyzing entire conversation context"""
        print("******************** VERIFYING STORY COMPLETION ****************")
        current_story = None
        story_content = None
        sequence_complete = {
            'memory_update': False,
            'plan': False,
            'setting': False,
            'scene': False,
            'feedback': False,
            'scene_final': False,
            'confirmation': False
        }
        
        # Analyze full conversation
        for msg in messages:
            content = msg.get("content", "")
            
            # Track story number
            if not current_story:
                num_match = re.search(r"Story (\d+):", content)
                if num_match:
                    current_story = int(num_match.group(1))
            
            # Track completion sequence
            if "MEMORY UPDATE:" in content: sequence_complete['memory_update'] = True
            if "PLAN:" in content: sequence_complete['plan'] = True
            if "SETTING:" in content: sequence_complete['setting'] = True
            if "SCENE:" in content: sequence_complete['scene'] = True
            if "FEEDBACK:" in content: sequence_complete['feedback'] = True
            if "SCENE FINAL:" in content:
                sequence_complete['scene_final'] = True
                story_content = content.split("SCENE FINAL:")[1].strip()
            if "**Confirmation:**" in content and "successfully" in content:
                sequence_complete['confirmation'] = True

            #print all sequence_complete flags
            print("******************** SEQUENCE COMPLETE **************", sequence_complete)
            print("******************** CURRENT_STORY ****************", current_story)
            print("******************** STORY_CONTENT ****************", story_content)
        
        # Verify all steps completed and content exists
        if all(sequence_complete.values()) and current_story and story_content:
            self._save_story(current_story, story_content)
            return True
            
        return False
    
    def _prepare_story_context(self, story_number: int, prompt: str) -> str:
        """Prepare context for story generation"""
        if story_number == 1:
            return f"Initial Story\nRequirements:\n{prompt}"
            
        context_parts = [
            "Previous Story Summaries:",
            *[f"Story {i+1}: {summary}" for i, summary in enumerate(self.stories_memory)],
            "\nCurrent Story Requirements:",
            prompt
        ]
        return "\n".join(context_parts)

    def generate_story(self, story_number: int, prompt: str) -> None:
        """Generate a single story with completion verification"""
        print(f"\nGenerating story {story_number}...")
        
        try:
            # Create group chat with reduced rounds
            groupchat = self.initiate_group_chat()
            manager = autogen.GroupChatManager(
                groupchat=groupchat,
                llm_config=self.agent_config
            )

            # Prepare context
            context = self._prepare_story_context(story_number, prompt)
            story_prompt = f"""
            IMPORTANT: Wait for confirmation before proceeding.
            IMPORTANT: This is Story {story_number}. Do not proceed to next story until explicitly instructed.
            DO  END THE STORY HERE.

            Current Task: Generate Story {story_number} content only.

            Story Outline:
            Title: {self.outline[story_number - 1]['title']}

            Story Requirements:
            {prompt}

            Previous Context for Reference:
            {context}

            Follow this exact sequence for Story {story_number} only:

            1. Memory Keeper: Context (MEMORY UPDATE)
            2. Writer: Draft (STORY)
            3. Editor: Review (FEEDBACK)
            4. Writer Final: Revision (STORY FINAL)

            Wait for each step to complete before proceeding."""

            # Start generation
            self.agents["user_proxy"].initiate_chat(
                manager,
                message=story_prompt
            )

            if not self._verify_story_complete(groupchat.messages):
                raise ValueError(f"Story {story_prompt} generation incomplete")
        
            self._process_story_results(story_number, groupchat.messages)
            story_file = os.path.join(self.output_dir, f"story_{story_number:02d}.txt")
            if not os.path.exists(story_file):
                raise FileNotFoundError(f"Story {story_number} file not created")
        
            completion_msg = f"Story {story_number} is complete. Proceed with next story."
            self.agents["user_proxy"].send(completion_msg, manager)
            
        except Exception as e:
            print(f"Error in story {story_number}: {str(e)}")
            self._handle_story_generation_failure(story_number, prompt)

    def _extract_final_scene(self, messages: List[Dict]) -> Optional[str]:
        """Extract story content with improved content detection"""
        for msg in reversed(messages):
            content = msg.get("content", "")
            sender = self._get_sender(msg)
            
            if sender in ["writer", "writer_final"]:
                # Handle complete scene content
                if "SCENE FINAL:" in content:
                    scene_text = content.split("SCENE FINAL:")[1].strip()
                    if scene_text:
                        return scene_text
                        
                # Fallback to scene content
                if "SCENE:" in content:
                    scene_text = content.split("SCENE:")[1].strip()
                    if scene_text:
                        return scene_text
                        
                # Handle raw content
                if len(content.strip()) > 100:  # Minimum content threshold
                    return content.strip()
                    
        return None

    def _handle_story_generation_failure(self, story_number: int, prompt: str) -> None:
        """Handle failed story generation with simplified retry"""
        print(f"Attempting simplified retry for Story {story_number}...")
        
        try:
            # Create a new group chat with just essential agents
            retry_groupchat = autogen.GroupChat(
                agents=[
                    self.agents["user_proxy"],
                    self.agents["story_planner"],
                    self.agents["writer"]
                ],
                messages=[],
                max_round=3
            )
            
            manager = autogen.GroupChatManager(
                groupchat=retry_groupchat,
                llm_config=self.agent_config
            )

            retry_prompt = f"""Emergency story generation for Story {story_number}.
            
{prompt}

Please generate this story in two steps:
1. Story Planner: Create a basic outline (tag: PLAN)
2. Writer: Write the complete story (tag: SCENE FINAL)

Keep it simple and direct."""

            self.agents["user_proxy"].initiate_chat(
                manager,
                message=retry_prompt
            )
            
            # Save the retry results
            self._process_story_results(story_number, retry_groupchat.messages)
            
        except Exception as e:
            print(f"Error in retry attempt for Story {story_number}: {str(e)}")
            print("Unable to generate story content after retry")

    def _process_story_results(self, story_number: int, messages: List[Dict]) -> None:
        """Process and save story results, updating memory"""
        try:
            # Extract the Memory Keeper's final summary
            memory_updates = []
            for msg in reversed(messages):
                sender = self._get_sender(msg)
                content = msg.get("content", "")
                
                if sender == "memory_keeper" and "MEMORY UPDATE:" in content:
                    update_start = content.find("MEMORY UPDATE:") + 14
                    memory_updates.append(content[update_start:].strip())
                    break
            
            # Add to memory even if no explicit update (use basic content summary)
            if memory_updates:
                self.stories_memory.append(memory_updates[0])
            else:
                # Create basic memory from story content
                story_content = self._extract_final_scene(messages)
                if story_content:
                    basic_summary = f"Story {story_number} Summary: {story_content[:200]}..."
                    self.stories_memory.append(basic_summary)
            
            # Extract and save the story content
            self._save_story(story_number, messages)
            
        except Exception as e:
            print(f"Error processing story results: {str(e)}")
            raise

    def _save_story(self, story_number: int, messages: List[Dict]) -> None:
        print(f"\nSaving Story {story_number}")
        try:
            story_content = self._extract_final_scene(messages)
            if not story_content:
                raise ValueError(f"No content found for Story {story_content}")
                
            story_content = self._clean_story_content(story_content)
            
            filename = os.path.join(self.output_dir, f"story_{story_number:02d}.txt")
            
            # Create backup if file exists
            if os.path.exists(filename):
                backup_filename = f"{filename}.backup"
                import shutil
                shutil.copy2(filename, backup_filename)
                
            with open(filename, "w", encoding='utf-8') as f:
                f.write(f"Story {story_number}\n\n{story_content}")
                
            # Verify file
            with open(filename, "r", encoding='utf-8') as f:
                saved_content = f.read()
                if len(saved_content.strip()) == 0:
                    raise IOError(f"File {filename} is empty")
                    
            print(f"✓ Saved to: {filename}")
            
        except Exception as e:
            print(f"Error saving story: {str(e)}")
            raise

    def generate_stories(self, outline: List[Dict]) -> None:
        """Generate the stories with strict story sequencing"""
        print("\nStarting story Generation...")
        print(f"Total stories: {len(outline)}")
        
        # Sort outline by story number
        sorted_outline = sorted(outline, key=lambda x: x["story_number"])
        
        for story in sorted_outline:
            story_number = story["story_number"]
            
            # Verify previous story exists and is valid
            if story_number > 1:
                prev_file = os.path.join(self.output_dir, f"story_{story_number-1:02d}.txt")
                if not os.path.exists(prev_file):
                    print(f"Previous story {story_number-1} not found. Stopping.")
                    break
                    
                # Verify previous story content
                with open(prev_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if not self._verify_story_content(content, story_number-1):
                        print(f"Previous story {story_number-1} content invalid. Stopping.")
                        break
            
            # Generate current story
            print(f"\n{'='*20} Story {story_number} {'='*20}")
            self.generate_story(story_number, story["prompt"])
            
            # Verify current story
            story_file = os.path.join(self.output_dir, f"story_{story_number:02d}.txt")
            if not os.path.exists(story_file):
                print(f"Failed to generate story {story_number}")
                break
                
            with open(story_file, 'r', encoding='utf-8') as f:
                content = f.read()
                if not self._verify_story_content(content, story_number):
                    print(f"Story {story_number} content invalid")
                    break
                    
            print(f"✓ Story {story_number} complete")
            time.sleep(5)

    def _verify_story_content(self, content: str, story_number: int) -> bool:
        """Verify story content is valid"""
        if not content:
            return False
            
        # Check for story header
        if f"Story {story_number}" not in content:
            return False
            
        # Ensure content isn't just metadata
        lines = content.split('\n')
        content_lines = [line for line in lines if line.strip() and 'MEMORY UPDATE:' not in line]
        
        return len(content_lines) >= 3  # At least story header + 2 content lines