"""Define the agents used in the story generation system with improved context management"""
from typing import Dict, List, Optional

import autogen


class StoryAgents:
    def __init__(self, agent_config: Dict, outline: Optional[List[Dict]] = None):
        """Initialize agents with book outline context"""
        self.agent_config = agent_config
        self.outline = outline
        self.world_elements = {}  # Track described locations/elements
        self.created_stories = {}  # Track stories arcs
        
    def _format_outline_context(self) -> str:
        """Format the book outline into a readable context"""
        if not self.outline:
            return ""
            
        context_parts = ["Complete Book Outline:"]
        for story in self.outline:
            context_parts.extend([
                f"\nStory {story['story_number']}: {story['title']}",
                story['prompt']
            ])
        return "\n".join(context_parts)

    def create_agents(self, initial_prompt, num_stories) -> Dict:
        """Create and return all agents needed for stories generation"""
        outline_context = self._format_outline_context()
        
        # Memory Keeper: Maintain the richness of these story series, the independence of each story, and the absence of repetitive content.
        memory_keeper = autogen.AssistantAgent(
            name="memory_keeper",
            system_message=f"""You are the keeper of the richness of these story series,the independence of each story, and the absence of repetitive content.
            Your responsibilities:
            1. Track and summarize each story's key events
            2. Keep each story's the independence and the absence of repetitive content
            3. Maintain world-building consistency
            4. Flag any continuity issues
            5. Please write the story in Chinese
            6. The stories must be independent of each other, and the content must not be repeated
            7. There is no character dialogue in the story, only narration.

            story Overview:
            {outline_context}
            
            Format your responses as follows:
            - Start updates with 'MEMORY UPDATE:'
            - List key events with 'EVENT:'
            - List all created stories with 'STORY:'
            - List world details with 'WORLD:'
            - Flag issues with 'CONTINUITY ALERT:'""",
            llm_config=self.agent_config,
        )
        
        # Story Planner - Focuses on high-level story structure
        story_planner = autogen.AssistantAgent(
            name="story_planner",
            system_message=f"""You are an expert story arc planner focused on overall narrative structure.

            Your sole responsibility is creating the high-level story arc.
            When given an initial story premise:
            1. Identify major plot points and story beats
            2. Map each story arcs's independence and uniqueness
            3. Please write the story in Chinese
            4. The stories must be independent of each other, and the content must not be repeated
            5. There is no character dialogue in the story, only narration.

            Format your output EXACTLY as:
            STORY_ARC:
            - Major Plot Points:
            [List each major event that drives the story]
                        
            - Story Beats:
            [List key emotional and narrative moments in sequence]
                        
            Always provide specific, detailed content and completion.""",
            llm_config=self.agent_config,
        )

        # Outline Creator - Creates detailed story outlines
        outline_creator = autogen.AssistantAgent(
            name="outline_creator",
            system_message=f"""Generate a detailed {num_stories}-story outline.

            YOU MUST USE EXACTLY THIS FORMAT FOR EACH STORY - NO DEVIATIONS:

            Story 1: [Title]
            Stroy Title: [Same title as above]
            Key Events:
            - [Event 1]
            - [Event 2]
            - [Event 3]
            Setting: [Specific location and atmosphere]
            Tone: [Specific emotional and narrative tone]

            [REPEAT THIS EXACT FORMAT FOR ALL {num_stories} Stories]

            Requirements:
            1. EVERY field must be present for EVERY story
            2. EVERY story must have AT LEAST 3 specific Key Events
            3. ALL stories must be detailed,completion and unique
            4. Format must match EXACTLY - including all headings and bullet points
            5. Please write the story in Chinese
            6. The stories must be independent of each other, and the content must not be repeated
            7. There is no character dialogue in the story, only narration.

            Initial Premise:
            {initial_prompt}

            START WITH 'OUTLINE:' AND END WITH 'END OF OUTLINE'
            """,
            llm_config=self.agent_config,
        )

        # World Builder: Creates and maintains the story setting
        world_builder = autogen.AssistantAgent(
            name="world_builder",
            system_message=f"""You are an expert in world-building who creates rich, consistent settings.
            
            Your role is to establish ALL settings and locations needed for the entire story based on a provided story arc.

            All stories Overview:
            {outline_context}
            
            Your responsibilities:
            1. Review the series of stories arc to identify every location and setting needed
            2. Create detailed descriptions for each setting, including:
            - Physical layout and appearance
            - Atmosphere and environmental details
            - Important objects or features
            - Sensory details (sights, sounds, smells)
            3. Identify recurring locations that appear multiple times
            4. Note how settings might change over time
            5. Create a cohesive world that supports the story's themes
            6. Please write the story in Chinese
            7. The stories must be independent of each other, and the content must not be repeated
            8. There is no character dialogue in the story, only narration.
            
            Format your response as:
            WORLD_ELEMENTS:
            
            [LOCATION NAME]:
            - Physical Description: [detailed description]
            - Atmosphere: [mood, time of day, lighting, etc.]
            - Key Features: [important objects, layout elements]
            - Sensory Details: [what characters would experience]
            
            [RECURRING ELEMENTS]:
            - List any settings that appear multiple times
            - Note any changes to settings over time
            """,
            llm_config=self.agent_config,
        )

        # Writer: Generates the actual prose
        writer = autogen.AssistantAgent(
            name="writer",
            system_message=f"""You are an expert creative writer who brings scenes to life.
            
            Book Context:
            {outline_context}
            
            Your focus:
            1. Write according to the outlined plot points
            2. Maintain the richness of these story series, the independence of each story, and the absence of repetitive content.
            3. Incorporate world-building details
            4. Create engaging prose
            5. Please make sure that you write the complete scene, do not leave it incomplete
            6. Each story MUST be at least 1000 words (approximately 6,000 characters). Consider this a hard requirement. If your output is shorter, continue writing until you reach this minimum length
            8. Do not cut off the scene, make sure it has a proper ending
            9. Add a lot of details, and describe the environment and storis where it makes sense
            10. Please write the story in Chinese
            11. The stories must be independent of each other, and the content must not be repeated
            12. There is no character dialogue in the story, only narration.
            
            Always reference the outline and previous content.
            Mark drafts with 'SCENE:' and final versions with 'SCENE FINAL:'""",
            llm_config=self.agent_config,
        )

        # Editor: Reviews and improves content
        editor = autogen.AssistantAgent(
            name="editor",
            system_message=f"""You are an expert editor ensuring quality and consistency.
            
            Book Overview:
            {outline_context}
            
            Your focus:
            1. Check alignment with outline
            2. Verify the richness of these story series, the independence of each story, and the absence of repetitive content.
            3. Maintain world-building rules
            4. Improve prose quality
            5. Return complete edited stories
            6. Each story MUST be at least 800 words. If the content is shorter, return it to the writer for expansion. This is a hard requirement - do not approve story shorter than 1200 words
            7. Please write the story in Chinese
            8. The stories must be independent of each other, and the content must not be repeated
            9. There is no character dialogue in the story, only narration.

            Format your responses:
            1. Start critiques with 'FEEDBACK:'
            2. Provide suggestions with 'SUGGEST:'
            3. Return full edited stories with 'EDITED_SCENE:'
            
            Reference specific outline elements in your feedback.""",
            llm_config=self.agent_config,
        )

        # User Proxy: Manages the interaction
        user_proxy = autogen.UserProxyAgent(
            name="user_proxy",
            human_input_mode="TERMINATE",
            code_execution_config={
                "work_dir": "book_output",
                "use_docker": False
            }
        )

        return {
            "story_planner": story_planner,
            "world_builder": world_builder,
            "memory_keeper": memory_keeper,
            "writer": writer,
            "editor": editor,
            "user_proxy": user_proxy,
            "outline_creator": outline_creator
        }

    def update_world_element(self, element_name: str, description: str) -> None:
        """Track a new or updated world element"""
        self.world_elements[element_name] = description

    def update_character_development(self, character_name: str, development: str) -> None:
        """Track character development"""
        if character_name not in self.character_developments:
            self.character_developments[character_name] = []
        self.character_developments[character_name].append(development)

    def get_world_context(self) -> str:
        """Get formatted world-building context"""
        if not self.world_elements:
            return "No established world elements yet."
        
        return "\n".join([
            "Established World Elements:",
            *[f"- {name}: {desc}" for name, desc in self.world_elements.items()]
        ])

    def get_created_stories_context(self) -> str:
        """Get formatted created story  context"""
        if not self.created_stories:
            return "No story tracked yet."
        
        return "\n".join([
            "created stories:",
            *[f"- {name}:\n  " + "\n  ".join(story) 
              for name, story in self.created_stories.items()]
        ])