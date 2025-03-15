"""Main script for running the book generation system"""
from dotenv import load_dotenv

from agents import StoryAgents
from config import get_config
from outline_generator import OutlineGenerator
from story_generator import StoryGenerator

load_dotenv()

def main():
    # Get configuration
    agent_config = get_config()

    
    # Initial prompt for the book
    initial_prompt = """
    根据下面提供的时间(When)，地点(Where)，人物(Who)，事件(What)等因素组合起来，写出一个系列的符合现实，安防摄像头视角的的故事，这些故事适合作为一个个小的剧本，快速拍成视频。

参考发生的时间(When)：
    黄昏
    凌晨

事件发生的主要可以选择的地点(Where)：
    室内/室内封闭区域
    院子封闭区域
    室外重点开放区域
    公共区域
    厨房
    客厅
    卧室
    厕所

可能出现的人物(Who):
    穿着亚马逊制服的快递员
    穿着制服的快递员
    穿着普通衣服的快递员
    从亚马逊快递车走下来的快递员
    从UPS快递车走下来的快递员
    从FedEx快递车走下来的快递员
    从DHL快递车走下来的快递员
    从USPS快递车走下来的快递员
    从货车走下来的快递员
    从轿车走下来的快递员

    身份未知的人
    带着口罩的人
    穿着连衣帽的人
    带着帽子的人
    打着伞的人
    用手捂住面部的人
    带走偷窥的人


可以参考的主要事件或者一些画面或者镜头(What)：
    发生肢体冲突
    暴力踢踹某人
    劫持某人
    拖拉/拖拽某人

    躺在地上不起来
    趴在地上不起来
    倒地不起

    拿走割草机
    扛走割草机
    拖走割草机
    搬走割草机
    拿走笔记本电脑



这个我列举一个故事的大致结构:

开端： 在黄昏时分，一位快递员持他穿着亚马逊制服的那位——从送货车辆中走出，来给他的一位雇主送快递。

初始冲突： 这位快递员看到在院子里，一位貌似房东的人和一些戴口罩、穿连帽衫或遮住面部的人发生肢体冲突。

高潮： 一些戴口罩、穿连帽衫或遮住面部的人暴力踢踹那个房东，直到躺在地上起不来。那些人拿走笔记本电脑。

结局： 最后剩下的那个房东躺在院子里，留下一篇狼藉的打斗场景。


叙事风格:

从上述的给定的时间(When)，地点(Where)，人物(Who)，事件(What)的因素中，挑选出来一个个因素编制成一个个符合现实世界场景的故事。文风在描述动作时直接而技术化。叙事特别关注时间的推进，并融入详尽的技术观察。
没有人物对话，就是从安防摄像头的角度阐述一个个故事。故事的叙述者是一个安防摄像头，目睹了这些事件的发生。叙述者的视角是客观的，专注于事件的细节和技术观察，而不是情感或主观判断。
叙述者不会对事件做出情感上的反应，而是专注于描述事件的细节和技术观察。叙述者会使用专业术语和技术语言来描述事件，以便让读者能够理解事件的发生和发展。
故事中没有人物对话，只是对故事的叙述。每个故事相互独立，绝对没有重复。但是尽可能使用上述提到的这些素材。
    """

    num_stories = 10
    # Create agents
    story_outline_Agents = StoryAgents(agent_config)
    agents = story_outline_Agents.create_agents(initial_prompt, num_stories)
    
    # Generate the outline
    outline_gen = OutlineGenerator(agents, agent_config)
    print("Generating stories outline...")
    outline = outline_gen.generate_outline(initial_prompt, num_stories)
    
    # Create new agents with outline context
    story_agents = StoryAgents(agent_config, outline)
    agents_with_context = story_agents.create_agents(initial_prompt, num_stories)
    
    # Initialize story generator with contextual agents
    story_gen = StoryGenerator(agents_with_context, agent_config, outline)
    
    # Print the generated outline
    print("\nGenerated Outline:")
    for story in outline:
        print(f"\Story {story['story_number']}: {story['title']}")
        print("-" * 50)
        print(story['prompt'])
    
    # Save the outline for reference
    print("\nSaving outline to file...")
    with open("story_output/outline.txt", "w") as f:
        for story in outline:
            f.write(f"\nStory {story['story_number']}: {story['title']}\n")
            f.write("-" * 50 + "\n")
            f.write(story['prompt'] + "\n")
    
    # Generate the stories using the outline
    print("\nGenerating storie...")
    if outline:
        story_gen.generate_stories(outline)
    else:
        print("Error: No outline was generated.")

if __name__ == "__main__":
    main()