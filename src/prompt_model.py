from langchain_openai import AzureChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from business_secrets import API_KEY
from langchain.schema import BaseOutputParser
import numpy as np

api_version = "2023-12-01-preview"
endpoint = "https://gpt-course.openai.azure.com/"

deployment_name = "gpt-4"

agent = AzureChatOpenAI(
    api_version=api_version, 
    azure_endpoint=endpoint,
    api_key=API_KEY, 
    deployment_name=deployment_name
)



template = """
Generate a new pokemon. Using the given information. A new requires the following information, generate the missing information. Make sure that the generated information is consistent with the other pieces of information. 
Name: The name of the pokemon. 

Type: This identifies the primary elemental attribute of the Pokémon, such as Electric, Water, Psychic, etc. Some Pokémon also have a secondary type, which is included if applicable. The type determines strengths, weaknesses, and the effectiveness of the Pokémon's moves against opponents in battles.

Description: This section provides a brief narrative about the Pokémon, including its evolutionary information, its role or significance within the Pokémon world, and any notable characteristics or abilities it possesses. This helps in understanding the Pokémon's background and its capabilities.
Real World Comparison: This is a one-word tag that draws a parallel between the Pokémon and a real-world animal or item. It helps to quickly associate the Pokémon's appearance or characteristics with something familiar, making it easier for someone unfamiliar with Pokémon to visualize and understand the creature.
Visual Description: This detailed description focuses on the physical appearance of the Pokémon. It includes information about the Pokémon's color, body shape, and any distinctive features. This is particularly useful for artists or anyone trying to imagine or create a visual representation of the Pokémon. Should only contain keywords.
Sound Prompt: Generate distinctive calls for a set of Pokémon, drawing inspiration from real-world counterparts and their characteristics. Each call should reflect the essence and unique traits of the respective Pokémon, evoking their personality and abilities. Consider real-world sounds associated with animals or elements closely related to each Pokémon's design and behavior. Ensure that the generated calls capture the imagination and essence of these iconic creatures.

Examples of already existing pokemon. Use | to separate the categories for the output formating.
Name: Squirtle
|Type: Water
|Description: Squirtle is a Water-type Pokémon that resembles a small turtle. It can evolve into Wartortle and eventually into Blastoise, gaining more power and bulk with each evolution.
|Real World Comparison: Turtle
|Visual Description Keywords: smooth blue skin, cream underside, dark blue shell, large eyes, smiling mouth, small curly tail
|Sound Prompt: Take inspiration from the high-pitched, gentle hiss and bubbly bursts of a turtle. Create a call that reflects playfulness and energy.

Name: Mewtwo
|Type: Psychic
|Description: Mewtwo is a legendary Psychic-type Pokémon created from the DNA of Mew, one of the rarest Pokémon. It is known for its incredible psychic abilities and is often regarded as one of the most powerful Pokémon.
|Real World Comparison: Feline
|Visual Description Keywords: sleek humanoid body, pale purple fur, bright purple eyes, long thin tail, muscular, streamlined
|Sound Prompt: Imagine a sound reminiscent of a powerful psychic force. Incorporate elements of deep resonance and otherworldly echoes to capture Mewtwo's enigmatic nature.

Name: Blastoise
|Type: Water
|Description: Blastoise is the final evolutionary stage of Squirtle, following Wartortle. Known for the powerful water cannons on its back, Blastoise can shoot water with enough force to penetrate steel.
|Real World Comparison: Tortoise
|Visual Description Keywords: large bipedal turtle, massive blue shell, water cannons, rugged blue body, small brown eyes, serious expression
|Signature call: Draw from the booming sound of a water cannon or the deep rumble of a large reptile. Emphasize strength and authority in the call.

Name: Treecko
|Type: Grass
|Description: Treecko, a Grass-type Pokémon, is known for its cool demeanor and is the first stage in its evolutionary line, leading to Grovyle and then Sceptile. It has abilities that allow it to scale vertical walls.
|Real World Comparison: Gecko
|Visual Description Keywords: small sleek, green, light underside, long curled tail, large yellow eyes, big feet
|Sound Prompt: Think about the chirps and chatters of agile lizards in the wild. Craft a call that evokes agility and nimbleness, perhaps with a touch of curiosity.

Name: Rayquaza
|Type: Dragon / Flying
|Description: Rayquaza is a legendary Pokémon that is part of the weather trio, along with Kyogre and Groudon. It has the ability to calm the other two members of the trio. It is known for living in the ozone layer and rarely descending to the ground.
|Real World Comparison: Serpent
|Visual Description Keywords: long serpentine body, green scales, yellow patterns, sharp red eyes, fins along body, majestic, fearsome.
|Sound Prompt: Consider the majestic roar of a dragon or the howl of a mighty wind. Create a call that echoes the grandeur and power of this legendary Pokémon.

Please define the pokemon from the following user provided input: {User_description}. If blank, make up a new pokemon.



IMPORTANT! Make sure the generated information is consistent with all the input information such as the types and real world comparison.
"""

template = PromptTemplate(
    template=template,
    input_variables=["User_description"], 
)

from langchain.schema import BaseOutputParser
class CommaSeparatedParser(BaseOutputParser):
    def parse(self, text):
        output = text.strip().split('|')
        output = [o.strip() for o in output]
        return output
    


def generate_prompt(prompt):
    chain = LLMChain(
        llm=agent,
        prompt=template,
        output_parser=CommaSeparatedParser(),
        verbose=True)
    
    suggestion = chain.run(User_description = prompt)
    Name = suggestion[0].split(":")[1].strip()
    Type = suggestion[1].split(":")[1].strip()
    Description = suggestion[2].split(":")[1].strip()
    Real_World_Comparison = suggestion[3].split(":")[1].strip()
    Visual_Description = suggestion[4].split(":")[1].strip()
    Signature_call = suggestion[5].split(":")[1].strip()

    del chain

    return Name, Type, Description, Real_World_Comparison, Visual_Description, Signature_call
