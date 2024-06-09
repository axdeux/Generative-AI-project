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
Visual Description: This detailed description focuses on the physical appearance of the Pokémon. It includes information about the Pokémon's color, body shape, and any distinctive features. This is particularly useful for artists or anyone trying to imagine or create a visual representation of the Pokémon.

Signature Cry: Every Pokémon has a unique sound or cry that it makes, which can be heard in the games or the anime. This section describes the nature of that sound, providing insight into the personality or elemental nature of the Pokémon. The description of the cry often includes auditory elements that suggest how it might sound, which adds to the immersive experience of understanding the Pokémon.

Examples of already existing pokemon. Use | to separate the categories for the output formating.
Name: Squirtle
|Type: Water
|Description: Squirtle is a Water-type Pokémon that resembles a small turtle. It can evolve into Wartortle and eventually into Blastoise, gaining more power and bulk with each evolution.
|Real World Comparison: Turtle
|Visual Description: Squirtle has a smooth, blue skin with a cream-colored underside and a shell that is a darker shade of blue. Its eyes are large and it has a small, smiling mouth, giving it a friendly appearance. The tail is small and curly.
|Signature Cry: Squirtle's cry is a bubbly and watery sound, reflecting its aquatic nature, typically a soft trilling that is soothing.


Name: Mewtwo
|Type: Psychic
|Description: Mewtwo is a legendary Psychic-type Pokémon created from the DNA of Mew, one of the rarest Pokémon. It is known for its incredible psychic abilities and is often regarded as one of the most powerful Pokémon.
|Real World Comparison: Feline
|Visual Description: Mewtwo stands upright and has a sleek, humanoid body covered in pale purple fur. Its eyes are bright purple, and it has a long, thin tail. Its structure is muscular yet streamlined for agility.
|Signature Cry: Mewtwo's cry is deep and telepathic-sounding, often resonating with a metallic echo, reflecting its mysterious and formidable nature.

Name: Blastoise
|Type: Water
|Description: Blastoise is the final evolutionary stage of Squirtle, following Wartortle. Known for the powerful water cannons on its back, Blastoise can shoot water with enough force to penetrate steel.
|Real World Comparison: Tortoise
|Visual Description: Blastoise is a large, bipedal turtle with a massive, blue shell featuring two powerful water cannons. It has a rugged, blue body, small brown eyes, and a serious expression on its face.
|Signature Cry: Blastoise has a deep, roaring cry, similar to the sound of a cannon firing, which is intimidating and powerful.

Name: Treecko
|Type: Grass
|Description: Treecko, a Grass-type Pokémon, is known for its cool demeanor and is the first stage in its evolutionary line, leading to Grovyle and then Sceptile. It has abilities that allow it to scale vertical walls.
|Real World Comparison: Gecko
|Visual Description: Treecko is small and sleek, primarily green with a light underskin and a long, curled tail. It has large, yellow eyes and feet that are big compared to its body, adapted for climbing.
|Signature Cry: Treecko's cry is sharp and quick, similar to a chirp, which reflects its alert and agile nature.

Name: Rayquaza
|Type: Dragon / Flying
|Description: Rayquaza is a legendary Pokémon that is part of the weather trio, along with Kyogre and Groudon. It has the ability to calm the other two members of the trio. It is known for living in the ozone layer and rarely descending to the ground.
|Real World Comparison: Serpent
|Visual Description: Rayquaza is a long, serpentine creature with a green body and yellow patterns that run along its length. It has an imposing presence with sharp, red eyes and fins along its body that give it a majestic and fearsome look.
|Signature Cry: Rayquaza's cry is a majestic and echoing roar, powerful enough to be heard across great distances, resonating through the skies.

Please define the pokemon from the following user provided input: {User_description}



IMPORTANT! Make sure the generated information is consistent with all the input information such as the types and real world comparison.
"""
#Kan legge til flere eksempler for å forbedre formateringen.


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
    



# first_suggestion = chain.run(name = "Empty", type = "fire", secondary_type = "religious", tag = "The popemobile")


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
    Signature_Cry = suggestion[5].split(":")[1].strip()

    del chain

    return Name, Type, Description, Real_World_Comparison, Visual_Description, Signature_Cry