from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()

selection_instructions = """
You are the SELECTION module in a Selection-Inference pipeline. You will be given 3 examples in this format: [premises (context), question, selection]. Then, you will be given the main premises (context) and question. Your task is to select the most relevant subset of the premises that might lead to the next most relevant premise, in addition to solving the final question. 
""" # in reality, it is 5

# selection_5_shot_template = """
# # 3-shot prompt
# # First example
# <premises>
# "If a Python code is well-tested, then the project is optimized.",
# "If a Python code does not follow PEP 8 standards, then it is not well-tested.",
# "All Python projects are easy to maintain.",
# "All Python code is well-tested.",
# "If a Python code follows PEP 8 standards, then it is easy to maintain.",
# "If a Python code is well-tested, then it follows PEP 8 standards.",
# "If a Python project is well-structured, then it is optimized.",
# "If a Python project is easy to maintain, then it is well-tested.",
# "If a Python project is optimized, then it has clean and readable code.",
# "All Python projects are well-structured.",
# "All Python projects have clean and readable code.",
# "There exists at least one Python project that follows best practices.",
# "There exists at least one Python project that is optimized.",
# "If a Python project is not well-structured, then it does not follow PEP 8 standards."
# <question>
# Does it follow that if all Python projects are well-structured, then all Python projects are optimized, according to the premises?
# <selection>
# We know that premise 10 says that all Python projects are well-structured, and premise 7 states that well-structured projects are optimized. Therefore, the answer is yes.

# # Second example
# <PREMISES>  
# 1. Students who have completed the core curriculum and passed the science assessment are qualified for advanced courses.  
# 2. Students who are qualified for advanced courses and have completed research methodology are eligible for the international program.  
# 3. Students who have passed the language proficiency exam are eligible for the international program.  
# 4. Students who are eligible for the international program and have completed a capstone project are awarded an honors diploma.  
# 5. Students who have been awarded an honors diploma and have completed community service qualify for the university scholarship.  
# 6. Students who have been awarded an honors diploma and have received a faculty recommendation qualify for the university scholarship.  
# 7. Sophia has completed the core curriculum.  
# 8. Sophia has passed the science assessment.  
# 9. Sophia has completed the research methodology course.  
# 10. Sophia has completed her capstone project.  
# 11. Sophia has completed the required community service hours.  
# <QUESTION>
# Does Sophia qualify for the university scholarship, according to the premises?
# <SELECTION>

# # Third example
# <PREMISES>
# 1. If a driver has passed vehicle inspection and has the appropriate license, they can transport standard goods.
# 2. If a driver can transport standard goods and has completed hazmat training and received a safety endorsement, they can transport hazardous materials.
# 3. If a driver can transport hazardous materials and has an interstate permit, they can cross state lines with hazardous cargo.
# 4. John has passed vehicle inspection.
# 5. John has the appropriate license.
# 6. John has completed hazmat training.
# 7. John has not received a safety endorsement.
# 8. John has an interstate permit.
# <QUESTION>
# Does John meet all requirements to cross state lines with hazardous cargo, according to the premises?
# <SELECTION>
# If a driver has passed vehicle inspection and has the appropriate license, they can transport standard goods (premise 1). We know that John has passed vehicle inspection (premise 4) and John has the appropriate license (premise 5). Therefore,
# """

selection_5_shot_template = """
Given a set of rules and facts, you have to reason whether a statement is true or false.

Here are some facts and rules:
Nice people are quiet.
If Dave is smart then Dave is nice.
All white people are smart.
Dave is smart.
Harry is cold.
Does it imply that the statement "Dave is not quiet" is true?
Reasoning: If Dave is smart then Dave is nice. We know that Dave is smart. Therefore,

Here are some facts and rules:
Blue things are green.
All blue things are white.
If Anne is not big then Anne is blue.
Big things are white.
All kind things are round.
If something is white and big then it is not kind.
If something is big and not rough then it is green.
If something is white and blue then it is not green.
Erin is not white.
Anne is big.
Bob is rough.
Anne is white.
Does it imply that the statement "Anne is kind" is true?
Reasoning: If something is white and big then it is not kind. We know that Anne is white and Anne is big. Therefore,

..."""
actual_question = """
Here are some facts and rules:
If something likes the squirrel and it is not young then it chases the lion.
If something likes the squirrel then it is rough.
If something chases the rabbit and the rabbit is not young then it chases the lion.
If something eats the lion then it is young.
If something likes the rabbit then it chases the rabbit.
All rough things are nice.
The rabbit is young.
The squirrel likes the rabbit.
The lion likes the squirrel.
Does it imply that the statement "The lion is not nice" is true?
Reasoning:"""
# ======================================================================= #

response = client.responses.create(
    model="gpt-4o-mini",
    instructions=selection_instructions,
    input=selection_5_shot_template + actual_question,
    temperature=0.3,
)

print(response.output_text)