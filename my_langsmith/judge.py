from my_langsmith.config import ls_client, groq_client

def helpfulness(inputs:dict, outputs:dict, reference_outputs:dict) -> float:
    """A function that scores the helpfulness of the agent"""
    question = inputs.get('question')
    if question is None and 'inputs' in inputs:
        question = inputs['inputs'].get('question')
    response = outputs.get("response")
    if response is None and "output" in outputs:
        resposne = outputs['output'].get("response")
    answer = reference_outputs.get("answer")
    if answer is None and "output" in reference_outputs:
        answer = reference_outputs["output"].get("answer")

    user_content = f"""You will score the helpfulness of the given answer on a scale from 1-5.
    you will be given the question, the actual answer, and the predicted answer. Respond with only one number: the helpfulness score.
    Question: {question}.
    Actual answer: {answer}
    Predicted answer: {response}
    Score: 
    """
    score = groq_client.chat.completions.create(
    model="openai/gpt-oss-20b",
    messages=[
        {
            "role": "user",
            "content": user_content
        }
    ]
).choices[0].message.content.strip()
    try:
        return float(score)
    except ValueError:
        return 0.0


def correctness(inputs: dict, outputs:dict, reference_outputs:dict) :

    question = inputs.get('question')
    if question is None and 'inputs' in inputs:
        question = inputs['inputs'].get('question')
    response = outputs.get("response")
    if response is None and "output" in outputs:
        resposne = outputs['output'].get("response")
    answer = reference_outputs.get("answer")
    if answer is None and "output" in reference_outputs:
        answer = reference_outputs["output"].get("answer")

    user_content = f"""You are an expert data labeler evaluating model outputs for correctness. Your task is to assign a score based on the following rubric:

<Rubric>
  A correct answer:
  - Provides accurate and complete information
  - Contains no factual errors
  - Addresses all parts of the question
  - Is logically consistent
  - Uses precise and accurate terminology

  When scoring, you should penalize:
  - Factual errors or inaccuracies
  - Incomplete or partial answers
  - Misleading or ambiguous statements
  - Incorrect terminology
  - Logical inconsistencies
  - Missing key information
</Rubric>

<Instructions>
  - Carefully read the input and output
  - Check for factual accuracy and completeness
  - Focus on correctness of information rather than style or verbosity
</Instructions>

<Reminder>
  The goal is to evaluate factual correctness and completeness of the response.
</Reminder>

<input>
{question}
</input>

<output>
{response}
</output>

Use the reference outputs below to help you evaluate the correctness of the response:

<reference_outputs>
{answer}
</reference_outputs>
"""
    score = groq_client.chat.completions.create(
        model='openai/gpt-oss-20b',
        messages= [{
            'role':'user','content':user_content,
        }],
    ).choices[0].message.content.strip()
    return score


    
    
